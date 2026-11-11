// SPDX-License-Identifier: Apache-2.0
#include "kernel_dispatch.cuh"
#include "torch_context.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <climits>
#include <limits>
#include <optional>
#include <torch/csrc/stable/ops.h>

namespace {
// Stable error marker for unsupported upstream MoE inputs.
constexpr const char* kMoeNotEligibleMarker = "VLLM_GGUF_MOE_NOT_ELIGIBLE";

void check_moe_inputs(const Tensor& X, const Tensor& W, const Tensor& topk_ids,
                      int64_t type, int64_t row, int64_t top_k, int64_t tokens,
                      bool aligned = false) {
  STD_TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
                  kMoeNotEligibleMarker,
                  ": all "
                  "tensors must be CUDA tensors");
  STD_TORCH_CHECK(X.get_device_index() == W.get_device_index() &&
                      X.get_device_index() == topk_ids.get_device_index(),
                  kMoeNotEligibleMarker,
                  ": "
                  "tensors must be on the same CUDA device");
  STD_TORCH_CHECK(
      X.dim() == 2 && W.dim() == 3 && topk_ids.dim() == (aligned ? 1 : 2),
      kMoeNotEligibleMarker,
      ": "
      "expected X[ tokens, K ], W[ experts, rows, packed ], and "
      "route IDs with matching dimensions");
  STD_TORCH_CHECK(
      X.is_contiguous() && W.is_contiguous() && topk_ids.is_contiguous(),
      kMoeNotEligibleMarker,
      ": "
      "tensors must be contiguous");
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int,
                  kMoeNotEligibleMarker,
                  ": "
                  "topk_ids must be int32");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  kMoeNotEligibleMarker,
                  ": X "
                  "must have dtype fp32, fp16, or bf16");
  STD_TORCH_CHECK(
      float_type_matches(W, type) || (is_upstream_weight_type(type) &&
                                      W.scalar_type() == ScalarType::Byte),
      kMoeNotEligibleMarker,
      ": unsupported weight type or dtype mismatch: ", type);
  STD_TORCH_CHECK(tokens > 0 && top_k > 0 && row > 0, kMoeNotEligibleMarker,
                  ": "
                  "tokens, top_k, and row must be positive");
  STD_TORCH_CHECK(
      X.size(0) == tokens &&
          (aligned ? topk_ids.numel() >= tokens * top_k
                   : topk_ids.size(0) == tokens && topk_ids.size(1) == top_k),
      kMoeNotEligibleMarker,
      ": "
      "shape arguments do not match X/topk_ids");
  STD_TORCH_CHECK(W.size(0) > 0 && W.size(1) == row && W.size(2) > 0,
                  kMoeNotEligibleMarker,
                  ": "
                  "shape arguments do not match W");
  STD_TORCH_CHECK(top_k <= W.size(0), kMoeNotEligibleMarker,
                  ": top_k exceeds the number of experts");
}

int64_t logical_k_from_moe_weight(const Tensor& W, int64_t type,
                                  const char* op_name) {
  return logical_k_from_packed_row_bytes(W.size(2), type, op_name,
                                         "expert row");
}

int64_t checked_moe_product(int64_t lhs, int64_t rhs, const char* name) {
  STD_TORCH_CHECK(
      lhs > 0 && rhs > 0 && lhs <= std::numeric_limits<int64_t>::max() / rhs,
      kMoeNotEligibleMarker, ": ", name, " overflows int64");
  return lhs * rhs;
}

// Same aligned route plan is used for W1 and W2. Upstream gathers W1's
// token r/top_k and W2's routed row r; padded entries gather a safe row.
__global__ void aligned_source_rows(const int32_t* ids, const int32_t* count,
                                    int32_t* src, int capacity, int routes,
                                    int top_k) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < capacity) {
    const int r = p < *count ? ids[p] : routes;
    src[p] = r >= 0 && r < routes ? r / top_k : 0;
  }
}

template <ggml_type type>
__launch_bounds__(
    ggml_cuda_mmq_get_nthreads(type, gguf_dispatch::kAlignedRouteTile, true),
    ggml_cuda_mmq_get_occupancy(type, gguf_dispatch::kAlignedRouteTile,
                                true)) __global__
    void aligned_mul_mat_q(const char* x, const int* y, const int32_t* ids,
                           const int32_t* experts, const int32_t* count,
                           float* dst, int k, int rows, int routes,
                           int capacity, int stride_row, int stride_expert,
                           int row_tiles) {
  constexpr int J = gguf_dispatch::kAlignedRouteTile;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, true);
  const int tile = blockIdx.x / row_tiles;
  const int row_tile = blockIdx.x % row_tiles;
  const int start = tile * J;
  if (start >= *count || experts[tile] < 0) return;
  extern __shared__ int sid[];
  __shared__ int last;
  const int lane = threadIdx.y * 32 + threadIdx.x;
  if (lane == 0) last = J - 1;
  __syncthreads();
  if (lane < J) {
    const int r = ids[start + lane];
    sid[lane] = r >= 0 && r < routes ? r : 0;
    if (r < 0 || r >= routes) atomicMin(&last, lane - 1);
  }
  __syncthreads();
  mul_mat_q_process_tile<type, J, true, false>(
      x, experts[tile] * stride_expert + row_tile * I * stride_row,
      y + static_cast<int64_t>(start) * (sizeof(block_q8_1_mmq) / sizeof(int)),
      sid, dst + row_tile * I, nullptr, nullptr, stride_row, capacity, rows,
      rows - row_tile * I - 1, last, 0, k / ggml_cuda_type_traits<type>::qk);
}

template <ggml_type type>
void launch_aligned_mmq(const ggml_tensor& w, const int* q8, const Tensor& ids,
                        const Tensor& experts, const Tensor& count, float* dst,
                        int routes, const ggml_cuda_mmq_config& config, int cc,
                        cudaStream_t stream) {
  const int shared = mmq_get_nbytes_shared(config, cc);
  const int row_tiles = (w.ne[1] + config.I - 1) / config.I;
  CUDA_SET_SHARED_MEMORY_LIMIT(aligned_mul_mat_q<type>, shared);
  aligned_mul_mat_q<type>
      <<<ids.numel() / gguf_dispatch::kAlignedRouteTile * row_tiles,
         dim3(32, config.nthreads / 32), shared, stream>>>(
          static_cast<const char*>(w.data), q8,
          static_cast<const int32_t*>(ids.data_ptr()),
          static_cast<const int32_t*>(experts.data_ptr()),
          static_cast<const int32_t*>(count.data_ptr()), dst, w.ne[0], w.ne[1],
          routes, ids.numel(), w.nb[1] / w.nb[0], w.nb[2] / w.nb[0], row_tiles);
}

template <typename T>
__global__ void gather_moe_routes(T* sorted, const T* input,
                                  const int32_t* routes, int64_t total,
                                  int64_t top_k, int64_t k) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total * k) {
    const int64_t route = index / k;
    sorted[index] = input[(routes[route] / top_k) * k + index % k];
  }
}

template <typename T>
__global__ void scatter_moe_routes(T* output, const T* sorted,
                                   const int32_t* routes, int64_t total,
                                   int64_t row) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total * row) {
    output[static_cast<int64_t>(routes[index / row]) * row + index % row] =
        sorted[index];
  }
}

Tensor run_grouped_dense(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, KernelMethod requested) {
  if (requested == KernelMethod::Blas) return ggml_dense_blas(W, X, type, row);
  if (requested == KernelMethod::DequantizeBlas)
    return ggml_dense_dequantize_blas(W, X, type, row);
  return ggml_dense(W, X, type, row);
}

Tensor run_upstream_moe_grouped(const Tensor& W, const Tensor& X,
                                const Tensor& topk_ids, int64_t type,
                                int64_t row, int64_t top_k, int64_t tokens,
                                int64_t k, cudaStream_t stream,
                                KernelMethod requested) {
  const int64_t total = tokens * top_k;
  STD_TORCH_CHECK(total <= INT_MAX, kMoeNotEligibleMarker,
                  ": too many routed tokens for int32 indices");
  const int64_t experts = W.size(0);
  std::vector<int32_t> ids(total);
  STD_TORCH_CHECK(
      cudaMemcpyAsync(ids.data(), topk_ids.data_ptr(), total * sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream) == cudaSuccess,
      "upstream MoE could not read expert indices");
  STD_TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                  "upstream MoE could not synchronize expert indices");

  std::vector<std::vector<int32_t>> by_expert(experts);
  for (int64_t route = 0; route < total; ++route) {
    STD_TORCH_CHECK(ids[route] >= 0 && ids[route] < experts,
                    "upstream MoE expert index out of range");
    by_expert[ids[route]].push_back(static_cast<int32_t>(route));
  }
  std::vector<int32_t> routes;
  routes.reserve(total);
  for (const auto& expert_routes : by_expert) {
    routes.insert(routes.end(), expert_routes.begin(), expert_routes.end());
  }
  Tensor device_routes =
      torch::stable::new_empty(topk_ids, {total}, ScalarType::Int);
  STD_TORCH_CHECK(
      cudaMemcpyAsync(device_routes.data_ptr(), routes.data(),
                      total * sizeof(int32_t), cudaMemcpyHostToDevice,
                      stream) == cudaSuccess,
      "upstream MoE could not upload sorted expert indices");
  STD_TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                  "upstream MoE could not finish uploading expert indices");

  Tensor sorted_input =
      torch::stable::new_empty(X, {total, k}, X.scalar_type());
  const int input_grid = static_cast<int>((total * k + 255) / 256);
  if (X.element_size() == sizeof(float)) {
    gather_moe_routes<<<input_grid, 256, 0, stream>>>(
        static_cast<uint32_t*>(sorted_input.data_ptr()),
        static_cast<const uint32_t*>(X.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, top_k, k);
  } else {
    gather_moe_routes<<<input_grid, 256, 0, stream>>>(
        static_cast<uint16_t*>(sorted_input.data_ptr()),
        static_cast<const uint16_t*>(X.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, top_k, k);
  }
  check_launch("upstream MoE gather");

  Tensor sorted_output =
      torch::stable::new_empty(X, {total, row}, X.scalar_type());
  int64_t first = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const int64_t count = by_expert[expert].size();
    if (count == 0) {
      continue;
    }
    Tensor expert_weight = torch::stable::select(W, 0, expert);
    Tensor expert_input = torch::stable::narrow(sorted_input, 0, first, count);
    Tensor expert_output =
        run_grouped_dense(expert_weight, expert_input, type, row, requested);
    auto* destination = static_cast<char*>(sorted_output.data_ptr()) +
                        first * row * X.element_size();
    STD_TORCH_CHECK(
        cudaMemcpyAsync(destination, expert_output.data_ptr(),
                        count * row * X.element_size(),
                        cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
        "upstream MoE could not copy expert output");
    first += count;
  }

  Tensor output = torch::stable::new_empty(X, {total, row}, X.scalar_type());
  const int output_grid = static_cast<int>((total * row + 255) / 256);
  if (X.element_size() == sizeof(float)) {
    scatter_moe_routes<<<output_grid, 256, 0, stream>>>(
        static_cast<uint32_t*>(output.data_ptr()),
        static_cast<const uint32_t*>(sorted_output.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, row);
  } else {
    scatter_moe_routes<<<output_grid, 256, 0, stream>>>(
        static_cast<uint16_t*>(output.data_ptr()),
        static_cast<const uint16_t*>(sorted_output.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, row);
  }
  check_launch("upstream MoE scatter");
  return output;
}

using MoeKernel = KernelMethod;

bool aligned_mmq_supported(const Tensor& W, int64_t type, int64_t row,
                           int64_t top_k, int64_t tokens, int cc,
                           size_t smpbo) {
  using namespace gguf_dispatch;
  if (!upstream_mmq_type_supported(type)) return false;
  // The shared aligned wrapper only describes Q8 activations. Native FP4
  // requires the upstream raw wrapper's separate activation/scale layout.
  if (blackwell_mma_available(cc) &&
      (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4))
    return false;
  const auto config = ggml_cuda_mmq_get_config(static_cast<ggml_type>(type),
                                               kAlignedRouteTile, true, cc);
  if (config.type == GGML_TYPE_COUNT ||
      mmq_get_nbytes_shared(config, cc) > smpbo)
    return false;
  const int64_t k = logical_k_from_moe_weight(W, type, "aligned MMQ support");
  if (!has_weight_padding(W, k, type, "aligned MMQ support")) return false;
  const int64_t routes = tokens * top_k;
  const int64_t capacity =
      GGML_PAD(routes + W.size(0) * (kAlignedRouteTile - 1), kAlignedRouteTile);
  const int guard = GGML_PAD(kAlignedRouteTile * MMQ_TILE_Y_K, config.nthreads);
  const int64_t ints_per_row =
      (padded_k(k) / QK8_1_MMQ) * (sizeof(block_q8_1_mmq) / sizeof(int));
  const auto w = make_moe_weight_tensor(W, type, k, row);
  return routes <= INT_MAX / row && k <= INT_MAX - MATRIX_ROW_PADDING &&
         row <= INT_MAX - config.I && ints_per_row > 0 &&
         capacity <= (INT_MAX - guard) / ints_per_row &&
         w.nb[2] / w.nb[0] <= INT_MAX / W.size(0) &&
         capacity / kAlignedRouteTile <=
             INT_MAX / ((row + config.I - 1) / config.I);
}

int64_t moe_supported_methods(const Tensor& W, int64_t type, int64_t row,
                              int64_t top_k, int64_t tokens, int cc,
                              int warp_size, size_t smpbo,
                              int64_t methods = -1) {
  using namespace gguf_dispatch;
  const bool floating = is_upstream_float_type(type);
  const int64_t k =
      floating ? W.size(2) : logical_k_from_moe_weight(W, type, "MoE support");
  if (row > INT_MAX || k > INT_MAX - MATRIX_ROW_PADDING ||
      W.size(0) > INT_MAX || tokens > INT_MAX / top_k)
    return 0;
  const int64_t routes = tokens * top_k;
  int64_t caps = 0;
  const auto wanted = [&](MoeKernel method) {
    return (methods & method_bit(method)) != 0;
  };
  const bool grouped =
      (wanted(MoeKernel::GroupedDense) ||
       wanted(floating ? MoeKernel::Blas : MoeKernel::DequantizeBlas)) &&
      !is_capturing(W.get_device_index()) &&
      (type != GGML_TYPE_MXFP4 || k % 256 == 0) &&
      upstream_blas_type_supported(type, cc) &&
      routes <= static_cast<int64_t>(INT_MAX) * 256 / std::max(k, row);
  if (grouped) {
    caps |= method_bit(floating ? MoeKernel::Blas : MoeKernel::DequantizeBlas);
    caps |= method_bit(MoeKernel::GroupedDense);
  }
  if (floating) {
    if (wanted(MoeKernel::Mmvf) && float_layout_supported(W, type))
      caps |= method_bit(MoeKernel::Mmvf);
    if (wanted(MoeKernel::Mmf) &&
        mmf_shape_supported(W, type, cc, warp_size, true, tokens) &&
        mmid_max_tokens(smpbo) > 0)
      caps |= method_bit(MoeKernel::Mmf);
  } else {
    if (has_weight_padding(W, k, type, "MoE support")) {
      if (wanted(MoeKernel::Mmvq) &&
          get_mmvq_mmid_max_batch(static_cast<ggml_type>(type), cc) > 0)
        caps |= method_bit(MoeKernel::Mmvq);
      if (wanted(MoeKernel::Mmq) &&
          mmid_max_tokens(smpbo) >= gguf_constants::kMmqTileStep &&
          mmq_launch_supported(type, row, cc, smpbo))
        caps |= method_bit(MoeKernel::Mmq);
      if (wanted(MoeKernel::MmqAligned) &&
          aligned_mmq_supported(W, type, row, top_k, tokens, cc, smpbo))
        caps |= method_bit(MoeKernel::MmqAligned);
    }
  }
  return caps & methods;
}

MoeKernel select_moe_method(const Tensor& W, int64_t type, int64_t row,
                            int64_t top_k, int64_t tokens, int cc,
                            int warp_size, size_t smpbo, int64_t* chunk_size,
                            MoeKernel requested, bool allow_aligned = false) {
  using namespace gguf_dispatch;
  // Query only the candidates reached by the policy. Fixed methods check only
  // their own hard constraints; the public support query still checks all.
  int64_t checked = 0, caps = 0;
  const auto has = [&](MoeKernel m) {
    const int64_t bit = method_bit(m);
    if (!(checked & bit)) {
      caps |= moe_supported_methods(W, type, row, top_k, tokens, cc, warp_size,
                                    smpbo, bit);
      checked |= bit;
    }
    return (caps & bit) != 0;
  };
  const auto q = static_cast<ggml_type>(type);
  MoeKernel selected = requested;
  if (requested == MoeKernel::None) {
    if (is_upstream_float_type(type)) {
      const auto weight = make_moe_float_weight_tensor(W, type);
      if (tokens <= MMVF_MAX_BATCH_SIZE && has(MoeKernel::Mmvf) &&
          ggml_cuda_should_use_mmvf(q, cc, weight.ne, weight.nb, tokens))
        selected = MoeKernel::Mmvf;
      else if (has(MoeKernel::Mmf))
        selected = MoeKernel::Mmf;
      else if (tokens * top_k <= kMmvfMaxRoutes && has(MoeKernel::Mmvf))
        selected = MoeKernel::Mmvf;
      else if (has(MoeKernel::Blas))
        selected = MoeKernel::Blas;
      else if (has(MoeKernel::Mmvf))
        selected = MoeKernel::Mmvf;
    } else {
      if (tokens * top_k <= kMmvqMaxRoutes && has(MoeKernel::Mmvq))
        selected = MoeKernel::Mmvq;
      else if (allow_aligned && W.size(0) <= kAlignedMaxExperts &&
               has(MoeKernel::MmqAligned))
        selected = MoeKernel::MmqAligned;
      // Keep aligned routing ahead of host grouping when the caller can
      // provide a route plan. The threshold still applies to the raw path.
      else if (tokens > kGroupedTokenThreshold && has(MoeKernel::GroupedDense))
        selected = MoeKernel::GroupedDense;
      else if (has(MoeKernel::Mmq))
        selected = MoeKernel::Mmq;
      else if (has(MoeKernel::Mmvq))
        selected = MoeKernel::Mmvq;
      else if (has(MoeKernel::DequantizeBlas))
        selected = MoeKernel::DequantizeBlas;
    }
  }
  if (selected == MoeKernel::None || !has(selected)) {
    // Keep diagnostics complete; successful calls need only the lazy checks.
    const int64_t supported = moe_supported_methods(W, type, row, top_k, tokens,
                                                    cc, warp_size, smpbo);
    STD_TORCH_CHECK(false, kMoeNotEligibleMarker,
                    ": requested MoE method cannot run this "
                    "type/shape/device/storage; supported=",
                    supported);
  }
  if (selected == MoeKernel::Mmvq)
    *chunk_size =
        std::min<int>(MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch(q, cc));
  else if (selected == MoeKernel::Mmvf)
    *chunk_size = MMVF_MAX_BATCH_SIZE;
  else if (selected == MoeKernel::Mmq || selected == MoeKernel::Mmf)
    *chunk_size = mmid_max_tokens(smpbo);
  if (selected == MoeKernel::Mmf && requested == MoeKernel::None) {
    const auto weight = make_moe_float_weight_tensor(W, type);
    int64_t low = 1, high = std::min(tokens, *chunk_size);
    while (low < high) {
      const int64_t mid = low + (high - low + 1) / 2;
      if (ggml_cuda_should_use_mmf(q, cc, warp_size, weight.ne, weight.nb, mid,
                                   true))
        low = mid;
      else
        high = mid - 1;
    }
    *chunk_size = low;
  }
  return selected;
}

Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k, MoeKernel requested) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const auto& device = ggml_cuda_info().devices[device_index];
  int64_t chunk_size = 0;
  const MoeKernel kernel =
      select_moe_method(W, type, row, top_k, tokens, device.cc,
                        device.warp_size, device.smpbo, &chunk_size, requested);
  if (kernel == MoeKernel::GroupedDense || kernel == MoeKernel::Blas ||
      kernel == MoeKernel::DequantizeBlas)
    return run_upstream_moe_grouped(W, X, topk_ids, type, row, top_k, tokens, k,
                                    current_stream(device_index), kernel);

  UpstreamCall call(X);
  const int64_t output_rows = tokens * top_k;
  const ProjectionBuffers buffers = projection_buffers(
      X, output_rows, row, 0, *call.scratch_pool, call.stream);

  ggml_tensor src0 = kernel == MoeKernel::Mmvf || kernel == MoeKernel::Mmf
                         ? make_moe_float_weight_tensor(W, type)
                         : make_moe_weight_tensor(W, type, k, row);
  ggml_tensor src1 = make_moe_input_tensor(buffers.input, k, tokens);
  ggml_tensor ids_tensor = make_moe_ids_tensor(
      static_cast<const int32_t*>(topk_ids.data_ptr()), top_k, tokens);
  ggml_tensor dst = make_moe_output_tensor(buffers.result, row, top_k, tokens);

  if (kernel == MoeKernel::Mmvq && tokens <= chunk_size) {
    ggml_cuda_mul_mat_vec_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVQ");
  } else if (kernel == MoeKernel::Mmq && tokens <= chunk_size) {
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMQ");
  } else if (kernel == MoeKernel::Mmvf && tokens <= chunk_size) {
    ggml_cuda_mul_mat_vec_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVF");
  } else if (kernel == MoeKernel::Mmf && tokens <= chunk_size) {
    ggml_cuda_mul_mat_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMF");
  } else {
    // Each chunk uses the selected kernel's token limit and writes directly
    // into its slice of the shared projection output.
    const bool chunked_mmq = kernel == MoeKernel::Mmq;
    for (int64_t start = 0; start < tokens;) {
      int64_t count = std::min<int64_t>(chunk_size, tokens - start);
      if (chunked_mmq && tokens - start - count > 0 &&
          tokens - start - count < gguf_constants::kMmqTileStep) {
        // ggml_cuda_mmq_get_J_max rounds down to one J tile.
        count -= gguf_constants::kMmqTileStep - (tokens - start - count);
      }
      ggml_tensor chunk_input =
          make_moe_input_tensor(buffers.input + start * k, k, count);
      ggml_tensor chunk_ids = make_moe_ids_tensor(
          static_cast<const int32_t*>(topk_ids.data_ptr()) + start * top_k,
          top_k, count);
      ggml_tensor chunk_output = make_moe_output_tensor(
          buffers.result + start * top_k * row, row, top_k, count);
      if (chunked_mmq) {
        ggml_cuda_mul_mat_q(call.context, &src0, &chunk_input, &chunk_ids,
                            &chunk_output);
        check_launch("upstream MoE chunked MMQ");
      } else if (kernel == MoeKernel::Mmf) {
        ggml_cuda_mul_mat_f(call.context, &src0, &chunk_input, &chunk_ids,
                            &chunk_output);
        check_launch("upstream MoE chunked MMF");
      } else if (kernel == MoeKernel::Mmvf) {
        ggml_cuda_mul_mat_vec_f(call.context, &src0, &chunk_input, &chunk_ids,
                                &chunk_output);
        check_launch("upstream MoE chunked MMVF");
      } else {
        ggml_cuda_mul_mat_vec_q(call.context, &src0, &chunk_input, &chunk_ids,
                                &chunk_output);
        check_launch("upstream MoE chunked MMVQ");
      }
      start += count;
    }
  }
  return finish_output(buffers, call.stream);
}

Tensor run_upstream_moe(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                        int64_t row, int64_t top_k, int64_t tokens,
                        MoeKernel requested) {
  check_moe_inputs(X, W, topk_ids, type, row, top_k, tokens);
  const bool float_weight = is_upstream_float_type(type);
  const int64_t k =
      float_weight ? W.size(2) : logical_k_from_moe_weight(W, type, "ggml_moe");
  STD_TORCH_CHECK(X.size(1) == k, kMoeNotEligibleMarker,
                  ": X K dimension does not match the expert row");
  const int64_t routes = checked_moe_product(tokens, top_k, "route count");
  checked_moe_product(routes, k, "routed input element count");
  checked_moe_product(routes, row, "routed output element count");
  return run_upstream_moe_projection(W, X, topk_ids, type, row, top_k, tokens,
                                     k, requested);
}

Tensor run_aligned_mmq(Tensor X, Tensor W, Tensor sorted_ids, Tensor expert_ids,
                       Tensor padded_count, int64_t type, int64_t row,
                       int64_t top_k, int64_t tokens) {
  const int64_t routes = checked_moe_product(tokens, top_k, "route count");
  check_moe_inputs(X, W, sorted_ids, type, row, top_k, tokens, true);
  const int64_t k = logical_k_from_moe_weight(W, type, "aligned MoE MMQ");
  const auto& device = ggml_cuda_info().devices[X.get_device_index()];
  STD_TORCH_CHECK(upstream_mmq_type_supported(type) &&
                      !(blackwell_mma_available(device.cc) &&
                        (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4)),
                  kMoeNotEligibleMarker,
                  ": aligned MMQ requires a supported Q8 activation layout");
  const auto config = ggml_cuda_mmq_get_config(static_cast<ggml_type>(type),
                                               gguf_dispatch::kAlignedRouteTile,
                                               true, device.cc);
  STD_TORCH_CHECK(config.type != GGML_TYPE_COUNT &&
                      mmq_get_nbytes_shared(config, device.cc) <= device.smpbo,
                  kMoeNotEligibleMarker,
                  ": aligned MMQ has no supported device configuration");
  STD_TORCH_CHECK(
      X.size(1) == k && has_weight_padding(W, k, type, "aligned MoE MMQ"),
      kMoeNotEligibleMarker, ": weight/input K or padding mismatch");
  for (const Tensor* t : {&expert_ids, &padded_count}) {
    STD_TORCH_CHECK(
        t->defined() && t->is_cuda() &&
            t->get_device_index() == X.get_device_index() &&
            t->is_contiguous() && t->scalar_type() == ScalarType::Int,
        kMoeNotEligibleMarker, ": aligned metadata must be CUDA int32");
  }
  const int64_t capacity = sorted_ids.numel();
  const int64_t kp = GGML_PAD(k, MATRIX_ROW_PADDING);
  const int guard = GGML_PAD(gguf_dispatch::kAlignedRouteTile * MMQ_TILE_Y_K,
                             config.nthreads);
  const int64_t ints_per_row =
      (kp / QK8_1_MMQ) * (sizeof(block_q8_1_mmq) / sizeof(int));
  const auto w = make_moe_weight_tensor(W, type, k, row);
  STD_TORCH_CHECK(
      capacity % gguf_dispatch::kAlignedRouteTile == 0 &&
          expert_ids.numel() >= capacity / gguf_dispatch::kAlignedRouteTile &&
          padded_count.numel() == 1 && routes <= INT_MAX / row &&
          k <= INT_MAX - MATRIX_ROW_PADDING && row <= INT_MAX - config.I &&
          capacity <= (INT_MAX - guard) / ints_per_row &&
          w.nb[2] / w.nb[0] <= INT_MAX / W.size(0) &&
          capacity / gguf_dispatch::kAlignedRouteTile <=
              INT_MAX / ((row + config.I - 1) / config.I),
      kMoeNotEligibleMarker, ": unsupported aligned MMQ dimensions");
  UpstreamCall call(X);
  const auto buffers =
      projection_buffers(X, routes, row, 0, *call.scratch_pool, call.stream);
  const int64_t q8_ints = capacity * ints_per_row;
  const int64_t offset = GGML_PAD(capacity, 64);
  Tensor scratch =
      torch::stable::new_empty(X, {offset + q8_ints + guard}, ScalarType::Int);
  int32_t* src = static_cast<int32_t*>(scratch.data_ptr());
  int* q8 = src + offset;
  CUDA_CHECK(
      cudaMemsetAsync(q8 + q8_ints, 0, guard * sizeof(int), call.stream));
  aligned_source_rows<<<(capacity + 255) / 256, 256, 0, call.stream>>>(
      static_cast<const int32_t*>(sorted_ids.data_ptr()),
      static_cast<const int32_t*>(padded_count.data_ptr()), src, capacity,
      routes, top_k);
  quantize_mmq_q8_1_cuda(buffers.input, src, q8, static_cast<ggml_type>(type),
                         k, k, 0, 0, kp, capacity, 1, 1, call.stream);
  switch (type) {
#define GGUF_ALIGNED_CASE(type)                                           \
  case type:                                                              \
    launch_aligned_mmq<type>(w, q8, sorted_ids, expert_ids, padded_count, \
                             buffers.result, routes, config, device.cc,   \
                             call.stream);                                \
    break;
#define GGUF_ALIGNED_TRAIT(type, block, cpp_type, name, quantized, mmq) \
  GGUF_IF_MMQ(mmq, GGUF_ALIGNED_CASE, type)
    GGUF_GGML_TYPE_TRAITS(GGUF_ALIGNED_TRAIT)
#undef GGUF_ALIGNED_TRAIT
#undef GGUF_ALIGNED_CASE
    default:
      GGML_ABORT("unsupported aligned MMQ type");
  }
  check_launch("upstream aligned MoE MMQ");
  return finish_output(buffers, call.stream);
}

}  // namespace

Tensor ggml_moe_a8_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                            int64_t row, int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, topk_ids, type, row, top_k, tokens,
                          MoeKernel::None);
}

Tensor ggml_moe_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                         int64_t row, int64_t top_k, int64_t tokens) {
  return ggml_moe_a8_upstream(X, W, topk_ids, type, row, top_k, tokens);
}

Tensor ggml_moe_mmvq(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                     int64_t row, int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, topk_ids, type, row, top_k, tokens,
                          MoeKernel::Mmvq);
}

Tensor ggml_moe_mmq(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                    int64_t row, int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, topk_ids, type, row, top_k, tokens,
                          MoeKernel::Mmq);
}

Tensor ggml_moe_grouped_dense(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                              int64_t row, int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, topk_ids, type, row, top_k, tokens,
                          MoeKernel::GroupedDense);
}

Tensor ggml_moe(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, ids, type, row, top_k, tokens, MoeKernel::None);
}
Tensor ggml_moe_mmvf(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                     int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, ids, type, row, top_k, tokens, MoeKernel::Mmvf);
}
Tensor ggml_moe_mmf(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                    int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, ids, type, row, top_k, tokens, MoeKernel::Mmf);
}
Tensor ggml_moe_blas(Tensor X, Tensor W, Tensor ids, int64_t type, int64_t row,
                     int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, ids, type, row, top_k, tokens, MoeKernel::Blas);
}
Tensor ggml_moe_dequantize_blas(Tensor X, Tensor W, Tensor ids, int64_t type,
                                int64_t row, int64_t top_k, int64_t tokens) {
  return run_upstream_moe(X, W, ids, type, row, top_k, tokens,
                          MoeKernel::DequantizeBlas);
}
Tensor ggml_moe_mmq_aligned(Tensor X, Tensor W, Tensor sorted_routes,
                            int64_t type, int64_t row, int64_t top_k,
                            int64_t tokens, Tensor experts, Tensor count) {
  return run_aligned_mmq(X, W, sorted_routes, experts, count, type, row, top_k,
                         tokens);
}
int64_t ggml_moe_supported_methods(Tensor X, Tensor W, Tensor ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens) {
  try {
    check_moe_inputs(X, W, ids, type, row, top_k, tokens);
    checked_moe_product(tokens, top_k, "routes");
    const int64_t k = is_upstream_float_type(type)
                          ? W.size(2)
                          : logical_k_from_moe_weight(W, type, "MoE support");
    if (X.size(1) != k) return 0;
    const DeviceGuard guard(X.get_device_index());
    const auto& d = ggml_cuda_info().devices[X.get_device_index()];
    return moe_supported_methods(W, type, row, top_k, tokens, d.cc, d.warp_size,
                                 d.smpbo);
  } catch (const std::runtime_error&) {
    return 0;
  }
}
int64_t ggml_moe_select_method(Tensor X, Tensor W, Tensor ids, int64_t type,
                               int64_t row, int64_t top_k, int64_t tokens,
                               bool allow_aligned) {
  try {
    check_moe_inputs(X, W, ids, type, row, top_k, tokens);
    checked_moe_product(tokens, top_k, "routes");
    const int64_t k = is_upstream_float_type(type)
                          ? W.size(2)
                          : logical_k_from_moe_weight(W, type, "MoE selection");
    if (X.size(1) != k) return 0;
    const DeviceGuard guard(X.get_device_index());
    const auto& d = ggml_cuda_info().devices[X.get_device_index()];
    int64_t chunk = 0;
    return method_bit(select_moe_method(W, type, row, top_k, tokens, d.cc,
                                        d.warp_size, d.smpbo, &chunk,
                                        MoeKernel::None, allow_aligned));
  } catch (const std::runtime_error&) {
    return 0;
  }
}
