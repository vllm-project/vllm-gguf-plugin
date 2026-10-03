// SPDX-License-Identifier: Apache-2.0
#include "kernel_dispatch.cuh"
#include "torch_context.cuh"
#include "quantize.cuh"

#include <climits>
#include <limits>
#include <torch/csrc/stable/ops.h>

using torch::headeronly::ScalarType;

Tensor run_upstream_blas(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k);

namespace {
constexpr int64_t kDenseMmvf = method_bit(KernelMethod::Mmvf);
constexpr int64_t kDenseMmf = method_bit(KernelMethod::Mmf);
constexpr int64_t kDenseMmvq = method_bit(KernelMethod::Mmvq);
constexpr int64_t kDenseMmq = method_bit(KernelMethod::Mmq);
constexpr int64_t kDenseDequantizeBlas =
    method_bit(KernelMethod::DequantizeBlas);
constexpr int64_t kDenseBlas = method_bit(KernelMethod::Blas);

void check_dense_inputs(const Tensor& W, const Tensor& X, int64_t row,
                        const char* op_name) {
  STD_TORCH_CHECK(W.is_cuda() && X.is_cuda(), op_name,
                  ": W and X must be CUDA tensors");
  STD_TORCH_CHECK(W.get_device_index() == X.get_device_index(), op_name,
                  ": W and X must be on the same CUDA device");
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2, op_name,
                  ": W and X must be rank-2 tensors");
  STD_TORCH_CHECK(W.is_contiguous() && X.is_contiguous(), op_name,
                  ": W and X must be contiguous");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  op_name, ": X must have dtype fp32, fp16, or bf16");
  STD_TORCH_CHECK(row > 0 && row <= W.size(0), op_name,
                  ": row must be in (0, W.size(0)]");
}

// Dense-only dispatch into the upstream MMQ template instances.
// Explicit instances are emitted by the upstream template-instances sources;
// this wrapper does not copy or specialize the upstream kernel body.
void ggml_upstream_mul_mat_q(ggml_backend_cuda_context& context,
                             const mmq_args& args, cudaStream_t stream) {
  switch (args.type_x) {
#define GGUF_MMQ_INSTANTIATE(type)               \
  case type:                                     \
    mul_mat_q_case<type>(context, args, stream); \
    break;
#define GGUF_MMQ_CASE(type, block, cpp_type, name, quantized, mmq) \
  GGUF_IF_MMQ(mmq, GGUF_MMQ_INSTANTIATE, type)
    GGUF_GGML_TYPE_TRAITS(GGUF_MMQ_CASE)
#undef GGUF_MMQ_CASE
#undef GGUF_MMQ_INSTANTIATE
    default:
      GGML_ABORT("unsupported upstream MMQ type");
  }
}

Tensor run_upstream_float(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t route) {
  UpstreamCall call(X);
  const ProjectionBuffers buffers =
      projection_buffers(X, X.size(0), row, 0, *call.scratch_pool, call.stream);
  ggml_tensor src0 = make_float_weight_tensor(W, type);
  ggml_tensor src1 = make_f32_tensor_2d(buffers.input, X.size(1), X.size(0));
  ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, X.size(0));
  if (route == kDenseMmvf) {
    ggml_cuda_mul_mat_vec_f(call.context, &src0, &src1, nullptr, &dst);
  } else {
    ggml_cuda_mul_mat_f(call.context, &src0, &src1, nullptr, &dst);
  }
  check_launch(route == kDenseMmvf ? "upstream MMVF" : "upstream MMF");
  return finish_output(buffers, call.stream);
}

Tensor run_upstream_mmvq(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);

  const size_t q8_bytes = static_cast<size_t>(batch) *
                          static_cast<size_t>(k_padded) * sizeof(block_q8_1) /
                          QK8_1;
  const ProjectionBuffers buffers = projection_buffers(
      X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
  void* q8_data = buffers.q8;

  quantize_row_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  ggml_tensor src0 = make_quant_tensor(W, type, k, row);
  ggml_tensor src1 = make_f32_tensor_2d(buffers.input, k_padded, batch);
  ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, batch);
  ggml_cuda_op_mul_mat_vec_q(call.context, &src0, &src1, &dst,
                             static_cast<const char*>(W.data_ptr()),
                             buffers.input, static_cast<const char*>(q8_data),
                             buffers.result, 0, row, batch, k_padded, stream);
  check_launch("upstream MMVQ");
  return finish_output(buffers, stream);
}

Tensor run_upstream_mmq(const Tensor& W, const Tensor& X, int64_t type,
                        int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const bool fallback = row % 128 != 0;
  const int j_max = ggml_cuda_mmq_get_J_max(
      static_cast<ggml_type>(type), fallback, cc,
      std::max<int64_t>(batch, gguf_constants::kMmqTileStep));
  STD_TORCH_CHECK(j_max > 0, "ggml_dense_mmq: no upstream MMQ configuration");
  STD_TORCH_CHECK(
      gguf_dispatch::mmq_launch_supported(
          type, row, cc, ggml_cuda_info().devices[device_index].smpbo),
      "ggml_dense_mmq: no launchable upstream MMQ tile");
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;

  // Native FP4 (Blackwell MMA path): upstream swaps the Q8_1_MMQ activation
  // format for block_fp4_mmq and needs a separate per-column scale buffer for
  // NVFP4, with different block sizes, strides and kernel-side ne_block. The
  // bridge's merged Q8 workspace cannot express that layout, so for these
  // types hand the whole operator to the upstream wrapper: build plain F32
  // descriptors for src1/dst and let ggml_cuda_mul_mat_q quantize, allocate
  // (via the installed TorchScratchPool), scale and launch on ctx.stream()
  // itself. Describing src1 with the logical k (never k_padded, which would
  // claim padding we did not allocate) keeps upstream's own
  // MATRIX_ROW_PADDING handling authoritative.
  const bool native_fp4 = blackwell_mma_available(cc) &&
                          (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
  if (native_fp4) {
    // No bridge quantized workspace is needed; output conversion still runs
    // through the shared dense-buffers conversion path (zero q8 region).
    const ProjectionBuffers buffers =
        projection_buffers(X, X.size(0), row, 0, *call.scratch_pool, stream);
    ggml_tensor src0 = make_quant_tensor(W, type, k, row);
    ggml_tensor src1 = make_f32_tensor_2d(buffers.input, k, batch);
    ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, batch);
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, /*ids=*/nullptr, &dst);
    check_launch("upstream MMQ (native FP4)");
    return finish_output(buffers, stream);
  }

  const size_t q8_bytes = static_cast<size_t>(batch) *
                              static_cast<size_t>(k_padded) *
                              sizeof(block_q8_1_mmq) / QK8_1_MMQ +
                          static_cast<size_t>(j_max) * sizeof(block_q8_1_mmq);
  const ProjectionBuffers buffers = projection_buffers(
      X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
  void* q8_data = buffers.q8;

  quantize_mmq_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  // Named-field initialization (C++17: no designated initializers) so an
  // upstream mmq_args field addition fails to compile here instead of
  // silently shifting all subsequent positional values. Every field is
  // annotated with its source; stride_channel/sample entries use 1 because
  // the bridge builds a single-channel, single-sample dense descriptor.
  mmq_args args{};
  args.x = static_cast<const char*>(W.data_ptr());
  args.type_x = static_cast<ggml_type>(type);
  args.y = static_cast<const int*>(q8_data);
  args.ids_dst = nullptr;        // dense path: no expert routing
  args.expert_bounds = nullptr;  // dense path: no expert bounds
  args.dst = buffers.result;
  args.y_scale = nullptr;  // Q8 activation path: no NVFP4 scale
  args.ncols_x = k;
  args.nrows_x = row;
  args.ncols_dst = batch;
  args.stride_row_x = static_cast<int64_t>(
      W.size(1) / type_size_for_type(type, "upstream MMQ"));
  args.ncols_y = batch;
  args.nrows_dst = row;
  args.nchannels_x = 1;
  args.nchannels_y = 1;
  args.stride_channel_x = 1;
  args.stride_channel_y = 1;
  args.stride_channel_dst = 1;
  args.nsamples_x = 1;
  args.nsamples_y = 1;
  args.stride_sample_x = 1;
  args.stride_sample_y = 1;
  args.stride_sample_dst = 1;
  args.ncols_max = batch;
  args.ncols_opt = batch;
  ggml_upstream_mul_mat_q(call.context, args, stream);
  check_launch("upstream MMQ");
  return finish_output(buffers, stream);
}

int64_t dense_supported_methods(const Tensor& W, const Tensor& X, int64_t type,
                                int64_t row) {
  if (!W.is_cuda() || !X.is_cuda() ||
      W.get_device_index() != X.get_device_index() || W.dim() != 2 ||
      X.dim() != 2 || !W.is_contiguous() || !X.is_contiguous() || row <= 0 ||
      row > W.size(0) || row > INT_MAX || X.size(1) > INT_MAX ||
      X.size(0) < 0 || X.size(0) > INT_MAX ||
      (X.scalar_type() != ScalarType::Float &&
       X.scalar_type() != ScalarType::Half &&
       X.scalar_type() != ScalarType::BFloat16))
    return 0;
  const DeviceGuard guard(X.get_device_index());
  const auto& device = ggml_cuda_info().devices[X.get_device_index()];
  const int64_t batch = X.size(0);
  if (is_upstream_float_type(type)) {
    if (!float_type_matches(W, type) || row != W.size(0) ||
        W.size(1) != X.size(1) || X.size(1) <= 0)
      return 0;
    int64_t caps = row <= INT_MAX && X.size(1) <= INT_MAX &&
                           upstream_blas_type_supported(type, device.cc)
                       ? kDenseBlas
                       : 0;
    if (gguf_dispatch::float_layout_supported(W, type) &&
        batch <= MMVF_MAX_BATCH_SIZE)
      caps |= kDenseMmvf;
    if (gguf_dispatch::mmf_shape_supported(W, type, device.cc, device.warp_size,
                                           false, batch))
      caps |= kDenseMmf;
    return caps;
  }
  if (!is_upstream_weight_type(type) || W.element_size() != 1) return 0;
  const size_t ts = type_size_for_type(type, "dense support");
  if (W.size(1) <= 0 || W.size(1) % ts != 0) return 0;
  const int64_t k = logical_k_from_weight(W, type, "dense support");
  if (X.size(1) != k || k > INT_MAX || row > INT_MAX) return 0;
  int64_t caps = upstream_blas_type_supported(type, device.cc) &&
                         (type != GGML_TYPE_MXFP4 || k % 256 == 0)
                     ? kDenseDequantizeBlas
                     : 0;
  if (k <= INT_MAX - MATRIX_ROW_PADDING &&
      has_weight_padding(W, k, type, "dense support")) {
    if (batch <= MMVQ_MAX_BATCH_SIZE) caps |= kDenseMmvq;
    if (gguf_dispatch::mmq_launch_supported(type, row, device.cc, device.smpbo))
      caps |= kDenseMmq;
  }
  return caps;
}

int64_t dense_recommended_methods(const Tensor& W, const Tensor& X,
                                  int64_t type, int64_t row) {
  int64_t caps = dense_supported_methods(W, X, type, row);
  if (!caps || X.size(0) == 0) return caps;
  const DeviceGuard guard(X.get_device_index());
  const auto& d = ggml_cuda_info().devices[X.get_device_index()];
  const auto q = static_cast<ggml_type>(type);
  if (is_upstream_float_type(type)) {
    const auto w = make_float_weight_tensor(W, type);
    if (!ggml_cuda_should_use_mmvf(q, d.cc, w.ne, w.nb, X.size(0)))
      caps &= ~kDenseMmvf;
    if (!ggml_cuda_should_use_mmf(q, d.cc, d.warp_size, w.ne, w.nb, X.size(0),
                                  false))
      caps &= ~kDenseMmf;
  } else {
    if (!ggml_cuda_should_use_mmvq(q, d.cc, X.size(0))) caps &= ~kDenseMmvq;
    // Hard MMQ support was checked above. Use the plugin's common batch
    // threshold instead of upstream's architecture-dependent BLAS crossover.
    if (X.size(0) > gguf_dispatch::kDenseMmqMaxBatch) caps &= ~kDenseMmq;
  }
  return caps;
}

int64_t select_dense_method(const Tensor& W, const Tensor& X, int64_t type,
                            int64_t row) {
  const int64_t caps = dense_recommended_methods(W, X, type, row);
  for (auto method :
       {KernelMethod::Mmvf, KernelMethod::Mmf, KernelMethod::Mmvq,
        KernelMethod::Mmq, KernelMethod::DequantizeBlas, KernelMethod::Blas}) {
    if (caps & method_bit(method)) return method_bit(method);
  }
  // A recommendation cannot turn a runnable input into an unsupported one.
  const int64_t supported = dense_supported_methods(W, X, type, row);
  for (auto method : {KernelMethod::Mmvf, KernelMethod::Mmf, KernelMethod::Mmvq,
                      KernelMethod::Mmq}) {
    if (supported & method_bit(method)) return method_bit(method);
  }
  return 0;
}

Tensor run_selected_dense(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t route, const char* op_name) {
  // Private executor: both callers have already checked the selected route's
  // hard constraints. Do not repeat the full capability query per expert.
  if (route == kDenseMmvf || route == kDenseMmf)
    return run_upstream_float(W, X, type, row, route);
  const int64_t k = is_upstream_float_type(type)
                        ? W.size(1)
                        : logical_k_from_weight(W, type, op_name);
  if (route == kDenseMmvq) return run_upstream_mmvq(W, X, type, row, k);
  if (route == kDenseMmq) return run_upstream_mmq(W, X, type, row, k);
  return run_upstream_blas(W, X, type, row, k);
}

Tensor run_fixed_dense(Tensor W, Tensor X, int64_t type, int64_t row,
                       int64_t route, const char* name) {
  check_dense_inputs(W, X, row, name);
  STD_TORCH_CHECK(
      route != 0 && (dense_supported_methods(W, X, type, row) & route), name,
      ": method cannot run this type/shape/device/storage");
  if (X.size(0) == 0) {
    return torch::stable::new_empty(X, {0, row}, X.scalar_type());
  }
  return run_selected_dense(W, X, type, row, route, name);
}
}  // namespace

bool ggml_should_use_mmvq(int64_t type, int64_t cc, int64_t batch) {
  // No device lookup here: Python supplies the tensor device's capability,
  // and policy tests can exercise every architecture without that hardware.
  return is_upstream_weight_type(type) && batch > 0 &&
         batch <= MMVQ_MAX_BATCH_SIZE && cc > 0 &&
         cc <= std::numeric_limits<int>::max() &&
         ggml_cuda_should_use_mmvq(static_cast<ggml_type>(type),
                                   static_cast<int>(cc), batch);
}

int64_t ggml_dense_upstream_capabilities(Tensor W, Tensor X, int64_t type,
                                         int64_t row) {
  return dense_recommended_methods(W, X, type, row);
}

Tensor ggml_dense_mmvq(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseMmvq, "ggml_dense_mmvq");
}
Tensor ggml_dense_mmq(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseMmq, "ggml_dense_mmq");
}
Tensor ggml_dense_mmvf(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseMmvf, "ggml_dense_mmvf");
}
Tensor ggml_dense_mmf(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseMmf, "ggml_dense_mmf");
}
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseBlas, "ggml_dense_blas");
}

int64_t ggml_dense_supported_methods(Tensor W, Tensor X, int64_t type,
                                     int64_t row) {
  return dense_supported_methods(W, X, type, row);
}
int64_t ggml_dense_select_method(Tensor W, Tensor X, int64_t type,
                                 int64_t row) {
  return select_dense_method(W, X, type, row);
}
Tensor ggml_dense(Tensor W, Tensor X, int64_t type, int64_t row) {
  check_dense_inputs(W, X, row, "ggml_dense");
  const int64_t route = select_dense_method(W, X, type, row);
  STD_TORCH_CHECK(
      route != 0,
      "ggml_dense: method cannot run this type/shape/device/storage");
  if (X.size(0) == 0)
    return torch::stable::new_empty(X, {0, row}, X.scalar_type());
  return run_selected_dense(W, X, type, row, route, "ggml_dense");
}
Tensor ggml_dense_dequantize_blas(Tensor W, Tensor X, int64_t type,
                                  int64_t row) {
  return run_fixed_dense(W, X, type, row, kDenseDequantizeBlas,
                         "ggml_dense_dequantize_blas");
}
