// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Values are also the bits returned by supported_methods. Zero is no method.
// MMQ_ALIGNED is a distinct execution layout, never a different backend.
enum class KernelMethod : int64_t {
  None = 0,
  Mmvf = 1,
  Mmf = 2,
  Mmvq = 4,
  Mmq = 8,
  DequantizeBlas = 16,
  Blas = 32,
  GroupedDense = 64,
  MmqAligned = 128,
};
constexpr int64_t method_bit(KernelMethod method) {
  return static_cast<int64_t>(method);
}

namespace gguf_dispatch {
// Device-independent auto thresholds; fixed methods keep their hard limits.
constexpr int64_t kDenseMmqMaxBatch = 128;
// For top_k=8, 256 routes correspond to 32 tokens in both FFN projections.
constexpr int64_t kMmvqMaxRoutes = 256;
// When MMF is unavailable, small routed batches avoid host grouping/BLAS.
constexpr int64_t kMmvfMaxRoutes = 1024;
constexpr int64_t kGroupedTokenThreshold = 8192;
constexpr int kAlignedRouteTile = 16;
constexpr int kAlignedMaxExperts = 992;
constexpr int kFloatMmfMaxColumns = 16;
}  // namespace gguf_dispatch

// Torch bindings and legacy-only builds need only the definitions above.
// CUDA execution helpers are compiled exclusively by the upstream CUDA bridge.
#if defined(__CUDACC__) && !defined(VLLM_GGUF_LEGACY_ONLY)
  #include "ggml_dypes.cuh"
  #include "torch_context.cuh"
  #include "mmq.cuh"
  #include "mmvq.cuh"
  #include "mmvf.cuh"
  #include "mmf.cuh"
  #include <algorithm>
  #include <climits>

// Shared hard execution constraints. Performance thresholds live only in
// the selectors; forced methods and supported_methods use these predicates.
namespace gguf_dispatch {
inline bool float_layout_supported(const Tensor& W, int64_t type) {
  if (!float_type_matches(W, type) || !W.is_contiguous()) return false;
  const int64_t k = W.size(W.dim() - 1);
  return k > 0 && k % 2 == 0 &&
         reinterpret_cast<uintptr_t>(W.data_ptr()) % (2 * W.element_size()) ==
             0;
}

inline bool mmf_shape_supported(const Tensor& W, int64_t type, int cc,
                                int warp_size, bool moe, int64_t batch) {
  if (!float_layout_supported(W, type) || (!moe && batch > kFloatMmfMaxColumns))
    return false;
  const ggml_tensor weight = moe ? make_moe_float_weight_tensor(W, type)
                                 : make_float_weight_tensor(W, type);
  // With count=1, upstream checks layout and compiled MMA availability but
  // none of its workload-size recommendations can reject the method.
  return ggml_cuda_should_use_mmf(static_cast<ggml_type>(type), cc, warp_size,
                                  weight.ne, weight.nb, 1, moe);
}

inline bool mmq_launch_supported(int64_t type, int64_t row, int cc,
                                 size_t smpbo, bool force_boundary = false) {
  if (!upstream_mmq_type_supported(type)) return false;
  const bool boundary = force_boundary || row % 128 != 0;
  for (int j = gguf_constants::kMmqTileStep;
       j <= gguf_constants::kMmqTileColumnsMax;
       j += gguf_constants::kMmqTileStep) {
    const auto cfg =
        ggml_cuda_mmq_get_config(static_cast<ggml_type>(type), j, boundary, cc);
    if (cfg.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(cfg, cc) <= smpbo)
      return true;
  }
  return false;
}

inline int64_t mmid_max_tokens(size_t smpbo) {
  return std::min<int64_t>(smpbo / sizeof(int32_t), (1u << 22) - 1);
}

inline bool is_capturing(int32_t device_index) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(current_stream(device_index), &status));
  return status != cudaStreamCaptureStatusNone;
}
}  // namespace gguf_dispatch

bool upstream_blas_type_supported(int64_t type, int cc);
Tensor ggml_dense(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_dequantize_blas(Tensor W, Tensor X, int64_t type,
                                  int64_t row);

#endif  // __CUDACC__ && !VLLM_GGUF_LEGACY_ONLY
