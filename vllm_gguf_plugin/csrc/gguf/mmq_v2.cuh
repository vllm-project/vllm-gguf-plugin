#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mma_v2.cuh"

#define VLLM_GGUF_MMQ_ITER_K 256
#define VLLM_GGUF_MMQ_NWARPS 8
#define VLLM_GGUF_MMQ_X_Q4_0 64
#define VLLM_GGUF_MMQ_X_IQ4_XS 64
#define VLLM_GGUF_MOE_MMQ_X_IQ4_XS 8
#define VLLM_GGUF_MMQ_Y 128
#define VLLM_GGUF_MMQ_TILE_NE_K 32
#define VLLM_GGUF_MMQ_TILE_Y_K \
  (VLLM_GGUF_MMQ_TILE_NE_K + VLLM_GGUF_MMQ_TILE_NE_K / QI8_1)
#define VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 \
  (2 * VLLM_GGUF_MMQ_TILE_NE_K + 2 * VLLM_GGUF_MMQ_TILE_NE_K / QI8_0 + 4)
#define VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// IQ4_XS keeps the same on-disk block_iq4_xs layout as the existing vLLM
// MMVQ/dequant paths, but modern MMQ uses the llama.cpp b4267+ QR/QI tiling.
// Keep these constants local to v2 so the legacy IQ4_XS paths stay unchanged.
#define VLLM_GGUF_MMQ_QR4_XS 2
#define VLLM_GGUF_MMQ_QI4_XS (QK_K / (4 * VLLM_GGUF_MMQ_QR4_XS))

struct block_q8_1_mmq_v2 {
  half2 ds4[4];
  int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq_v2) == 4 * sizeof(block_q8_1),
              "Unexpected block_q8_1_mmq_v2 size");

static constexpr __host__ __device__ int vllm_gguf_mmq_pad(const int x,
                                                           const int n) {
  return ((x + n - 1) / n) * n;
}

static constexpr __device__ int vllm_gguf_mmq_get_granularity_device(
    const int mmq_x) {
#ifdef VLLM_GGUF_TURING_MMA_AVAILABLE
  return mmq_x >= 48 ? 16 : 8;
#else
  return 8;
#endif
}

template <typename scalar_t>
static __global__ void vllm_gguf_quantize_mmq_q8_1(
    const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx,
    const int kx_padded, const int batch, const int batch_padded) {
  const int i0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;  // K offset
  if (i0 >= kx_padded) {
    return;
  }

  const int i1 = blockIdx.x;  // token row

  float xi0 = 0.0f;
  float xi1 = 0.0f;
  float xi2 = 0.0f;
  float xi3 = 0.0f;
  if (i1 < batch) {
    if (i0 + 0 < kx) {
      xi0 = static_cast<float>(x[i1 * kx + i0 + 0]);
    }
    if (i0 + 1 < kx) {
      xi1 = static_cast<float>(x[i1 * kx + i0 + 1]);
    }
    if (i0 + 2 < kx) {
      xi2 = static_cast<float>(x[i1 * kx + i0 + 2]);
    }
    if (i0 + 3 < kx) {
      xi3 = static_cast<float>(x[i1 * kx + i0 + 3]);
    }
  }

  float amax = fabsf(xi0);
  amax = fmaxf(amax, fabsf(xi1));
  amax = fmaxf(amax, fabsf(xi2));
  amax = fmaxf(amax, fabsf(xi3));
  float sum = xi0 + xi1 + xi2 + xi3;

#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;

  char4 q;
  q.x = static_cast<int8_t>(roundf(xi0 * d_inv));
  q.y = static_cast<int8_t>(roundf(xi1 * d_inv));
  q.z = static_cast<int8_t>(roundf(xi2 * d_inv));
  q.w = static_cast<int8_t>(roundf(xi3 * d_inv));

  block_q8_1_mmq_v2* y = (block_q8_1_mmq_v2*)vy;
  const int ib = (i0 / (4 * QK8_1)) * batch_padded + i1;
  const int iqs = i0 % (4 * QK8_1);

  char4* yqs4 = (char4*)y[ib].qs;
  yqs4[iqs / 4] = q;

  if (iqs % 32 == 0) {
    y[ib].ds4[iqs / 32] = make_half2(__float2half(d), __float2half(sum));
  }
}

template <typename scalar_t>
static void vllm_gguf_quantize_mmq_q8_1_cuda(const scalar_t* x, void* vy,
                                             const int kx, const int kx_padded,
                                             const int batch,
                                             const int batch_padded,
                                             cudaStream_t stream) {
  const int block_num_y =
      (kx_padded + 4 * VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
      (4 * VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(batch_padded, block_num_y, 1);
  const dim3 block_size(VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  vllm_gguf_quantize_mmq_q8_1<<<num_blocks, block_size, 0, stream>>>(
      x, vy, kx, kx_padded, batch, batch_padded);
}

template <typename scalar_t>
static __global__ void vllm_gguf_quantize_moe_mmq_q8_1(
    const scalar_t* __restrict__ x, void* __restrict__ vy,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ num_tokens_post_padded, const int kx,
    const int kx_padded, const int batch, const int top_k,
    const int sorted_tokens_padded) {
  const int i0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;  // K offset
  if (i0 >= kx_padded) {
    return;
  }

  const int sorted_row = blockIdx.x;
  const bool valid_sorted_row = sorted_row < num_tokens_post_padded[0];
  const int token_off = valid_sorted_row ? sorted_token_ids[sorted_row] : -1;
  const int token = token_off >= 0 ? token_off / top_k : -1;

  float xi0 = 0.0f;
  float xi1 = 0.0f;
  float xi2 = 0.0f;
  float xi3 = 0.0f;
  if (token >= 0 && token < batch) {
    if (i0 + 0 < kx) {
      xi0 = static_cast<float>(x[token * kx + i0 + 0]);
    }
    if (i0 + 1 < kx) {
      xi1 = static_cast<float>(x[token * kx + i0 + 1]);
    }
    if (i0 + 2 < kx) {
      xi2 = static_cast<float>(x[token * kx + i0 + 2]);
    }
    if (i0 + 3 < kx) {
      xi3 = static_cast<float>(x[token * kx + i0 + 3]);
    }
  }

  float amax = fabsf(xi0);
  amax = fmaxf(amax, fabsf(xi1));
  amax = fmaxf(amax, fabsf(xi2));
  amax = fmaxf(amax, fabsf(xi3));
  float sum = xi0 + xi1 + xi2 + xi3;

#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;

  char4 q;
  q.x = static_cast<int8_t>(roundf(xi0 * d_inv));
  q.y = static_cast<int8_t>(roundf(xi1 * d_inv));
  q.z = static_cast<int8_t>(roundf(xi2 * d_inv));
  q.w = static_cast<int8_t>(roundf(xi3 * d_inv));

  block_q8_1_mmq_v2* y = (block_q8_1_mmq_v2*)vy;
  const int ib = (i0 / (4 * QK8_1)) * sorted_tokens_padded + sorted_row;
  const int iqs = i0 % (4 * QK8_1);

  char4* yqs4 = (char4*)y[ib].qs;
  yqs4[iqs / 4] = q;

  if (iqs % 32 == 0) {
    y[ib].ds4[iqs / 32] = make_half2(__float2half(d), __float2half(sum));
  }
}

template <typename scalar_t>
static void vllm_gguf_quantize_moe_mmq_q8_1_cuda(
    const scalar_t* x, void* vy, const int* sorted_token_ids,
    const int* num_tokens_post_padded, const int kx, const int kx_padded,
    const int batch, const int top_k, const int sorted_tokens_padded,
    cudaStream_t stream) {
  const int block_num_y =
      (kx_padded + 4 * VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
      (4 * VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(sorted_tokens_padded, block_num_y, 1);
  const dim3 block_size(VLLM_GGUF_CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  vllm_gguf_quantize_moe_mmq_q8_1<<<num_blocks, block_size, 0, stream>>>(
      x, vy, sorted_token_ids, num_tokens_post_padded, kx, kx_padded, batch,
      top_k, sorted_tokens_padded);
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void vllm_gguf_load_tiles_q4_0_v2(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0,
    const int i_max, const int stride) {
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int warp_size = WARP_SIZE_GGUF;

  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2 * VLLM_GGUF_MMQ_TILE_NE_K);

  constexpr int threads_per_row = VLLM_GGUF_MMQ_ITER_K / (4 * QR4_0);
  constexpr int nrows = warp_size / threads_per_row;
  const int txi =
      warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
  const int kbx = txi / QI4_0;
  const int kqsx = txi % QI4_0;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
    int i =
        i0 + (nrows == 1 ? threadIdx.y
                         : threadIdx.y * nrows + threadIdx.x / threads_per_row);

    if (need_check) {
      i = min(i, i_max);
    }

    const block_q4_0* bxi = (const block_q4_0*)x + kbx0 + i * stride + kbx;
    const int qs0 = get_int_b2(bxi->qs, kqsx);

    x_qs[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + 0] =
        __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
    x_qs[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx +
         QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
  }

  constexpr int blocks_per_tile_x_row = VLLM_GGUF_MMQ_TILE_NE_K / QI4_0;
  constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
  const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
    int i =
        i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

    if (need_check) {
      i = min(i, i_max);
    }

    const block_q4_0* bxi = (const block_q4_0*)x + kbx0 + i * stride + kbxd;
    x_df[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + kbxd] =
        __half2float(bxi->d);
  }
}

static __device__ __forceinline__ int2
vllm_gguf_get_int_from_table_16_v2(const int q4, const uint8_t* values) {
  int v1;
  int v2;
  get_int_from_table_16(static_cast<uint32_t>(q4), values, v1, v2);
  return make_int2(v1, v2);
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void vllm_gguf_load_tiles_iq4_xs_v2(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0,
    const int i_max, const int stride) {
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int warp_size = WARP_SIZE_GGUF;

  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2 * VLLM_GGUF_MMQ_TILE_NE_K);

  constexpr int threads_per_row =
      VLLM_GGUF_MMQ_ITER_K / (4 * VLLM_GGUF_MMQ_QR4_XS);
  constexpr int nrows = warp_size / threads_per_row;
  const int kqsx = threadIdx.x % threads_per_row;
  const uint8_t* values = (const uint8_t*)kvalues_iq4nl;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
    int i =
        i0 + (nrows == 1 ? threadIdx.y
                         : threadIdx.y * nrows + threadIdx.x / threads_per_row);

    if (need_check) {
      i = min(i, i_max);
    }

    const block_iq4_xs* bxi = (const block_iq4_xs*)x + kbx0 + i * stride;
    const int aux_q4 = get_int_b4(bxi->qs, kqsx);
    const int2 v = vllm_gguf_get_int_from_table_16_v2(aux_q4, values);
    const int k0 = 8 * (kqsx / 4) + kqsx % 4;

    x_qs[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
    x_qs[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
  }

  constexpr int scale_groups = VLLM_GGUF_MMQ_TILE_NE_K / 4;
  constexpr int rows_per_warp = warp_size / scale_groups;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
    int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / scale_groups;

    if (need_check) {
      i = min(i, i_max);
    }

    const block_iq4_xs* bxi = (const block_iq4_xs*)x + kbx0 + i * stride;
    const int scale_idx = threadIdx.x % scale_groups;
    const int ls =
        ((bxi->scales_l[scale_idx / 2] >> (4 * (scale_idx % 2))) & 0x0F) |
        (((bxi->scales_h >> (2 * scale_idx)) & 0x03) << 4);

    x_df[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + scale_idx] =
        __half2float(bxi->d) * (ls - 32);
  }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vllm_gguf_vec_dot_q8_0_q8_1_mma_v2(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, const int k00) {
  using namespace vllm_gguf_mma;

  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int granularity = vllm_gguf_mmq_get_granularity_device(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * VLLM_GGUF_MMQ_TILE_Y_K);

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + 2 * VLLM_GGUF_MMQ_TILE_NE_K;
  const int* y_qs = (const int*)y + 4;
  const half2* y_ds = (const half2*)y;

  tile_A A[ntx][VLLM_GGUF_MMQ_TILE_NE_K / QI8_0];
  float dA[ntx][tile_C::ne / 2][VLLM_GGUF_MMQ_TILE_NE_K / QI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < VLLM_GGUF_MMQ_TILE_NE_K; k01 += QI8_0) {
      const int k0 = k00 + k01;
      load_ldmatrix(
          A[n][k01 / QI8_0],
          x_qs + (i0 + n * tile_A::I) * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + k0,
          VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0);
    }

#pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);

#pragma unroll
      for (int k01 = 0; k01 < VLLM_GGUF_MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;
        dA[n][l][k01 / QI8_0] =
            x_df[i * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0 + k0 / QI8_0];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < VLLM_GGUF_MMQ_TILE_NE_K; k01 += QI8_0) {
      tile_B B;
      float dB[tile_C::ne / 2];

      load_generic(B, y_qs + j0 * VLLM_GGUF_MMQ_TILE_Y_K + k01,
                   VLLM_GGUF_MMQ_TILE_Y_K);

#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = __low2float(y_ds[j * VLLM_GGUF_MMQ_TILE_Y_K + k01 / QI8_1]);
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01 / QI8_0], B);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
        }
      }
    }
  }
}

template <typename scalar_t, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vllm_gguf_mmq_write_back_mma_v2(
    const float* __restrict__ sum, scalar_t* __restrict__ dst, const int stride,
    const int i_max, const int j_max) {
  using namespace vllm_gguf_mma;

  typedef tile<16, 8, int> tile_C;
  constexpr int granularity = vllm_gguf_mmq_get_granularity_device(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j > j_max) {
          continue;
        }

        const int i = i0 + n * tile_C::I + tile_C::get_i(l);
        if (need_check && i > i_max) {
          continue;
        }

        dst[j * stride + i] =
            static_cast<scalar_t>(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

template <int mmq_x>
static __device__ __forceinline__ void vllm_gguf_load_moe_q8_1_tile_mma_v2(
    const int* __restrict__ y, int* __restrict__ tile_y, const int col_dst_0,
    const int block_y, const int ncols_y_padded) {
  constexpr int warp_size = WARP_SIZE_GGUF;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int sz = sizeof(block_q8_1_mmq_v2) / sizeof(int);
  const int* by0 = y + (block_y * ncols_y_padded + col_dst_0) * sz;

#pragma unroll
  for (int l0 = 0; l0 < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K;
       l0 += nwarps * warp_size) {
    const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
    if (l < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K) {
      const int k = l % VLLM_GGUF_MMQ_TILE_Y_K;
      tile_y[l] = by0[(l / VLLM_GGUF_MMQ_TILE_Y_K) * sz + k];
    }
  }
}

template <typename scalar_t, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vllm_gguf_moe_mmq_write_back_mma_v2(
    const float* __restrict__ sum, scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids, const int col_dst_0,
    const int row_dst_0, const int ncols_dst, const int stride,
    const int i_max) {
  using namespace vllm_gguf_mma;

  typedef tile<16, 8, int> tile_C;
  constexpr int granularity = vllm_gguf_mmq_get_granularity_device(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j >= mmq_x) {
          continue;
        }
        const int col_dst = sorted_token_ids[col_dst_0 + j];
        if (col_dst < 0 || col_dst >= ncols_dst) {
          continue;
        }

        const int i = i0 + n * tile_C::I + tile_C::get_i(l);
        if (need_check && i > i_max) {
          continue;
        }

        dst[col_dst * stride + row_dst_0 + i] =
            static_cast<scalar_t>(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

template <typename scalar_t, int mmq_x, bool need_check>
static __device__ __forceinline__ void vllm_gguf_mul_mat_q4_0_process_tile_v2(
    const char* __restrict__ x, const int offset_x, const int* __restrict__ y,
    scalar_t* __restrict__ dst, const int stride_row_x, const int ncols_y,
    const int stride_col_dst, const int tile_x_max_i, const int tile_y_max_j,
    const int kb0_start, const int kb0_stop) {
  constexpr int warp_size = WARP_SIZE_GGUF;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int blocks_per_iter = VLLM_GGUF_MMQ_ITER_K / QK4_0;
  constexpr int sz = sizeof(block_q8_1_mmq_v2) / sizeof(int);
  constexpr int tile_y_ints =
      vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K, nwarps * warp_size);

  extern __shared__ int data_mul_mat_q_v2[];
  int* tile_y = data_mul_mat_q_v2;
  int* tile_x = tile_y + tile_y_ints;

  float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    vllm_gguf_load_tiles_q4_0_v2<mmq_y, need_check>(x, tile_x, offset_x + kb0,
                                                    tile_x_max_i, stride_row_x);

    {
      const int* by0 = y + ncols_y * (kb0 * QK4_0 / (4 * QK8_1)) * sz;
#pragma unroll
      for (int l0 = 0; l0 < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K;
           l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
    __syncthreads();

    {
      const int* by0 = y + ncols_y * ((kb0 * QK4_0 / (4 * QK8_1)) * sz + sz);
#pragma unroll
      for (int l0 = 0; l0 < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K;
           l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum,
                                                     VLLM_GGUF_MMQ_TILE_NE_K);
    __syncthreads();
  }

  vllm_gguf_mmq_write_back_mma_v2<scalar_t, mmq_x, mmq_y, need_check>(
      sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

template <typename scalar_t, int mmq_x, bool need_check>
static __device__ __forceinline__ void vllm_gguf_mul_mat_iq4_xs_process_tile_v2(
    const char* __restrict__ x, const int offset_x, const int* __restrict__ y,
    scalar_t* __restrict__ dst, const int stride_row_x, const int ncols_y,
    const int stride_col_dst, const int tile_x_max_i, const int tile_y_max_j,
    const int kb0_start, const int kb0_stop) {
  constexpr int warp_size = WARP_SIZE_GGUF;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int blocks_per_iter = VLLM_GGUF_MMQ_ITER_K / QK_K;
  constexpr int sz = sizeof(block_q8_1_mmq_v2) / sizeof(int);
  constexpr int tile_y_ints =
      vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K, nwarps * warp_size);

  extern __shared__ int data_mul_mat_q_v2[];
  int* tile_y = data_mul_mat_q_v2;
  int* tile_x = tile_y + tile_y_ints;

  float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    vllm_gguf_load_tiles_iq4_xs_v2<mmq_y, need_check>(
        x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

    {
      const int* by0 = y + ncols_y * (kb0 * QK_K / (4 * QK8_1)) * sz;
#pragma unroll
      for (int l0 = 0; l0 < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K;
           l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
    __syncthreads();

    {
      const int* by0 = y + ncols_y * ((kb0 * QK_K / (4 * QK8_1)) * sz + sz);
#pragma unroll
      for (int l0 = 0; l0 < mmq_x * VLLM_GGUF_MMQ_TILE_Y_K;
           l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum,
                                                     VLLM_GGUF_MMQ_TILE_NE_K);
    __syncthreads();
  }

  vllm_gguf_mmq_write_back_mma_v2<scalar_t, mmq_x, mmq_y, need_check>(
      sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

template <typename scalar_t, int mmq_x, bool need_check>
static __device__ __forceinline__ void vllm_gguf_moe_iq4_xs_process_tile_v2(
    const char* __restrict__ x, const int offset_x, const int* __restrict__ y,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int row_dst_0, const int col_dst_0, const int stride_row_x,
    const int ncols_y, const int ncols_y_padded, const int top_k,
    const int ncols_dst, const int stride_col_dst, const int tile_x_max_i,
    const int kb0_start, const int kb0_stop) {
  constexpr int warp_size = WARP_SIZE_GGUF;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int blocks_per_iter = VLLM_GGUF_MMQ_ITER_K / QK_K;
  constexpr int tile_y_ints =
      vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K, nwarps * warp_size);

  extern __shared__ int data_mul_mat_q_v2[];
  int* tile_y = data_mul_mat_q_v2;
  int* tile_x = tile_y + tile_y_ints;

  float sum[mmq_x * mmq_y / (nwarps * warp_size)] = {0.0f};

  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    vllm_gguf_load_tiles_iq4_xs_v2<mmq_y, need_check>(
        x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

    vllm_gguf_load_moe_q8_1_tile_mma_v2<mmq_x>(
        y, tile_y, col_dst_0, kb0 * QK_K / (4 * QK8_1), ncols_y_padded);

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
    __syncthreads();

    vllm_gguf_load_moe_q8_1_tile_mma_v2<mmq_x>(
        y, tile_y, col_dst_0, kb0 * QK_K / (4 * QK8_1) + 1, ncols_y_padded);

    __syncthreads();
    vllm_gguf_vec_dot_q8_0_q8_1_mma_v2<mmq_x, mmq_y>(tile_x, tile_y, sum,
                                                     VLLM_GGUF_MMQ_TILE_NE_K);
    __syncthreads();
  }

  vllm_gguf_moe_mmq_write_back_mma_v2<scalar_t, mmq_x, mmq_y, need_check>(
      sum, dst, sorted_token_ids, col_dst_0, row_dst_0, ncols_dst,
      stride_col_dst, tile_x_max_i);
}

template <typename scalar_t, bool need_check>
static __global__ __launch_bounds__(
    WARP_SIZE_GGUF* VLLM_GGUF_MMQ_NWARPS,
    1) void vllm_gguf_mul_mat_q4_0_v2(const void* __restrict__ vx,
                                      const int* __restrict__ vy,
                                      scalar_t* __restrict__ dst,
                                      const int ncols_x, const int nrows_x,
                                      const int ncols_y,
                                      const int ncols_y_padded,
                                      const int nrows_dst) {
  constexpr int mmq_x = VLLM_GGUF_MMQ_X_Q4_0;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;

  const int it = blockIdx.x;
  const int jt = blockIdx.y;
  const int row_dst_0 = it * mmq_y;
  const int col_dst_0 = jt * mmq_x;
  const int blocks_per_row_x = ncols_x / QK4_0;
  const int tile_x_max_i = nrows_x - row_dst_0 - 1;
  const int tile_y_max_j = ncols_y - col_dst_0 - 1;
  const int offset_x = row_dst_0 * blocks_per_row_x;
  constexpr int sz = sizeof(block_q8_1_mmq_v2) / sizeof(int);
  const int offset_y = col_dst_0 * sz;

  vllm_gguf_mul_mat_q4_0_process_tile_v2<scalar_t, mmq_x, need_check>(
      (const char*)vx, offset_x, vy + offset_y,
      dst + col_dst_0 * nrows_dst + row_dst_0, blocks_per_row_x, ncols_y_padded,
      nrows_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_row_x);
}

template <typename scalar_t, bool need_check>
static __global__ __launch_bounds__(
    WARP_SIZE_GGUF* VLLM_GGUF_MMQ_NWARPS,
    1) void vllm_gguf_mul_mat_iq4_xs_v2(const void* __restrict__ vx,
                                        const int* __restrict__ vy,
                                        scalar_t* __restrict__ dst,
                                        const int ncols_x, const int nrows_x,
                                        const int ncols_y,
                                        const int ncols_y_padded,
                                        const int nrows_dst) {
  constexpr int mmq_x = VLLM_GGUF_MMQ_X_IQ4_XS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;

  const int it = blockIdx.x;
  const int jt = blockIdx.y;
  const int row_dst_0 = it * mmq_y;
  const int col_dst_0 = jt * mmq_x;
  const int blocks_per_row_x = ncols_x / QK_K;
  const int tile_x_max_i = nrows_x - row_dst_0 - 1;
  const int tile_y_max_j = ncols_y - col_dst_0 - 1;
  const int offset_x = row_dst_0 * blocks_per_row_x;
  constexpr int sz = sizeof(block_q8_1_mmq_v2) / sizeof(int);
  const int offset_y = col_dst_0 * sz;

  vllm_gguf_mul_mat_iq4_xs_process_tile_v2<scalar_t, mmq_x, need_check>(
      (const char*)vx, offset_x, vy + offset_y,
      dst + col_dst_0 * nrows_dst + row_dst_0, blocks_per_row_x, ncols_y_padded,
      nrows_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_row_x);
}

template <typename scalar_t, bool need_check>
static __global__ __launch_bounds__(
    WARP_SIZE_GGUF* VLLM_GGUF_MMQ_NWARPS,
    1) void vllm_gguf_moe_iq4_xs_v2(const void* __restrict__ vx,
                                    const int* __restrict__ vy,
                                    scalar_t* __restrict__ dst,
                                    const int* __restrict__ sorted_token_ids,
                                    const int* __restrict__ expert_ids,
                                    const int* __restrict__ num_tokens_post_padded,
                                    const int exp_stride, const int ncols_x,
                                    const int nrows_x, const int ncols_y,
                                    const int ncols_y_padded,
                                    const int nrows_dst, const int top_k) {
  constexpr int mmq_x = VLLM_GGUF_MOE_MMQ_X_IQ4_XS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int col_dst_0 = blockIdx.y * mmq_x;

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) {
    return;
  }
  if (col_dst_0 >= num_tokens_post_padded[0]) {
    return;
  }

  const char* x = (const char*)vx + exp_idx * exp_stride;
  const int blocks_per_row_x = ncols_x / QK_K;
  const int tile_x_max_i = nrows_x - row_dst_0 - 1;
  const int offset_x = row_dst_0 * blocks_per_row_x;
  const int ncols_dst = ncols_y * top_k;

  vllm_gguf_moe_iq4_xs_process_tile_v2<scalar_t, mmq_x, need_check>(
      x, offset_x, vy, dst, sorted_token_ids, row_dst_0, col_dst_0,
      blocks_per_row_x, ncols_y, ncols_y_padded, top_k, ncols_dst, nrows_dst,
      tile_x_max_i, 0, blocks_per_row_x);
}

template <typename scalar_t>
static void vllm_gguf_mul_mat_q4_0_q8_1_mmq_v2_cuda(
    const void* vx, const void* vy, scalar_t* dst, const int ncols_x,
    const int nrows_x, const int ncols_y, const int ncols_y_padded,
    const int nrows_dst, cudaStream_t stream) {
  constexpr int mmq_x = VLLM_GGUF_MMQ_X_Q4_0;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int tile_y_ints = vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K,
                                                nwarps * WARP_SIZE_GGUF);
  constexpr int shared_mem =
      (tile_y_ints + mmq_y * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0) * sizeof(int);

  const dim3 block_nums((nrows_x + mmq_y - 1) / mmq_y,
                        (ncols_y + mmq_x - 1) / mmq_x, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    vllm_gguf_mul_mat_q4_0_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, ncols_x, nrows_x, ncols_y, ncols_y_padded,
            nrows_dst);
  } else {
    constexpr bool need_check = true;
    vllm_gguf_mul_mat_q4_0_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, ncols_x, nrows_x, ncols_y, ncols_y_padded,
            nrows_dst);
  }
}

template <typename scalar_t>
static void vllm_gguf_moe_iq4_xs_q8_1_mmq_v2_cuda(
    const void* vx, const void* vy, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int ncols_y_padded, const int nrows_dst,
    const int top_k, const int tokens_post_padded, cudaStream_t stream) {
  constexpr int mmq_x = VLLM_GGUF_MOE_MMQ_X_IQ4_XS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int tile_y_ints = vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K,
                                                nwarps * WARP_SIZE_GGUF);
  constexpr int shared_mem =
      (tile_y_ints + mmq_y * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0) * sizeof(int);

  const dim3 block_nums((nrows_x + mmq_y - 1) / mmq_y,
                        tokens_post_padded / mmq_x, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    vllm_gguf_moe_iq4_xs_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, sorted_token_ids, expert_ids,
            num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
            ncols_y_padded, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    vllm_gguf_moe_iq4_xs_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, sorted_token_ids, expert_ids,
            num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
            ncols_y_padded, nrows_dst, top_k);
  }
}

template <typename scalar_t>
static void vllm_gguf_mul_mat_iq4_xs_q8_1_mmq_v2_cuda(
    const void* vx, const void* vy, scalar_t* dst, const int ncols_x,
    const int nrows_x, const int ncols_y, const int ncols_y_padded,
    const int nrows_dst, cudaStream_t stream) {
  constexpr int mmq_x = VLLM_GGUF_MMQ_X_IQ4_XS;
  constexpr int mmq_y = VLLM_GGUF_MMQ_Y;
  constexpr int nwarps = VLLM_GGUF_MMQ_NWARPS;
  constexpr int tile_y_ints = vllm_gguf_mmq_pad(mmq_x * VLLM_GGUF_MMQ_TILE_Y_K,
                                                nwarps * WARP_SIZE_GGUF);
  constexpr int shared_mem =
      (tile_y_ints + mmq_y * VLLM_GGUF_MMQ_MMA_TILE_X_K_Q8_0) * sizeof(int);

  const dim3 block_nums((nrows_x + mmq_y - 1) / mmq_y,
                        (ncols_y + mmq_x - 1) / mmq_x, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    vllm_gguf_mul_mat_iq4_xs_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, ncols_x, nrows_x, ncols_y, ncols_y_padded,
            nrows_dst);
  } else {
    constexpr bool need_check = true;
    vllm_gguf_mul_mat_iq4_xs_v2<scalar_t, need_check>
        <<<block_nums, block_dims, shared_mem, stream>>>(
            vx, (const int*)vy, dst, ncols_x, nrows_x, ncols_y, ncols_y_padded,
            nrows_dst);
  }
}
