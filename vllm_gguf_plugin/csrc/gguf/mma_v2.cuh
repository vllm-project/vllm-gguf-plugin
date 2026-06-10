#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#ifndef GGML_CUDA_CC_TURING
  #define GGML_CUDA_CC_TURING 750
#endif

#ifndef GGML_CUDA_CC_AMPERE
  #define GGML_CUDA_CC_AMPERE 800
#endif

#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  #define VLLM_GGUF_TURING_MMA_AVAILABLE
#endif

static __device__ __forceinline__ void vllm_gguf_no_device_code() {
#ifdef __CUDA_ARCH__
  asm("trap;");
#endif
}

namespace vllm_gguf_mma {

enum data_layout {
  DATA_LAYOUT_I_MAJOR = 0,
};

template <int I_, int J_, typename T, data_layout dl_ = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;
  static constexpr int ne = I * J / 32;

  T x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return ((l / 2) * 8) + (threadIdx.x / 4);
    } else {
      vllm_gguf_no_device_code();
      return -1;
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 8) {
      return (l * 4) + (threadIdx.x % 4);
    } else if constexpr (I == 16 && J == 8) {
      return ((threadIdx.x % 4) * 2) + (l % 2);
    } else {
      vllm_gguf_no_device_code();
      return -1;
    }
  }
};

template <int I, int J, typename T, data_layout dl>
static __device__ __forceinline__ void load_generic(tile<I, J, T, dl>& t,
                                                    const T* __restrict__ xs0,
                                                    const int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix(tile<16, 8, T, dl>& t,
                                                     const T* __restrict__ xs0,
                                                     const int stride) {
#ifdef VLLM_GGUF_TURING_MMA_AVAILABLE
  int* xi = (int*)t.x;
  const int* xs = (const int*)xs0 + (threadIdx.x % t.I) * stride +
                  (threadIdx.x / t.I) * (t.J / 2);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
#else
  load_generic(t, xs0, stride);
#endif
}

static __device__ __forceinline__ void mma(tile<16, 8, int>& D,
                                           const tile<16, 8, int>& A,
                                           const tile<8, 8, int>& B) {
#ifdef VLLM_GGUF_TURING_MMA_AVAILABLE
  #if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]),
        "r"(B.x[1]));
  #else
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(D.x[0]), "+r"(D.x[1])
      : "r"(A.x[0]), "r"(B.x[0]));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[1]), "r"(B.x[0]));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(D.x[0]), "+r"(D.x[1])
      : "r"(A.x[2]), "r"(B.x[1]));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[3]), "r"(B.x[1]));
  #endif
#else
  (void)D;
  (void)A;
  (void)B;
  vllm_gguf_no_device_code();
#endif
}

}  // namespace vllm_gguf_mma
