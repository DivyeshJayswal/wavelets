#pragma once
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            abort();                                                           \
        }                                                                      \
    } while (0)

template <typename T>
__device__ __forceinline__ T& at(T* base, int i, int stride) {
    return base[i * stride];
}

template <typename T>
__global__ void k_deinterleave(const T* src, T* dst, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    dst[(k) * stride] = src[(2 * k) * stride];
    dst[(k + n2) * stride] = src[(2 * k + 1) * stride];
}

template <typename T>
__global__ void k_copy_strided(const T* src, T* dst, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i * stride] = src[i * stride];
}

template <typename T>
__global__ void k_interleave(const T* src, T* dst, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    dst[(2 * k) * stride] = src[k * stride];
    dst[(2 * k + 1) * stride] = src[(k + n2) * stride];
}

template <typename T>
__global__ void k_fwd_predict1(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k + n2, stride) -= at(base, k, stride);
}

template <typename T>
__global__ void k_fwd_update(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k, stride) += at(base, k + n2, stride) / T(2);
}

template <typename T>
__global__ void k_fwd_predict2(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    T* s = base;
    T* d = base + n2 * stride;
    if (k == 0) {
        d[0] += T(0.75) * s[0] - T(1.0) * s[stride] + T(0.25) * s[2 * stride];
    } else if (k == n2 - 1) {
        d[(n2 - 1) * stride] += T(0.25) * s[(n2 - 1) * stride]
                              - T(1.0) * s[(n2 - 2) * stride]
                              + T(0.75) * s[(n2 - 3) * stride];
    } else {
        d[k * stride] += (s[(k - 1) * stride] - s[(k + 1) * stride]) / T(4);
    }
}

template <typename T>
__global__ void k_inv_predict2(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    T* s = base;
    T* d = base + n2 * stride;
    if (k == 0) {
        d[0] -= T(0.75) * s[0] - T(1.0) * s[stride] + T(0.25) * s[2 * stride];
    } else if (k == n2 - 1) {
        d[(n2 - 1) * stride] -= T(0.25) * s[(n2 - 1) * stride]
                              - T(1.0) * s[(n2 - 2) * stride]
                              + T(0.75) * s[(n2 - 3) * stride];
    } else {
        d[k * stride] -= (s[(k - 1) * stride] - s[(k + 1) * stride]) / T(4);
    }
}

template <typename T>
__global__ void k_inv_update(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k, stride) -= at(base, k + n2, stride) / T(2);
}

template <typename T>
__global__ void k_inv_predict1(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k + n2, stride) += at(base, k, stride);
}
