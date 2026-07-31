// Device-side lifting steps, shared by all GPU kernels.
//
// The transform is the same CDF-5/3-style lifting as wavelet_cpu.hpp. It is
// decomposed into individual steps so that each step can be one kernel launch:
// the cross-step data dependency (step 3 reads s[] written by step 2) is
// resolved by the implicit global barrier between launches. This mirrors the
// CPU passes exactly and has no signal-size limit.
//
// All steps operate on a logical 1D signal of length n laid out with a stride:
//   element i lives at base[i * stride].
// stride=1 is a plain row; stride=row_length is a column of a 2D image. This
// lets the same steps serve row passes, column passes, and MRA sub-bands.
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

// Accessor for element i of a strided logical signal.
template <typename T>
__device__ __forceinline__ T& at(T* base, int i, int stride) {
    return base[i * stride];
}

// --- Forward steps (one thread per element of the relevant half) ---

// Deinterleave [x0 x1 x2 ...] -> [even... | odd...] from src into dst.
// Separate buffers avoid the in-place shuffle hazard; caller ping-pongs.
template <typename T>
__global__ void k_deinterleave(const T* src, T* dst, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    dst[(k) * stride] = src[(2 * k) * stride];            // even -> s[k]
    dst[(k + n2) * stride] = src[(2 * k + 1) * stride];   // odd  -> d[k]
}

// Strided copy of n elements from src to dst (used to move a transformed
// signal out of scratch back into the working buffer).
template <typename T>
__global__ void k_copy_strided(const T* src, T* dst, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i * stride] = src[i * stride];
}

// Inverse of k_deinterleave: [even | odd] -> interleaved.
template <typename T>
__global__ void k_interleave(const T* src, T* dst, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    dst[(2 * k) * stride] = src[k * stride];
    dst[(2 * k + 1) * stride] = src[(k + n2) * stride];
}

// Step 1 fwd: d[k] -= s[k].
template <typename T>
__global__ void k_fwd_predict1(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k + n2, stride) -= at(base, k, stride);
}

// Step 2 fwd: s[k] += d[k]/2.
template <typename T>
__global__ void k_fwd_update(T* base, int n2, int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n2) return;
    at(base, k, stride) += at(base, k + n2, stride) / T(2);
}

// Step 3 fwd: lifting of details (interior + two edge formulas).
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

// --- Inverse steps (exact reverse order) ---

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
