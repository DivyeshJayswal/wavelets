// Host-side orchestration of the device lifting steps.
//
// A single 1D transform = a fixed sequence of step-kernel launches on a strided
// signal. Deinterleave/interleave need a scratch buffer of the same length; the
// caller supplies it (reused across many rows/columns to avoid per-call malloc).
#pragma once
#include "wavelet.cuh"

// Launch config for n2 threads.
static inline dim3 grid_for(int n2, int block) { return dim3((n2 + block - 1) / block); }

// Forward transform of one strided signal of length n (n even).
// `base` and `scratch` are device pointers; scratch must hold >= n strided elems.
template <typename T>
void gpu_wavelet_forward(T* base, T* scratch, int n, int stride,
                         cudaStream_t stream = 0, int block = 256) {
    const int n2 = n >> 1;
    dim3 g = grid_for(n2, block);
    k_deinterleave<T><<<g, block, 0, stream>>>(base, scratch, n2, stride);
    // scratch now holds [even|odd]; copy back so we stay in `base`.
    k_fwd_predict1<T><<<g, block, 0, stream>>>(scratch, n2, stride);
    k_fwd_update<T><<<g, block, 0, stream>>>(scratch, n2, stride);
    k_fwd_predict2<T><<<g, block, 0, stream>>>(scratch, n2, stride);
    // Move result back into base (stride-aware copy via interleave's sibling).
    k_copy_strided<T><<<grid_for(n, block), block, 0, stream>>>(scratch, base, n, stride);
}

template <typename T>
void gpu_wavelet_inverse(T* base, T* scratch, int n, int stride,
                         cudaStream_t stream = 0, int block = 256) {
    const int n2 = n >> 1;
    dim3 g = grid_for(n2, block);
    k_inv_predict2<T><<<g, block, 0, stream>>>(base, n2, stride);
    k_inv_update<T><<<g, block, 0, stream>>>(base, n2, stride);
    k_inv_predict1<T><<<g, block, 0, stream>>>(base, n2, stride);
    k_interleave<T><<<g, block, 0, stream>>>(base, scratch, n2, stride);
    k_copy_strided<T><<<grid_for(n, block), block, 0, stream>>>(scratch, base, n, stride);
}
