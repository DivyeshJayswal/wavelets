#pragma once
#include "wavelet.cuh"

static inline dim3 grid_for(int n2, int block) { return dim3((n2 + block - 1) / block); }

template <typename T>
void gpu_wavelet_forward(T* base, T* scratch, int n, int stride,
                         cudaStream_t stream = 0, int block = 256) {
    const int n2 = n >> 1;
    dim3 g = grid_for(n2, block);
    k_deinterleave<T><<<g, block, 0, stream>>>(base, scratch, n2, stride);

    k_fwd_predict1<T><<<g, block, 0, stream>>>(scratch, n2, stride);
    k_fwd_update<T><<<g, block, 0, stream>>>(scratch, n2, stride);
    k_fwd_predict2<T><<<g, block, 0, stream>>>(scratch, n2, stride);

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
