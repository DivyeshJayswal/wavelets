#pragma once
#include <cstdio>

#include "wavelet.cuh"

template <typename T>
__global__ void k_row_forward_shared(T* img, int n, int W) {
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);
    T* d = s + (n >> 1);
    const int n2 = n >> 1;
    T* g = img + blockIdx.x * W;

    for (int k = threadIdx.x; k < n2; k += blockDim.x) {
        s[k] = g[2 * k];
        d[k] = g[2 * k + 1];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) d[k] -= s[k];
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) s[k] += d[k] / T(2);
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) {
        if (k == 0)
            d[0] += T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
        else if (k == n2 - 1)
            d[n2 - 1] += T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];
        else
            d[k] += (s[k - 1] - s[k + 1]) / T(4);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n; k += blockDim.x) g[k] = s[k];
}

template <typename T>
bool row_forward_shared(T* img, int W, int H, cudaStream_t st = 0) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    int maxShared;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    size_t need = size_t(W) * sizeof(T);
    if (need > size_t(maxShared)) {
        fprintf(stderr, "row_forward_shared: W=%d needs %zu B shared > %d B limit\n",
                W, need, maxShared);
        return false;
    }
    int block = W / 2 < 256 ? (W / 2 > 0 ? W / 2 : 1) : 256;
    k_row_forward_shared<T><<<H, block, need, st>>>(img, W, W);
    return true;
}
