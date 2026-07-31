// Item 7: shared-memory tiling.
//
// The baseline (wavelet2d.cuh) runs each lifting step as its own kernel, so a
// row is streamed through global memory ~5 times (deinterleave + 3 steps +
// copy-back), each step a full read+write. Here one block owns a whole row: it
// stages the row in shared memory, runs the entire forward lifting there, and
// writes the result once. Global traffic drops to 1 read + 1 write per element.
//
// The "tile" is the full row, so boundaries are exact and the result matches
// the baseline row pass bit-for-bit. Full 2D tiles with halo exchange (where
// results legitimately differ at tile seams, as the brief notes) are discussed
// in the report as the next step; this version isolates the traffic win.
//
// Constraint: the row (n elements) must fit in shared memory. row_forward_shared
// checks this and reports if a width exceeds the device's per-block limit.
#pragma once
#include <cstdio>

#include "wavelet.cuh"  // CUDA_CHECK

// One block per row. Dynamic shared holds the n-element row as [approx | detail].
template <typename T>
__global__ void k_row_forward_shared(T* img, int n, int W) {
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);  // s[0..n2): approx, s[n2..n): detail (d)
    T* d = s + (n >> 1);
    const int n2 = n >> 1;
    T* g = img + blockIdx.x * W;  // this row (elem stride 1)

    // Load + deinterleave: even -> s[k], odd -> d[k]. Reads global, writes shared.
    for (int k = threadIdx.x; k < n2; k += blockDim.x) {
        s[k] = g[2 * k];
        d[k] = g[2 * k + 1];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) d[k] -= s[k];  // step 1
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) s[k] += d[k] / T(2);  // step 2
    __syncthreads();
    for (int k = threadIdx.x; k < n2; k += blockDim.x) {  // step 3 (predict2)
        if (k == 0)
            d[0] += T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
        else if (k == n2 - 1)
            d[n2 - 1] += T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];
        else
            d[k] += (s[k - 1] - s[k + 1]) / T(4);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n; k += blockDim.x) g[k] = s[k];  // write [s|d]
}

// Forward row pass over a full W x H image, one shared-memory tile per row.
// Returns false (without launching) if a row does not fit in shared memory.
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
