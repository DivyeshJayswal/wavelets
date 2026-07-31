#pragma once
#include <cstdio>
#include <cstdlib>

#include "wavelet.cuh"

template <typename T>
__global__ void k_extract_region(const T* img, int W, int x0, int y0, int w,
                                  int h, T* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[y * w + x] = img[(y0 + y) * W + (x0 + x)];
}

static inline int level_dim(int full, int level) { return full >> level; }

template <typename T>
void extract_region(const T* d_img, int W, int H, int level, int x0, int y0,
                    int w, int h, T* d_out, cudaStream_t st = 0) {
    int lw = level_dim(W, level), lh = level_dim(H, level);
    if (level < 0 || x0 < 0 || y0 < 0 || w <= 0 || h <= 0 || x0 + w > lw ||
        y0 + h > lh) {
        fprintf(stderr, "extract_region: region (%d,%d)+%dx%d out of level-%d band %dx%d\n",
                x0, y0, w, h, level, lw, lh);
        abort();
    }
    dim3 blk(32, 8);
    dim3 grid((w + blk.x - 1) / blk.x, (h + blk.y - 1) / blk.y);
    k_extract_region<T><<<grid, blk, 0, st>>>(d_img, W, x0, y0, w, h, d_out);
}
