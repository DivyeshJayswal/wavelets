#pragma once
#include "wavelet2d.cuh"

#define TT_TILE 32

template <typename T>
__global__ void k_transpose(const T* in, T* out, int cols, int rows) {
    __shared__ T tile[TT_TILE][TT_TILE + 1];
    int x = blockIdx.x * TT_TILE + threadIdx.x;
    int y = blockIdx.y * TT_TILE + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    int xo = blockIdx.y * TT_TILE + threadIdx.x;
    int yo = blockIdx.x * TT_TILE + threadIdx.y;
    if (xo < rows && yo < cols) out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}

template <typename T>
static void transpose(const T* in, T* out, int cols, int rows, cudaStream_t st = 0) {
    dim3 blk(TT_TILE, TT_TILE);
    dim3 grid((cols + TT_TILE - 1) / TT_TILE, (rows + TT_TILE - 1) / TT_TILE);
    k_transpose<T><<<grid, blk, 0, st>>>(in, out, cols, rows);
}

template <typename T>
void column_pass_naive(T* img, T* scratch, int W, int H, cudaStream_t st = 0) {
    pass_forward(img, scratch, H, W, 1, W, st);
}

template <typename T>
void column_pass_transpose(T* img, T* tbuf, T* tscratch, int W, int H,
                           cudaStream_t st = 0) {
    transpose(img, tbuf, W, H, st);
    pass_forward(tbuf, tscratch, H, W, H, 1, st);
    transpose(tbuf, img, H, W, st);
}
