// Item 6: coalesced column pass.
//
// The naive column pass transforms each column in place with elem_stride = W:
// consecutive threads (varying the element index within a column) touch
// addresses W apart, so every warp scatters across many cache lines —
// uncoalesced. The row pass, by contrast, is already coalesced (elem_stride=1).
//
// Fix: don't read columns strided at all. Transpose the image (a coalesced,
// shared-memory-tiled operation), run the *row* pass on the transpose (now
// coalesced), and transpose back. This trades strided global traffic for two
// coalesced transposes plus a coalesced pass.
//
// Both variants operate on a full W x H image and produce identical results;
// bench/bench.cu times them head-to-head (before/after bandwidth). Deeper MRA
// levels apply the same idea to the (lw x lh) corner with an added offset.
#pragma once
#include "wavelet2d.cuh"

#define TT_TILE 32

// Transpose a (rows x cols) row-major array `in` (row stride = cols) into `out`
// (dims cols x rows, row stride = rows). Padded tile avoids shared-bank conflicts.
template <typename T>
__global__ void k_transpose(const T* in, T* out, int cols, int rows) {
    __shared__ T tile[TT_TILE][TT_TILE + 1];
    int x = blockIdx.x * TT_TILE + threadIdx.x;  // col in input
    int y = blockIdx.y * TT_TILE + threadIdx.y;  // row in input
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    int xo = blockIdx.y * TT_TILE + threadIdx.x;  // col in output ( < rows )
    int yo = blockIdx.x * TT_TILE + threadIdx.y;  // row in output ( < cols )
    if (xo < rows && yo < cols) out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}

template <typename T>
static void transpose(const T* in, T* out, int cols, int rows, cudaStream_t st = 0) {
    dim3 blk(TT_TILE, TT_TILE);
    dim3 grid((cols + TT_TILE - 1) / TT_TILE, (rows + TT_TILE - 1) / TT_TILE);
    k_transpose<T><<<grid, blk, 0, st>>>(in, out, cols, rows);
}

// Naive strided column forward pass over a full W x H image (elem_stride = W).
template <typename T>
void column_pass_naive(T* img, T* scratch, int W, int H, cudaStream_t st = 0) {
    pass_forward(img, scratch, /*n=*/H, /*lines=*/W, /*line_stride=*/1,
                 /*elem_stride=*/W, st);
}

// Coalesced column forward pass: transpose -> row pass -> transpose back.
// `tbuf` and `tscratch` are W*H device scratch buffers for the transpose.
template <typename T>
void column_pass_transpose(T* img, T* tbuf, T* tscratch, int W, int H,
                           cudaStream_t st = 0) {
    transpose(img, tbuf, /*cols=*/W, /*rows=*/H, st);   // tbuf: W rows x H cols, stride H
    pass_forward(tbuf, tscratch, /*n=*/H, /*lines=*/W, /*line_stride=*/H,
                 /*elem_stride=*/1, st);                // coalesced row pass
    transpose(tbuf, img, /*cols=*/H, /*rows=*/W, st);   // back to W x H
}
