// Item 8: multi-resolution access kernel.
//
// After mra_forward(img, W, H, levels), the level-L approximation (LL band)
// occupies the top-left (W>>L) x (H>>L) corner of the image, still stored with
// the full image row stride W (that is the whole point of the in-place MRA
// layout — coarser levels are sub-sampled corners, not separate buffers).
//
// The benefit MRA sells is *cheap access to a coarse view*: to show a zoomed-out
// thumbnail of a gigantic image you only touch the tiny LL corner. This kernel
// gathers an arbitrary (w x h) region of the level-L LL band, whose top-left is
// (x0,y0) in that band's own coordinates, into a compact contiguous output
// (row stride w) that a single cudaMemcpy sends to the host.
#pragma once
#include <cstdio>
#include <cstdlib>

#include "wavelet.cuh"  // CUDA_CHECK

// Gather: out[y*w + x] = img[(y0+y)*W + (x0+x)]. Coalesced along x for both
// reads (stride 1 within a row of the corner) and writes (compact output).
template <typename T>
__global__ void k_extract_region(const T* img, int W, int x0, int y0, int w,
                                  int h, T* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[y * w + x] = img[(y0 + y) * W + (x0 + x)];
}

// Level-L LL band dimensions for a W x H image (top-left corner).
static inline int level_dim(int full, int level) { return full >> level; }

// Extract a (w x h) region at (x0,y0) of the level-L LL band from a resident
// MRA image (d_img, row stride W) into device buffer d_out (>= w*h elements).
// Bounds are validated against the level's LL band size.
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
