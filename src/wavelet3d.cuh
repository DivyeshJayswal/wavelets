#pragma once
#include "wavelet2d.cuh"

template <typename T>
void level_forward3d(T* vol, T* scratch, int lw, int lh, int ld, int W, int H,
                     cudaStream_t st = 0) {
    const size_t slice = size_t(W) * H;
    for (int z = 0; z < ld; z++)
        pass_forward(vol + z * slice, scratch + z * slice, lw, lh, W, 1, st);
    for (int z = 0; z < ld; z++)
        pass_forward(vol + z * slice, scratch + z * slice, lh, lw, 1, W, st);
    for (int y = 0; y < lh; y++)
        pass_forward(vol + size_t(y) * W, scratch + size_t(y) * W, ld, lw, 1, slice, st);
}

template <typename T>
void level_inverse3d(T* vol, T* scratch, int lw, int lh, int ld, int W, int H,
                     cudaStream_t st = 0) {
    const size_t slice = size_t(W) * H;
    for (int y = 0; y < lh; y++)
        pass_inverse(vol + size_t(y) * W, scratch + size_t(y) * W, ld, lw, 1, slice, st);
    for (int z = 0; z < ld; z++)
        pass_inverse(vol + z * slice, scratch + z * slice, lh, lw, 1, W, st);
    for (int z = 0; z < ld; z++)
        pass_inverse(vol + z * slice, scratch + z * slice, lw, lh, W, 1, st);
}

template <typename T>
void mra_forward3d(T* vol, T* scratch, int W, int H, int D, int levels,
                   cudaStream_t st = 0) {
    int lw = W, lh = H, ld = D;
    for (int l = 0; l < levels; l++) {
        level_forward3d(vol, scratch, lw, lh, ld, W, H, st);
        lw >>= 1; lh >>= 1; ld >>= 1;
    }
}

template <typename T>
void mra_inverse3d(T* vol, T* scratch, int W, int H, int D, int levels,
                   cudaStream_t st = 0) {
    for (int l = levels - 1; l >= 0; l--)
        level_inverse3d(vol, scratch, W >> l, H >> l, D >> l, W, H, st);
}
