// Item 9: 3D separable transform and MRA on volumetric data.
//
// A volume is row-major W x H x D: voxel (x,y,z) at z*(W*H) + y*W + x. Element
// strides are x:1, y:W, z:W*H. A 3D level is three separable 1D passes along x,
// y, z; each reuses the strided pass_forward/pass_inverse from wavelet2d.cuh.
//
// Passes on a sub-volume (lw x lh x ld) inside the full (W,H,D) volume are
// driven by looping over the orthogonal slices so the (line_stride, elem_stride)
// pair is always a real arithmetic stride — the same "sub-region shrinks, full
// stride stays" discipline as the 2D MRA. MRA recurses into the LLL octant.
//
// Constraint (inherited from the lifting scheme's boundary formula, same in 2D):
// predict2 reads s[n2-3], so every *processed* dimension must stay >= 8. Hence
// levels is bounded by min(W,H,D) >> (levels-1) >= 8.
#pragma once
#include "wavelet2d.cuh"

// One forward 3D level on the top-left-front (lw x lh x ld) sub-volume.
template <typename T>
void level_forward3d(T* vol, T* scratch, int lw, int lh, int ld, int W, int H,
                     cudaStream_t st = 0) {
    const size_t slice = size_t(W) * H;
    for (int z = 0; z < ld; z++)  // x-pass: rows within each z-slice
        pass_forward(vol + z * slice, scratch + z * slice, lw, lh, W, 1, st);
    for (int z = 0; z < ld; z++)  // y-pass: columns within each z-slice
        pass_forward(vol + z * slice, scratch + z * slice, lh, lw, 1, W, st);
    for (int y = 0; y < lh; y++)  // z-pass: depth lines for each y-row
        pass_forward(vol + size_t(y) * W, scratch + size_t(y) * W, ld, lw, 1, slice, st);
}

template <typename T>
void level_inverse3d(T* vol, T* scratch, int lw, int lh, int ld, int W, int H,
                     cudaStream_t st = 0) {
    const size_t slice = size_t(W) * H;
    for (int y = 0; y < lh; y++)  // z-pass inverse
        pass_inverse(vol + size_t(y) * W, scratch + size_t(y) * W, ld, lw, 1, slice, st);
    for (int z = 0; z < ld; z++)  // y-pass inverse
        pass_inverse(vol + z * slice, scratch + z * slice, lh, lw, 1, W, st);
    for (int z = 0; z < ld; z++)  // x-pass inverse
        pass_inverse(vol + z * slice, scratch + z * slice, lw, lh, W, 1, st);
}

// Multi-level 3D MRA: recurse into the LLL octant. W,H,D divisible by 2^levels.
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
