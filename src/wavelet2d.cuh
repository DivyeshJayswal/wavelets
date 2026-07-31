#pragma once
#include "wavelet_gpu.cuh"

template <typename T>
__global__ void k2_deinterleave(const T* src, T* dst, int n2, int lines,
                                int line_stride, int elem_stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int L = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= n2 || L >= lines) return;
    const T* s = src + L * line_stride;
    T* o = dst + L * line_stride;
    o[k * elem_stride] = s[(2 * k) * elem_stride];
    o[(k + n2) * elem_stride] = s[(2 * k + 1) * elem_stride];
}

template <typename T>
__global__ void k2_interleave(const T* src, T* dst, int n2, int lines,
                              int line_stride, int elem_stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int L = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= n2 || L >= lines) return;
    const T* s = src + L * line_stride;
    T* o = dst + L * line_stride;
    o[(2 * k) * elem_stride] = s[k * elem_stride];
    o[(2 * k + 1) * elem_stride] = s[(k + n2) * elem_stride];
}

template <typename T>
__global__ void k2_copy(const T* src, T* dst, int n, int lines,
                        int line_stride, int elem_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int L = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || L >= lines) return;
    dst[L * line_stride + i * elem_stride] = src[L * line_stride + i * elem_stride];
}

#define WAVE2D_STEP(NAME, BODY)                                                    \
    template <typename T>                                                          \
    __global__ void NAME(T* base, int n2, int lines, int line_stride,              \
                         int elem_stride) {                                        \
        int k = blockIdx.x * blockDim.x + threadIdx.x;                             \
        int L = blockIdx.y * blockDim.y + threadIdx.y;                             \
        if (k >= n2 || L >= lines) return;                                         \
        T* s = base + L * line_stride;                                             \
        T* d = s + n2 * elem_stride;                                               \
        const int e = elem_stride;                                                 \
        BODY                                                                       \
    }

WAVE2D_STEP(k2_fwd_predict1, { d[k * e] -= s[k * e]; })
WAVE2D_STEP(k2_fwd_update, { s[k * e] += d[k * e] / T(2); })
WAVE2D_STEP(k2_inv_update, { s[k * e] -= d[k * e] / T(2); })
WAVE2D_STEP(k2_inv_predict1, { d[k * e] += s[k * e]; })

WAVE2D_STEP(k2_fwd_predict2, {
    if (k == 0)
        d[0] += T(0.75) * s[0] - T(1.0) * s[e] + T(0.25) * s[2 * e];
    else if (k == n2 - 1)
        d[(n2 - 1) * e] += T(0.25) * s[(n2 - 1) * e] - T(1.0) * s[(n2 - 2) * e] +
                           T(0.75) * s[(n2 - 3) * e];
    else
        d[k * e] += (s[(k - 1) * e] - s[(k + 1) * e]) / T(4);
})
WAVE2D_STEP(k2_inv_predict2, {
    if (k == 0)
        d[0] -= T(0.75) * s[0] - T(1.0) * s[e] + T(0.25) * s[2 * e];
    else if (k == n2 - 1)
        d[(n2 - 1) * e] -= T(0.25) * s[(n2 - 1) * e] - T(1.0) * s[(n2 - 2) * e] +
                           T(0.75) * s[(n2 - 3) * e];
    else
        d[k * e] -= (s[(k - 1) * e] - s[(k + 1) * e]) / T(4);
})
#undef WAVE2D_STEP

static inline dim3 grid2(int nx, int lines, dim3 blk) {
    return dim3((nx + blk.x - 1) / blk.x, (lines + blk.y - 1) / blk.y);
}

template <typename T>
void pass_forward(T* base, T* scratch, int n, int lines, int line_stride,
                  int elem_stride, cudaStream_t st = 0) {
    const int n2 = n >> 1;
    dim3 blk(32, 8);
    dim3 gh = grid2(n2, lines, blk), gf = grid2(n, lines, blk);
    k2_deinterleave<T><<<gh, blk, 0, st>>>(base, scratch, n2, lines, line_stride, elem_stride);
    k2_fwd_predict1<T><<<gh, blk, 0, st>>>(scratch, n2, lines, line_stride, elem_stride);
    k2_fwd_update<T><<<gh, blk, 0, st>>>(scratch, n2, lines, line_stride, elem_stride);
    k2_fwd_predict2<T><<<gh, blk, 0, st>>>(scratch, n2, lines, line_stride, elem_stride);
    k2_copy<T><<<gf, blk, 0, st>>>(scratch, base, n, lines, line_stride, elem_stride);
}

template <typename T>
void pass_inverse(T* base, T* scratch, int n, int lines, int line_stride,
                  int elem_stride, cudaStream_t st = 0) {
    const int n2 = n >> 1;
    dim3 blk(32, 8);
    dim3 gh = grid2(n2, lines, blk), gf = grid2(n, lines, blk);
    k2_inv_predict2<T><<<gh, blk, 0, st>>>(base, n2, lines, line_stride, elem_stride);
    k2_inv_update<T><<<gh, blk, 0, st>>>(base, n2, lines, line_stride, elem_stride);
    k2_inv_predict1<T><<<gh, blk, 0, st>>>(base, n2, lines, line_stride, elem_stride);
    k2_interleave<T><<<gh, blk, 0, st>>>(base, scratch, n2, lines, line_stride, elem_stride);
    k2_copy<T><<<gf, blk, 0, st>>>(scratch, base, n, lines, line_stride, elem_stride);
}

template <typename T>
void level_forward(T* img, T* scratch, int lw, int lh, int W, cudaStream_t st = 0) {
    pass_forward(img, scratch, lw, lh, W, 1, st);
    pass_forward(img, scratch, lh, lw, 1, W, st);
}

template <typename T>
void level_inverse(T* img, T* scratch, int lw, int lh, int W, cudaStream_t st = 0) {
    pass_inverse(img, scratch, lh, lw, 1, W, st);
    pass_inverse(img, scratch, lw, lh, W, 1, st);
}

template <typename T>
void mra_forward(T* img, T* scratch, int W, int H, int levels, cudaStream_t st = 0) {
    int lw = W, lh = H;
    for (int l = 0; l < levels; l++) {
        level_forward(img, scratch, lw, lh, W, st);
        lw >>= 1;
        lh >>= 1;
    }
}

template <typename T>
void mra_inverse(T* img, T* scratch, int W, int H, int levels, cudaStream_t st = 0) {

    for (int l = levels - 1; l >= 0; l--) {
        int lw = W >> l, lh = H >> l;
        level_inverse(img, scratch, lw, lh, W, st);
    }
}
