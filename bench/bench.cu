#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

#include "morton.hpp"
#include "wavelet2d.cuh"
#include "wavelet_coalesced.cuh"
#include "wavelet_tiled.cuh"

static const int WARMUP = 5, REPS = 30;

template <typename F>
static float time_ms(F&& fn) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < WARMUP; i++) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < REPS; i++) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / REPS;
}

static float gbps(size_t bytes, float ms) { return bytes / (ms * 1e-3f) / 1e9f; }

template <typename T>
static T* make_plane(int W, int H) {
    std::vector<T> h(size_t(W) * H);
    for (size_t i = 0; i < h.size(); i++) h[i] = T(float(i % 251));
    T* d; CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static void mra_table() {
    printf("\n### MRA forward throughput (fp32, levels=1)\n\n");
    printf("| Size | time (ms) | Melem/s |\n|---|---|---|\n");
    for (int S : {256, 512, 1024, 2048, 4096}) {
        float* img = make_plane<float>(S, S);
        float* scr; CUDA_CHECK(cudaMalloc(&scr, size_t(S) * S * sizeof(float)));
        float ms = time_ms([&] { mra_forward(img, scr, S, S, 1); });
        printf("| %d^2 | %.3f | %.0f |\n", S, ms, (double(S) * S) / (ms * 1e-3) / 1e6);
        cudaFree(img); cudaFree(scr);
    }
}

static void dtype_table() {
    printf("\n### fp32 vs fp16 MRA forward (levels=3)\n\n");
    printf("| Size | fp32 (ms) | fp16 (ms) | speedup |\n|---|---|---|---|\n");
    for (int S : {512, 1024, 2048, 4096}) {
        float* f = make_plane<float>(S, S);
        float* fs; CUDA_CHECK(cudaMalloc(&fs, size_t(S) * S * sizeof(float)));
        __half* h = make_plane<__half>(S, S);
        __half* hs; CUDA_CHECK(cudaMalloc(&hs, size_t(S) * S * sizeof(__half)));
        float mf = time_ms([&] { mra_forward(f, fs, S, S, 3); });
        float mh = time_ms([&] { mra_forward(h, hs, S, S, 3); });
        printf("| %d^2 | %.3f | %.3f | %.2fx |\n", S, mf, mh, mf / mh);
        cudaFree(f); cudaFree(fs); cudaFree(h); cudaFree(hs);
    }
}

static void column_correctness(int S) {
    size_t n = size_t(S) * S;
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++) h[i] = float((i * 7 % 251) - 125);
    float *a, *b, *scr, *tb, *ts;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scr, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tb, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ts, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    column_pass_naive(a, scr, S, S);
    column_pass_transpose(b, tb, ts, S, S);
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    float m = 0.f;
    for (size_t i = 0; i < n; i++) m = fmaxf(m, fabsf(ha[i] - hb[i]));
    printf("_naive vs transpose max diff at %d^2 = %.2e (%s)_\n\n", S, m,
           m < 1e-3f ? "match" : "MISMATCH");
    cudaFree(a); cudaFree(b); cudaFree(scr); cudaFree(tb); cudaFree(ts);
}

static void column_table() {
    printf("\n### Column pass: naive strided vs coalesced transpose (fp32, item 6)\n\n");
    column_correctness(1024);
    printf("| Size | naive (ms) | naive GB/s | transpose (ms) | transp GB/s | speedup |\n");
    printf("|---|---|---|---|---|---|\n");
    for (int S : {512, 1024, 2048, 4096}) {
        size_t bytes = 2ull * S * S * sizeof(float);
        float *img = make_plane<float>(S, S), *scr, *tb, *ts;
        CUDA_CHECK(cudaMalloc(&scr, size_t(S) * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tb, size_t(S) * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ts, size_t(S) * S * sizeof(float)));
        float mn = time_ms([&] { column_pass_naive(img, scr, S, S); });
        float mt = time_ms([&] { column_pass_transpose(img, tb, ts, S, S); });
        printf("| %d^2 | %.3f | %.1f | %.3f | %.1f | %.2fx |\n", S, mn,
               gbps(bytes, mn), mt, gbps(bytes, mt), mn / mt);
        cudaFree(img); cudaFree(scr); cudaFree(tb); cudaFree(ts);
    }
}

static void tiled_table() {
    printf("\n### Row pass: step-per-launch vs shared-memory tile (fp32, item 7)\n\n");
    printf("| Size | baseline (ms) | base GB/s | shared (ms) | shared GB/s | speedup |\n");
    printf("|---|---|---|---|---|---|\n");
    for (int S : {512, 1024, 2048, 4096}) {
        size_t bytes = 2ull * S * S * sizeof(float);
        float *img = make_plane<float>(S, S), *scr;
        CUDA_CHECK(cudaMalloc(&scr, size_t(S) * S * sizeof(float)));
        float mb = time_ms([&] { pass_forward(img, scr, S, S, S, 1); });

        bool avail = row_forward_shared(img, S, S);
        cudaDeviceSynchronize();
        if (!avail) { printf("| %d^2 | %.3f | %.1f | n/a (row > smem) | | |\n", S, mb, gbps(bytes, mb)); cudaFree(img); cudaFree(scr); continue; }
        float ms = time_ms([&] { row_forward_shared(img, S, S); });
        printf("| %d^2 | %.3f | %.1f | %.3f | %.1f | %.2fx |\n", S, mb,
               gbps(bytes, mb), ms, gbps(bytes, ms), mb / ms);
        cudaFree(img); cudaFree(scr);
    }
}

__global__ void k_stencil_rowmajor(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H - 1) return;
    out[y * W + x] = in[y * W + x] + in[(y + 1) * W + x];
}
__global__ void k_stencil_morton(const float* m, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H - 1) return;
    out[morton_encode(x, y)] = m[morton_encode(x, y)] + m[morton_encode(x, y + 1)];
}
__global__ void k_to_morton(const float* in, float* m, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    m[morton_encode(x, y)] = in[y * W + x];
}

static void morton_table() {
    printf("\n### Layout: row-major vs Morton for a vertical-neighbour stencil (item 5)\n\n");
    printf("| Size | row-major (ms) | Morton (ms) | speedup |\n|---|---|---|---|\n");
    for (int S : {512, 1024, 2048, 4096}) {
        size_t n = size_t(S) * S;
        float *in = make_plane<float>(S, S), *out, *m;
        CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m, n * sizeof(float)));
        dim3 blk(32, 8), grid((S + 31) / 32, (S + 7) / 8);
        k_to_morton<<<grid, blk>>>(in, m, S, S);
        float mr = time_ms([&] { k_stencil_rowmajor<<<grid, blk>>>(in, out, S, S); });
        float mm = time_ms([&] { k_stencil_morton<<<grid, blk>>>(m, out, S, S); });
        printf("| %d^2 | %.4f | %.4f | %.2fx |\n", S, mr, mm, mr / mm);
        cudaFree(in); cudaFree(out); cudaFree(m);
    }
}

int main() {
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("# Benchmark results\n\nGPU: %s (sm_%d%d), warmup=%d reps=%d\n",
           p.name, p.major, p.minor, WARMUP, REPS);
    mra_table();
    dtype_table();
    column_table();
    tiled_table();
    morton_table();
    printf("\n_Paste these tables into report.md's FILL slots._\n");
    return 0;
}
