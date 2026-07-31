#include <cmath>
#include <cstdio>
#include <vector>

#include "wavelet2d.cuh"
#include "wavelet_cpu.hpp"

static bool finite_all(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static bool check(int W, int H, int levels) {
    std::vector<float> orig(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            orig[y * W + x] = std::sin(0.05f * x) * std::cos(0.03f * y) * 50.f + (x ^ y) % 13;

    float *d_img, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_img, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, orig.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));

    mra_forward(d_img, d_scratch, W, H, levels);
    mra_inverse(d_img, d_scratch, W, H, levels);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> recon(W * H);
    CUDA_CHECK(cudaMemcpy(recon.data(), d_img, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_scratch));

    float err = max_abs_diff(orig, recon);
    bool ok = finite_all(recon) && err < WAVELET_RECON_TOL;
    printf("  %4dx%-4d levels=%d roundtrip=%.2e  %s\n", W, H, levels, err,
           ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    printf("test_2d: MRA round-trip (tol=%.0e)\n", WAVELET_RECON_TOL);
    bool all = true;
    all &= check(8, 8, 1);
    all &= check(64, 64, 1);
    all &= check(256, 256, 3);
    all &= check(512, 256, 4);
    all &= check(1024, 1024, 5);
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
