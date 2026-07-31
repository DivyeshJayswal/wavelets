#include <cmath>
#include <cstdio>
#include <vector>

#include "wavelet3d.cuh"
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

static bool check(int W, int H, int D, int levels) {
    std::vector<float> orig(size_t(W) * H * D);
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                orig[(size_t(z) * H + y) * W + x] =
                    std::sin(0.05f * x) * std::cos(0.03f * y) * std::sin(0.02f * z) * 40.f +
                    (x ^ y ^ z) % 11;

    float *d_vol, *d_scratch;
    size_t bytes = orig.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_vol, bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch, bytes));
    CUDA_CHECK(cudaMemcpy(d_vol, orig.data(), bytes, cudaMemcpyHostToDevice));

    mra_forward3d(d_vol, d_scratch, W, H, D, levels);
    mra_inverse3d(d_vol, d_scratch, W, H, D, levels);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> recon(orig.size());
    CUDA_CHECK(cudaMemcpy(recon.data(), d_vol, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_scratch));

    float err = max_abs_diff(orig, recon);
    bool ok = finite_all(recon) && err < WAVELET_RECON_TOL;
    printf("  %dx%dx%d levels=%d roundtrip=%.2e  %s\n", W, H, D, levels, err,
           ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    printf("test_3d: 3D MRA round-trip (tol=%.0e)\n", WAVELET_RECON_TOL);
    bool all = true;
    all &= check(8, 8, 8, 1);
    all &= check(32, 32, 32, 2);

    all &= check(128, 64, 32, 3);
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
