// Item 1 validation: GPU 1D forward/inverse must match the CPU oracle and
// round-trip within tolerance, across multiple signal sizes.
#include <cmath>
#include <cstdio>
#include <vector>

#include "wavelet_cpu.hpp"
#include "wavelet_gpu.cuh"

// Max abs difference between two host vectors.
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Run GPU forward then inverse on a copy; return round-trip error and,
// separately, the forward-coefficient error vs the CPU oracle.
static bool check_size(int n) {
    std::vector<float> orig(n);
    for (int i = 0; i < n; i++) orig[i] = std::sin(0.1f * i) * 100.f + i % 7;

    // CPU oracle forward coefficients.
    std::vector<float> cpu = orig;
    wavelet_forward(n, cpu.data());

    // GPU forward.
    float *d_base, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_base, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base, orig.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    gpu_wavelet_forward(d_base, d_scratch, n, /*stride=*/1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_fwd(n);
    CUDA_CHECK(cudaMemcpy(gpu_fwd.data(), d_base, n * sizeof(float), cudaMemcpyDeviceToHost));

    gpu_wavelet_inverse(d_base, d_scratch, n, /*stride=*/1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_recon(n);
    CUDA_CHECK(cudaMemcpy(gpu_recon.data(), d_base, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_scratch));

    float fwd_err = max_abs_diff(cpu, gpu_fwd);
    float rt_err = max_abs_diff(orig, gpu_recon);
    bool ok = fwd_err < WAVELET_RECON_TOL && rt_err < WAVELET_RECON_TOL;
    printf("  n=%-9d fwd_vs_cpu=%.2e roundtrip=%.2e  %s\n", n, fwd_err, rt_err,
           ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    printf("test_1d: GPU vs CPU oracle (tol=%.0e)\n", WAVELET_RECON_TOL);
    bool all = true;
    for (int n : {16, 256, 1024, 1 << 16, 1 << 20, 1 << 24}) all &= check_size(n);
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
