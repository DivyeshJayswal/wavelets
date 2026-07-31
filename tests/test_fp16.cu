#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

#include "wavelet2d.cuh"

static const float FP16_TOL = 5.0f;

static bool check(int W, int H, int levels) {
    std::vector<float> orig(size_t(W) * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            orig[y * W + x] = std::sin(0.05f * x) * std::cos(0.03f * y) * 20.f;

    std::vector<__half> host(orig.size());
    for (size_t i = 0; i < orig.size(); i++) host[i] = __float2half(orig[i]);

    __half *d_img, *d_scratch;
    size_t bytes = host.size() * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_img, bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch, bytes));
    CUDA_CHECK(cudaMemcpy(d_img, host.data(), bytes, cudaMemcpyHostToDevice));

    mra_forward(d_img, d_scratch, W, H, levels);
    mra_inverse(d_img, d_scratch, W, H, levels);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host.data(), d_img, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_scratch));

    float err = 0.f;
    bool finite = true;
    for (size_t i = 0; i < orig.size(); i++) {
        float r = __half2float(host[i]);
        if (!std::isfinite(r)) finite = false;
        err = std::max(err, std::fabs(r - orig[i]));
    }
    bool ok = finite && err < FP16_TOL;
    printf("  %dx%d levels=%d fp16 roundtrip=%.3e  %s\n", W, H, levels, err,
           ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    printf("test_fp16: FP16 MRA round-trip (tol=%.1f)\n", FP16_TOL);
    bool all = true;
    all &= check(64, 64, 1);
    all &= check(256, 256, 3);
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
