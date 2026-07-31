// Item 8 validation: the extraction kernel must return exactly the pixels of the
// requested level-L LL region as they sit in the resident MRA image.
#include <cmath>
#include <cstdio>
#include <vector>

#include "mra_access.cuh"
#include "wavelet2d.cuh"

static bool check(int W, int H, int levels, int level, int x0, int y0, int w, int h) {
    std::vector<float> orig(size_t(W) * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            orig[y * W + x] = std::sin(0.05f * x) * std::cos(0.03f * y) * 50.f + (x ^ y) % 13;

    float *d_img, *d_scratch, *d_out;
    CUDA_CHECK(cudaMalloc(&d_img, size_t(W) * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, size_t(W) * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, size_t(w) * h * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, orig.data(), size_t(W) * H * sizeof(float), cudaMemcpyHostToDevice));

    mra_forward(d_img, d_scratch, W, H, levels);

    // Reference: full transformed image on host, then crop the same region.
    std::vector<float> full(size_t(W) * H);
    CUDA_CHECK(cudaMemcpy(full.data(), d_img, size_t(W) * H * sizeof(float), cudaMemcpyDeviceToHost));

    extract_region(d_img, W, H, level, x0, y0, w, h, d_out);
    std::vector<float> got(size_t(w) * h);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, size_t(w) * h * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (got[y * w + x] != full[(y0 + y) * W + (x0 + x)]) ok = false;

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_out));
    printf("  %dx%d levels=%d extract L%d (%d,%d)+%dx%d  %s\n", W, H, levels,
           level, x0, y0, w, h, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    printf("test_access: multi-resolution extraction\n");
    bool all = true;
    all &= check(256, 256, 3, 3, 0, 0, 32, 32);    // whole coarsest LL band
    all &= check(256, 256, 3, 1, 10, 20, 50, 40);  // sub-region of level-1 LL
    all &= check(512, 256, 4, 2, 5, 5, 60, 30);    // non-square, offset region
    printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
