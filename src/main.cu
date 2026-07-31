#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <string>

#include "image_io.hpp"
#include "wavelet2d.cuh"

static int pad_to(int n, int levels) {
    int m = 1 << levels;
    return ((n + m - 1) / m) * m;
}

static std::vector<float> pad_image(const Image& src, int dw, int dh) {
    std::vector<float> out(size_t(dw) * dh);
    for (int y = 0; y < dh; y++) {
        int sy = y < src.h ? y : src.h - 1;
        for (int x = 0; x < dw; x++) {
            int sx = x < src.w ? x : src.w - 1;
            out[size_t(y) * dw + x] = src.data[size_t(sy) * src.w + sx];
        }
    }
    return out;
}

static std::vector<float> crop_image(const std::vector<float>& in, int dw, int w, int h) {
    std::vector<float> out(size_t(w) * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) out[size_t(y) * w + x] = in[size_t(y) * dw + x];
    return out;
}

template <typename T>
static void run_gpu(std::vector<float>& plane, int W, int H, int levels, bool forward_only) {
    const size_t n = size_t(W) * H;
    std::vector<T> host(n);
    for (size_t i = 0; i < n; i++) host[i] = T(plane[i]);

    T *d_img, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_img, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_scratch, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_img, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    mra_forward(d_img, d_scratch, W, H, levels);
    if (!forward_only) mra_inverse(d_img, d_scratch, W, H, levels);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host.data(), d_img, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_scratch));
    for (size_t i = 0; i < n; i++) plane[i] = float(host[i]);
}

static const char* arg_value(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc - 1; i++)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return nullptr;
}
static bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

int main(int argc, char** argv) {
    const char* in = arg_value(argc, argv, "--input");
    const char* out = arg_value(argc, argv, "--output");
    if (!in || !out) {
        fprintf(stderr,
                "usage: %s --input <img> --output <png> [--levels N] "
                "[--dtype fp32|fp16] [--forward]\n",
                argv[0]);
        return 2;
    }
    const char* lv = arg_value(argc, argv, "--levels");
    const char* dt = arg_value(argc, argv, "--dtype");
    int levels = lv ? std::atoi(lv) : 1;
    bool fp16 = dt && std::strcmp(dt, "fp16") == 0;
    bool forward_only = has_flag(argc, argv, "--forward");
    if (levels < 1) {
        fprintf(stderr, "error: --levels must be >= 1\n");
        return 2;
    }

    try {
        Image img = load_gray(in);
        int W = pad_to(img.w, levels), H = pad_to(img.h, levels);
        if ((std::min(W, H) >> (levels - 1)) < 8) {
            fprintf(stderr,
                    "error: --levels too high for image size after padding "
                    "(deepest processed band must be at least 8 pixels wide/high)\n");
            return 2;
        }
        std::vector<float> plane = pad_image(img, W, H);

        if (fp16)
            run_gpu<__half>(plane, W, H, levels, forward_only);
        else
            run_gpu<float>(plane, W, H, levels, forward_only);

        if (forward_only) {
            // Display transform: show the coarsest LL corner as-is (an
            // approximation image), and amplify |detail| elsewhere so the
            // near-zero sub-bands become visible. Coefficients are unchanged;
            // this only affects the saved picture.
            const int llw = W >> levels, llh = H >> levels;
            const float gain = 4.f;
            std::vector<float> viz(plane.size());
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float c = plane[size_t(y) * W + x];
                    bool ll = (x < llw && y < llh);
                    viz[size_t(y) * W + x] = ll ? c : std::fabs(c) * gain;
                }
            save_gray(out, viz, W, H);
            printf("wrote MRA representation %dx%d (levels=%d, %s) -> %s\n", W, H,
                   levels, fp16 ? "fp16" : "fp32", out);
        } else {
            std::vector<float> recon = crop_image(plane, W, img.w, img.h);
            float maxerr = 0.f;
            for (size_t i = 0; i < recon.size(); i++)
                maxerr = std::max(maxerr, std::fabs(recon[i] - img.data[i]));
            save_gray(out, recon, img.w, img.h);
            printf("round-trip %dx%d levels=%d %s: max_err=%.3e -> %s\n", img.w,
                   img.h, levels, fp16 ? "fp16" : "fp32", maxerr, out);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
