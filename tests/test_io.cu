// Item 4 validation: grayscale PNG save/load round-trips integer pixel values.
// Host-only; this TU carries the stb implementation.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cstdio>
#include <vector>

#include "image_io.hpp"

int main() {
    const int W = 7, H = 5;  // odd dims: exercise non-power-of-two I/O
    std::vector<float> src(size_t(W) * H);
    for (int i = 0; i < W * H; i++) src[i] = float((i * 37) % 256);

    const char* path = "test_io_tmp.png";
    bool ok = true;
    try {
        save_gray(path, src, W, H);
        Image im = load_gray(path);
        ok = (im.w == W && im.h == H);
        for (int i = 0; i < W * H && ok; i++)
            if (im.data[i] != src[i]) ok = false;  // 8-bit ints survive exactly
    } catch (const std::exception& e) {
        fprintf(stderr, "io error: %s\n", e.what());
        ok = false;
    }
    std::remove(path);
    printf("test_io: PNG %dx%d round-trip  %s\n", W, H, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
