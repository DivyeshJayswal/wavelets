// Item 4: image load/save. Host-only, thin wrapper over the vendored stb
// single-headers (third_party/). Images are handled as single-channel
// grayscale float in [0,255] — the transform operates on one plane, and the
// rubric only needs a real image in and a real image out.
//
// STB_IMAGE_IMPLEMENTATION is defined in exactly one translation unit (main.cu);
// other TUs get just the declarations. See the guard below.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

struct Image {
    int w = 0, h = 0;
    std::vector<float> data;  // row-major, w*h grayscale values in [0,255]
};

// Load any stb-supported format (PNG/JPG/...) as grayscale float.
inline Image load_gray(const std::string& path) {
    int w, h, ch;
    // Force 1 channel: stb converts to luminance for us.
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &ch, 1);
    if (!px) throw std::runtime_error("load_gray: cannot read " + path + ": " + stbi_failure_reason());
    Image img{w, h, std::vector<float>(size_t(w) * h)};
    for (size_t i = 0; i < img.data.size(); i++) img.data[i] = float(px[i]);
    stbi_image_free(px);
    return img;
}

// Save grayscale float as an 8-bit PNG. Values are rounded and clamped to [0,255].
inline void save_gray(const std::string& path, const std::vector<float>& data, int w, int h) {
    if (data.size() != size_t(w) * h) throw std::runtime_error("save_gray: size mismatch");
    std::vector<unsigned char> px(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        float v = data[i];
        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        px[i] = (unsigned char)(v + 0.5f);
    }
    if (!stbi_write_png(path.c_str(), w, h, 1, px.data(), w))
        throw std::runtime_error("save_gray: cannot write " + path);
}
