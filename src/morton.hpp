#pragma once
#include <cstdint>

__host__ __device__ inline uint32_t part1by1(uint32_t v) {
    v &= 0x0000ffffu;
    v = (v | (v << 8)) & 0x00ff00ffu;
    v = (v | (v << 4)) & 0x0f0f0f0fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__host__ __device__ inline uint32_t compact1by1(uint32_t v) {
    v &= 0x55555555u;
    v = (v | (v >> 1)) & 0x33333333u;
    v = (v | (v >> 2)) & 0x0f0f0f0fu;
    v = (v | (v >> 4)) & 0x00ff00ffu;
    v = (v | (v >> 8)) & 0x0000ffffu;
    return v;
}

__host__ __device__ inline uint32_t morton_encode(uint32_t x, uint32_t y) {
    return part1by1(x) | (part1by1(y) << 1);
}
__host__ __device__ inline void morton_decode(uint32_t code, uint32_t& x, uint32_t& y) {
    x = compact1by1(code);
    y = compact1by1(code >> 1);
}
