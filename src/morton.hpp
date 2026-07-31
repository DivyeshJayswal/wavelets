// Item 5: Z-order (Morton) indexing, used by the data-layout discussion.
//
// Row-major storage makes the MRA column pass stride by W; a Morton (Z-order)
// layout interleaves x/y bits so that a 2D neighbourhood is contiguous in
// memory, which *should* help the strided column access and deep-level locality.
// bench/bench.cu measures whether it actually does. These helpers are the layout
// primitive; the report weighs the trade-off (index arithmetic cost vs locality).
#pragma once
#include <cstdint>

// Spread the low 16 bits of v into even bit positions (v_i -> bit 2i).
__host__ __device__ inline uint32_t part1by1(uint32_t v) {
    v &= 0x0000ffffu;
    v = (v | (v << 8)) & 0x00ff00ffu;
    v = (v | (v << 4)) & 0x0f0f0f0fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}
// Inverse of part1by1: gather even bits back into the low 16.
__host__ __device__ inline uint32_t compact1by1(uint32_t v) {
    v &= 0x55555555u;
    v = (v | (v >> 1)) & 0x33333333u;
    v = (v | (v >> 2)) & 0x0f0f0f0fu;
    v = (v | (v >> 4)) & 0x00ff00ffu;
    v = (v | (v >> 8)) & 0x0000ffffu;
    return v;
}

// Interleave (x,y) in [0,65536) into a 32-bit Morton code, and back.
__host__ __device__ inline uint32_t morton_encode(uint32_t x, uint32_t y) {
    return part1by1(x) | (part1by1(y) << 1);
}
__host__ __device__ inline void morton_decode(uint32_t code, uint32_t& x, uint32_t& y) {
    x = compact1by1(code);
    y = compact1by1(code >> 1);
}
