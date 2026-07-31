// CPU reference wavelet transform — the correctness oracle for every GPU kernel.
//
// This is the lifting scheme from the provided baseline (wavelets/1D_wavelets.cpp),
// refactored into a reusable header. The numerics are IDENTICAL to the baseline:
// a floating-point CDF-5/3-style lifting transform. Do not change the math here —
// GPU kernels are validated against it.
//
// Layout after forward: [ approx (n/2) | detail (n/2) ].
// Reconstruction is exact only up to floating-point epsilon (see WAVELET_RECON_TOL).
#pragma once
#include <cassert>
#include <cstdlib>
#include <vector>

// Tolerance for "perfect reconstruction" checks in fp32. The transform is
// floating point (not integer lifting), so round-trip error is nonzero.
constexpr float WAVELET_RECON_TOL = 1e-4f;

// Forward (analysis): base signal -> wavelet coefficients, in place.
// `s` has `size` elements; `size` must be even.
template <typename T>
void wavelet_forward(int size, T* s) {
    const int n = size;
    assert(size % 2 == 0 && "size is not even");
    const int n2 = n >> 1;

    // Split even and odd samples into [ even | odd ].
    std::vector<T> temp(n2);
    T* d = s + n2;
    for (int k = 0; k < n2; k++) temp[k] = s[2 * k + 1];
    for (int k = 0; k < n2; k++) s[k] = s[2 * k];
    for (int k = 0; k < n2; k++) d[k] = temp[k];

    // Step 1: details -= samples.
    for (int k = 0; k < n2; k++) d[k] -= s[k];
    // Step 2: samples += details/2.
    for (int k = 0; k < n2; k++) s[k] += d[k] / 2;
    // Step 3: lifting of details.
    for (int k = 1; k < n2 - 1; k++) d[k] += (s[k - 1] - s[k + 1]) / 4;
    // Edge cases (asymmetric boundary handling).
    d[0] += T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
    d[n2 - 1] += T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];
}

// Inverse (synthesis): wavelet coefficients -> base signal, in place.
template <typename T>
void wavelet_inverse(int size, T* s) {
    const int n = size;
    assert(size % 2 == 0 && "size is not even");
    const int n2 = n >> 1;
    T* d = s + n2;

    // Step 3 undone: unlift details.
    for (int k = 1; k < n2 - 1; k++) d[k] -= (s[k - 1] - s[k + 1]) / 4;
    d[0] -= T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
    d[n2 - 1] -= T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];

    // Step 2 undone: samples -= details/2.
    for (int k = 0; k < n2; k++) s[k] -= d[k] / 2;
    // Step 1 undone: details += samples.
    for (int k = 0; k < n2; k++) d[k] += s[k];

    // Unsplit even/odd back to interleaved layout.
    std::vector<T> temp(n2);
    for (int k = 0; k < n2; k++) temp[k] = d[k];
    for (int k = n2 - 1; k > 0; k--) s[2 * k] = s[k];
    for (int k = 0; k < n2; k++) s[2 * k + 1] = temp[k];
}
