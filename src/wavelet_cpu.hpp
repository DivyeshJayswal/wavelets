#pragma once
#include <cassert>
#include <cstdlib>
#include <vector>

constexpr float WAVELET_RECON_TOL = 1e-4f;

template <typename T>
void wavelet_forward(int size, T* s) {
    const int n = size;
    assert(size % 2 == 0 && "size is not even");
    const int n2 = n >> 1;

    std::vector<T> temp(n2);
    T* d = s + n2;
    for (int k = 0; k < n2; k++) temp[k] = s[2 * k + 1];
    for (int k = 0; k < n2; k++) s[k] = s[2 * k];
    for (int k = 0; k < n2; k++) d[k] = temp[k];

    for (int k = 0; k < n2; k++) d[k] -= s[k];

    for (int k = 0; k < n2; k++) s[k] += d[k] / 2;

    for (int k = 1; k < n2 - 1; k++) d[k] += (s[k - 1] - s[k + 1]) / 4;

    d[0] += T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
    d[n2 - 1] += T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];
}

template <typename T>
void wavelet_inverse(int size, T* s) {
    const int n = size;
    assert(size % 2 == 0 && "size is not even");
    const int n2 = n >> 1;
    T* d = s + n2;

    for (int k = 1; k < n2 - 1; k++) d[k] -= (s[k - 1] - s[k + 1]) / 4;
    d[0] -= T(0.75) * s[0] - T(1.0) * s[1] + T(0.25) * s[2];
    d[n2 - 1] -= T(0.25) * s[n2 - 1] - T(1.0) * s[n2 - 2] + T(0.75) * s[n2 - 3];

    for (int k = 0; k < n2; k++) s[k] -= d[k] / 2;

    for (int k = 0; k < n2; k++) d[k] += s[k];

    std::vector<T> temp(n2);
    for (int k = 0; k < n2; k++) temp[k] = d[k];
    for (int k = n2 - 1; k > 0; k--) s[2 * k] = s[k];
    for (int k = 0; k < n2; k++) s[2 * k + 1] = temp[k];
}
