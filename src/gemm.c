#include "gemm.h"

float* gemm(const float* a, const size_t m, const size_t n, const float* b,
            const size_t p, float* c) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;

            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }

            c[i * p + j] = sum;
        }
    }

    return c;
}
