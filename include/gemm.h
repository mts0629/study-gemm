#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

float* gemm(const float* a, const size_t m, const size_t n, const float* b,
            const size_t p, float* c);

#endif  // GEMM_H