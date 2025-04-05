#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

// Transpose flag for matrix
typedef enum GEMM_TRANSPOSE { GEMM_NOTRANS, GEMM_TRANS } GEMM_TRANSPOSE;

// Generic matrix multiplication for single-precision floating point numbers
// C = alpha * AB + beta * C
// - A: M x K, B: K x N, C: M x N
// - alpha, beta: scalar coefficients
// - transa, transb: transpose flags for matrix A, B
// - lda, ldb, ldc: leading dimensions, length of the first dimension for each
// matrix
void sgemm(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb, const size_t m,
           const size_t n, const size_t k, const float alpha, const float* a,
           const size_t lda, const float* b, const size_t ldb, const float beta,
           float* c, const size_t ldc);

#endif  // GEMM_H
