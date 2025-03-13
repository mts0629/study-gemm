#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

typedef enum GEMM_TRANSPOSE { GEMM_NOTRANS, GEMM_TRANS } GEMM_TRANSPOSE;

void sgemm(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb, const size_t m,
           const size_t n, const size_t k, const float alpha, const float* a,
           const size_t lda, const float* b, const size_t ldb, const float beta,
           float* c, const size_t ldc);

#endif  // GEMM_H
