#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

typedef enum {
    GEMM_ROW_MAJOR,
    // GEMM_COL_MAJOR
} GEMM_ORDER;

typedef enum {
    GEMM_NOTRANS,
    // GEMM_TRANS
} GEMM_TRANSPOSE;

void sgemm(GEMM_ORDER order, GEMM_TRANSPOSE trans_a, GEMM_TRANSPOSE trans_b,
           const size_t M, const size_t N, const size_t K, const float alpha,
           const float* a, const size_t lda, const float* b, const size_t ldb,
           const float beta, float* c, const size_t ldc);

#endif  // GEMM_H