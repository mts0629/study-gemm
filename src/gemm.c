#include "gemm.h"

#include <assert.h>

void sgemm(GEMM_ORDER order, GEMM_TRANSPOSE trans_a, GEMM_TRANSPOSE trans_b,
           const size_t M, const size_t N, const size_t K, const float alpha,
           const float* a, const size_t lda, const float* b, const size_t ldb,
           const float beta, float* c, const size_t ldc) {
    assert(order == GEMM_ROW_MAJOR);

    if ((trans_a == GEMM_TRANS) && (trans_b == GEMM_NOTRANS)) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[k * lda + i] * b[k * ldb + j];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * sum;
            }
        }
    } else if ((trans_a == GEMM_NOTRANS) && (trans_b == GEMM_TRANS)) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * lda + k] * b[j * ldb + k];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * sum;
            }
        }
    } else if ((trans_a == GEMM_TRANS) && (trans_b == GEMM_TRANS)) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[k * lda + i] * b[j * ldb + k];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * lda + k] * b[k * ldb + j];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * sum;
            }
        }
    }
}
