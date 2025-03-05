#include "gemm.h"

#include <assert.h>

void sgemm(GEMM_LAYOUT layout, GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb,
           const size_t m, const size_t n, const size_t k, const float alpha,
           const float* a, const size_t lda, const float* b, const size_t ldb,
           const float beta, float* c, const size_t ldc) {
    assert(layout == GEMM_ROW_MAJOR);

    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float mac = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    mac += a[l * lda + i] * b[l * ldb + j];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * mac;
            }
        }
    } else if ((transa == GEMM_NOTRANS) && (transb == GEMM_TRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float mac = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    mac += a[i * lda + l] * b[j * ldb + l];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * mac;
            }
        }
    } else if ((transa == GEMM_TRANS) && (transb == GEMM_TRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float mac = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    mac += a[l * lda + i] * b[j * ldb + l];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * mac;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float mac = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    mac += a[i * lda + l] * b[l * ldb + j];
                }
                c[i * ldc + j] *= beta;
                c[i * ldc + j] += alpha * mac;
            }
        }
    }
}
