#include "gemm.h"

#include <assert.h>
#ifdef OPENMP
#include "omp.h"
#endif

#if defined(CHANGE_LOOP_ORDER)
// Change loop order to improve cache hit ratio in the innermost loop
static void sgemm_change_loop_order(GEMM_TRANSPOSE transa,
                                    GEMM_TRANSPOSE transb, const size_t m,
                                    const size_t n, const size_t k,
                                    const float alpha, const float* a,
                                    const size_t lda, const float* b,
                                    const size_t ldb, const float beta,
                                    float* c, const size_t ldc) {
    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }
            for (size_t l = 0; l < k; ++l) {
                for (size_t j = 0; j < n; ++j) {
                    c[i * ldc + j] += alpha * a[l * lda + i] * b[l * ldb + j];
                }
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
        // Cannot optimize by loop ordering
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
                c[i * ldc + j] *= beta;
            }
            for (size_t l = 0; l < k; ++l) {
                for (size_t j = 0; j < n; ++j) {
                    c[i * ldc + j] += alpha * a[i * lda + l] * b[l * ldb + j];
                }
            }
        }
    }
}
#elif defined(LOOP_UNROLLING)
// Unroll the innermost loop
static void sgemm_loop_unrolling(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb,
                                 const size_t m, const size_t n, const size_t k,
                                 const float alpha, const float* a,
                                 const size_t lda, const float* b,
                                 const size_t ldb, const float beta, float* c,
                                 const size_t ldc) {
    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
        for (size_t i = 0; i < m; ++i) {
            size_t j = 0;
            for (; j < n; j += 4) {
                c[i * ldc + j] *= beta;
                c[i * ldc + j + 1] *= beta;
                c[i * ldc + j + 2] *= beta;
                c[i * ldc + j + 3] *= beta;

                for (size_t l = 0; l < k; ++l) {
                    c[i * ldc + j] += alpha * a[l * lda + i] * b[l * ldb + j];
                    c[i * ldc + j + 1] +=
                        alpha * a[l * lda + i] * b[l * ldb + j + 1];
                    c[i * ldc + j + 2] +=
                        alpha * a[l * lda + i] * b[l * ldb + j + 2];
                    c[i * ldc + j + 3] +=
                        alpha * a[l * lda + i] * b[l * ldb + j + 3];
                }
            }
            for (; j < n; ++j) {
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
            size_t j = 0;
            for (; j < n; j += 4) {
                c[i * ldc + j] *= beta;
                c[i * ldc + j + 1] *= beta;
                c[i * ldc + j + 2] *= beta;
                c[i * ldc + j + 3] *= beta;

                for (size_t l = 0; l < k; ++l) {
                    c[i * ldc + j] += alpha * a[i * lda + l] * b[j * ldb + l];
                    c[i * ldc + j + 1] +=
                        alpha * a[i * lda + l] * b[(j + 1) * ldb + l];
                    c[i * ldc + j + 2] +=
                        alpha * a[i * lda + l] * b[(j + 2) * ldb + l];
                    c[i * ldc + j + 3] +=
                        alpha * a[i * lda + l] * b[(j + 3) * ldb + l];
                }
            }
            for (; j < n; ++j) {
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
            size_t j = 0;
            for (; j < n; j += 4) {
                c[i * ldc + j] *= beta;
                c[i * ldc + j + 1] *= beta;
                c[i * ldc + j + 2] *= beta;
                c[i * ldc + j + 3] *= beta;

                for (size_t l = 0; l < k; ++l) {
                    c[i * ldc + j] += alpha * a[l * lda + i] * b[j * ldb + l];
                    c[i * ldc + j + 1] +=
                        alpha * a[l * lda + i] * b[(j + 1) * ldb + l];
                    c[i * ldc + j + 2] +=
                        alpha * a[l * lda + i] * b[(j + 2) * ldb + l];
                    c[i * ldc + j + 3] +=
                        alpha * a[l * lda + i] * b[(j + 3) * ldb + l];
                }
            }
            for (; j < n; ++j) {
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
            size_t j = 0;
            for (; j < n; j += 4) {
                c[i * ldc + j] *= beta;
                c[i * ldc + j + 1] *= beta;
                c[i * ldc + j + 2] *= beta;
                c[i * ldc + j + 3] *= beta;

                for (size_t l = 0; l < k; ++l) {
                    c[i * ldc + j] += alpha * a[i * lda + l] * b[l * ldb + j];
                    c[i * ldc + j + 1] +=
                        alpha * a[i * lda + l] * b[l * ldb + j + 1];
                    c[i * ldc + j + 2] +=
                        alpha * a[i * lda + l] * b[l * ldb + j + 2];
                    c[i * ldc + j + 3] +=
                        alpha * a[i * lda + l] * b[l * ldb + j + 3];
                }
            }
            for (; j < n; ++j) {
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
#elif defined(CACHE_BLOCKING)
#define BLOCK_SIZE 4
// Cache blocking,
// split process into small blocks to improve cache hit ratio
static void sgemm_cache_blocking(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb,
                                 const size_t m, const size_t n, const size_t k,
                                 const float alpha, const float* a,
                                 const size_t lda, const float* b,
                                 const size_t ldb, const float beta, float* c,
                                 const size_t ldc) {
    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }

            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                size_t j_max = (j + BLOCK_SIZE) > n ? n : j + BLOCK_SIZE;
                for (size_t l = 0; l < k; l += BLOCK_SIZE) {
                    size_t l_max = (l + BLOCK_SIZE) > k ? k : l + BLOCK_SIZE;
                    for (size_t jj = j; jj < j_max; ++jj) {
                        for (size_t ll = l; ll < l_max; ++ll) {
                            c[i * ldc + jj] +=
                                alpha * a[ll * lda + i] * b[ll * ldb + jj];
                        }
                    }
                }
            }
        }
    } else if ((transa == GEMM_NOTRANS) && (transb == GEMM_TRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }

            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                size_t j_max = (j + BLOCK_SIZE) > n ? n : j + BLOCK_SIZE;
                for (size_t l = 0; l < k; l += BLOCK_SIZE) {
                    size_t l_max = (l + BLOCK_SIZE) > k ? k : l + BLOCK_SIZE;
                    for (size_t jj = j; jj < j_max; ++jj) {
                        for (size_t ll = l; ll < l_max; ++ll) {
                            c[i * ldc + jj] +=
                                alpha * a[i * lda + ll] * b[jj * ldb + ll];
                        }
                    }
                }
            }
        }
    } else if ((transa == GEMM_TRANS) && (transb == GEMM_TRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }

            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                size_t j_max = (j + BLOCK_SIZE) > n ? n : j + BLOCK_SIZE;
                for (size_t l = 0; l < k; l += BLOCK_SIZE) {
                    size_t l_max = (l + BLOCK_SIZE) > k ? k : l + BLOCK_SIZE;
                    for (size_t jj = j; jj < j_max; ++jj) {
                        for (size_t ll = l; ll < l_max; ++ll) {
                            c[i * ldc + jj] +=
                                alpha * a[ll * lda + i] * b[jj * ldb + ll];
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }

            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                size_t j_max = (j + BLOCK_SIZE) > n ? n : j + BLOCK_SIZE;
                for (size_t l = 0; l < k; l += BLOCK_SIZE) {
                    size_t l_max = (l + BLOCK_SIZE) > k ? k : l + BLOCK_SIZE;
                    for (size_t jj = j; jj < j_max; ++jj) {
                        for (size_t ll = l; ll < l_max; ++ll) {
                            c[i * ldc + jj] +=
                                alpha * a[i * lda + ll] * b[ll * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}
#else
// Naive implementation
static void sgemm_naive(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb,
                        const size_t m, const size_t n, const size_t k,
                        const float alpha, const float* a, const size_t lda,
                        const float* b, const size_t ldb, const float beta,
                        float* c, const size_t ldc) {
    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
#ifdef OPENMP
#pragma omp parallel for
#endif
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
#ifdef OPENMP
#pragma omp parallel for
#endif
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
#ifdef OPENMP
#pragma omp parallel for
#endif
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
#ifdef OPENMP
#pragma omp parallel for
#endif
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
#endif

void sgemm(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb, const size_t m,
           const size_t n, const size_t k, const float alpha, const float* a,
           const size_t lda, const float* b, const size_t ldb, const float beta,
           float* c, const size_t ldc) {
#if defined(CHANGE_LOOP_ORDER)
    sgemm_change_loop_order(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc);
#elif defined(LOOP_UNROLLING)
    sgemm_loop_unrolling(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc);
#elif defined(CACHE_BLOCKING)
    sgemm_cache_blocking(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc);
#else
    sgemm_naive(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}
