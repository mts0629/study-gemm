#include "gemm.h"

#include <assert.h>

void sgemm(GEMM_TRANSPOSE transa, GEMM_TRANSPOSE transb, const size_t m,
           const size_t n, const size_t k, const float alpha, const float* a,
           const size_t lda, const float* b, const size_t ldb, const float beta,
           float* c, const size_t ldc) {
#if defined(CACHE_BLOCKING)
    if ((transa == GEMM_TRANS) && (transb == GEMM_NOTRANS)) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] *= beta;
            }

            for (size_t j = 0; j < n; j += 4) {
                size_t j_max = (j + 4) > n ? n : j + 4;
                for (size_t l = 0; l < k; l += 4) {
                    size_t l_max = (l + 4) > k ? k : l + 4;
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

            for (size_t j = 0; j < n; j += 4) {
                size_t j_max = (j + 4) > n ? n : j + 4;
                for (size_t l = 0; l < k; l += 4) {
                    size_t l_max = (l + 4) > k ? k : l + 4;
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

            for (size_t j = 0; j < n; j += 4) {
                size_t j_max = (j + 4) > n ? n : j + 4;
                for (size_t l = 0; l < k; l += 4) {
                    size_t l_max = (l + 4) > k ? k : l + 4;
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

            for (size_t j = 0; j < n; j += 4) {
                size_t j_max = (j + 4) > n ? n : j + 4;
                for (size_t l = 0; l < k; l += 4) {
                    size_t l_max = (l + 4) > k ? k : l + 4;
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
#elif defined(LOOP_UNROLLING)
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
#elif defined(CHANGE_LOOP_ORDER)
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
#else
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
#endif
}
