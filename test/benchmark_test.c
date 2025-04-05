#include <stdio.h>
#include <time.h>

#include "gemm.h"
#include "matrix.h"
#include "timer.h"

// Matrices
#define M 512
#define N 512
#define K 1024
Matrix a = {.rows = M, .cols = K, .data = (float[M * K]){0}};
Matrix ta = {.rows = K, .cols = M, .data = (float[K * M]){0}};
Matrix b = {.rows = K, .cols = N, .data = (float[K * N]){0}};
Matrix tb = {.rows = N, .cols = K, .data = (float[N * K]){0}};

int main(void) {
    rand_seed((unsigned)time(NULL));

    // Initialize matrices
    mat_rand(&a);
    mat_rand(&ta);
    mat_rand(&b);
    mat_rand(&tb);

    // Normal GEMM, AxB
    double ms = 0.0;
    {
        Matrix c = MAT_ZEROS(M, N);
        struct timespec elapsed;
        MEASURE_AVG_WALL_TIME(
            elapsed,
            sgemm(GEMM_NOTRANS, GEMM_NOTRANS, a.rows, b.cols, a.cols, 1.0,
                  a.data, a.cols, b.data, b.cols, 0.0, c.data, c.cols),
            10);

        ms = cvt2msec(elapsed);
    }

    // A^TxB
    double ms_ta = 0.0;
    {
        Matrix c = MAT_ZEROS(M, N);
        struct timespec elapsed;
        MEASURE_AVG_WALL_TIME(
            elapsed,
            sgemm(GEMM_TRANS, GEMM_NOTRANS, ta.cols, b.cols, ta.rows, 1.0,
                  ta.data, ta.cols, b.data, b.cols, 0.0, c.data, c.cols),
            10);

        ms_ta = cvt2msec(elapsed);
    }

    // AxB^T
    double ms_tb = 0.0;
    {
        Matrix c = MAT_ZEROS(M, N);
        struct timespec elapsed;
        MEASURE_AVG_WALL_TIME(
            elapsed,
            sgemm(GEMM_NOTRANS, GEMM_TRANS, a.rows, tb.rows, a.cols, 1.0,
                  a.data, a.cols, tb.data, tb.cols, 0.0, c.data, c.cols),
            10);

        ms_tb = cvt2msec(elapsed);
    }

    // A^TxB^T
    double ms_tatb = 0.0;
    {
        Matrix c = MAT_ZEROS(M, N);
        struct timespec elapsed;
        MEASURE_AVG_WALL_TIME(
            elapsed,
            sgemm(GEMM_TRANS, GEMM_TRANS, ta.cols, tb.rows, ta.rows, 1.0,
                  ta.data, ta.cols, tb.data, tb.cols, 0.0, c.data, c.cols),
            10);

        ms_tatb = cvt2msec(elapsed);
    }

    printf("%lf,%lf,%lf,%lf\n", ms, ms_ta, ms_tb, ms_tatb);

    return 0;
}
