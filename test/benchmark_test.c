#include <stdio.h>
#include <time.h>

#include "gemm.h"
#include "matrix.h"
#include "timer.h"

int main(void) {
    rand_seed((unsigned)time(NULL));

    Matrix a = MAT_ZEROS(512, 1024);
    mat_rand_norm(&a);

    Matrix b = MAT_ZEROS(1024, 512);
    mat_rand_norm(&b);

    Matrix c = MAT_ZEROS(512, 512);

    struct timespec elapsed;
    MEASURE_AVG_WALL_TIME(
        elapsed,
        sgemm(GEMM_NOTRANS, GEMM_NOTRANS, a.rows, b.cols, a.cols, 1.0, a.data,
              a.cols, b.data, b.cols, 1.0, c.data, c.cols),
        10);

    // Elapsed time in ms
    printf("%lf\n", cvt2msec(elapsed));

    return 0;
}
