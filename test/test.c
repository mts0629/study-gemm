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
        elapsed, gemm(a.data, a.rows, a.cols, b.data, b.cols, c.data), 10
    );

    printf("Elapsed: %lf[ms]\n", cvt2msec(elapsed));

    return 0;
}
