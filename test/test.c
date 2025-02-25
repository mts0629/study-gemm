#include <stdio.h>
#include <time.h>

#include "gemm.h"
#include "matrix.h"

int main(void) {
    rand_seed((unsigned)time(NULL));

    Matrix a = MAT_ZEROS(2, 3);
    mat_rand_norm(&a);

    Matrix b = MAT_ZEROS(3, 2);
    mat_rand_norm(&b);

    Matrix c = MAT_ZEROS(2, 2);
    gemm(a.data, a.rows, a.cols, b.data, b.cols, c.data);

    mat_print(&c);

    return 0;
}
