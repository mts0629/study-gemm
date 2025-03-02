#include <stdio.h>

#include "gemm.h"
#include "matrix.h"

void compare_matrix(const Matrix* a, const Matrix* b) {
    size_t size = a->cols * a->rows;

    for (size_t i = 0; i < size; ++i) {
        if (a->data[i] != b->data[i]) {
            printf("Diff. at (%ld, %ld): %f / %f\n", (i / a->cols),
                   (i % b->cols), a->data[i], b->data[i]);
        }
    }
}

void test_sgemm(void) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Matrix a = MAT_FROM_ARRAY(2, 3, ARRAY(1, 2, 3, 4, 5, 6));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8],
    //  [9, 10, 11, 12]]
    Matrix b =
        MAT_FROM_ARRAY(3, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a * b - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_NOTRANS, GEMM_NOTRANS, a.rows, b.cols, a.cols,
          2.0, a.data, a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    compare_matrix(&c, &expected);
}

void test_sgemm_trans_a(void) {
    // [[1, 4],
    //  [2, 5]
    //  [3, 6]]
    Matrix a = MAT_FROM_ARRAY(3, 2, ARRAY(1, 4, 2, 5, 3, 6));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8],
    //  [9, 10, 11, 12]]
    Matrix b =
        MAT_FROM_ARRAY(3, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a^T * b - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_TRANS, GEMM_NOTRANS, a.rows, b.cols, a.cols, 2.0,
          a.data, a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    compare_matrix(&c, &expected);
}

int main(void) {
    test_sgemm();
    test_sgemm_trans_a();

    return 0;
}
