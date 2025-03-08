#include <stdio.h>

#include "gemm.h"
#include "matrix.h"

// Compare two matrices and return the number of different elements
static size_t compare_matrix(const char* func_name, const Matrix* a,
                             const Matrix* b) {
    size_t n_diffs = 0;

    size_t size = a->cols * a->rows;
    for (size_t i = 0; i < size; ++i) {
        if (a->data[i] != b->data[i]) {
            printf("%s: diff. at (%ld, %ld): %f / %f\n", func_name,
                   (i / a->cols), (i % b->cols), a->data[i], b->data[i]);
            n_diffs++;
        }
    }

    return n_diffs;
}

static size_t test_sgemm(void) {
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

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_trans_a(void) {
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
    sgemm(GEMM_ROW_MAJOR, GEMM_TRANS, GEMM_NOTRANS, a.cols, b.cols, a.rows, 2.0,
          a.data, a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_trans_b(void) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Matrix a = MAT_FROM_ARRAY(2, 3, ARRAY(1, 2, 3, 4, 5, 6));

    // [[1, 5, 9],
    //  [2, 6, 10],
    //  [3, 7, 11],
    //  [4, 8, 12]]
    Matrix b =
        MAT_FROM_ARRAY(4, 3, ARRAY(1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a * b^T - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_NOTRANS, GEMM_TRANS, a.rows, b.rows, a.cols, 2.0,
          a.data, a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_trans_ab(void) {
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    Matrix a = MAT_FROM_ARRAY(3, 2, ARRAY(1, 4, 2, 5, 3, 6));

    // [[1, 5, 9],
    //  [2, 6, 10],
    //  [3, 7, 11],
    //  [4, 8, 12]]
    Matrix b =
        MAT_FROM_ARRAY(4, 3, ARRAY(1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a^T * b^T - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_TRANS, GEMM_TRANS, a.cols, b.rows, a.rows, 2.0,
          a.data, a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_lda(void) {
    // [[0, 1, 2, 3, 4],
    //  [3, 4, 5, 6, 7]]
    Matrix a = MAT_FROM_ARRAY(2, 5, ARRAY(0, 1, 2, 3, 4, 3, 4, 5, 6, 7));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8],
    //  [9, 10, 11, 12]]
    Matrix b =
        MAT_FROM_ARRAY(3, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a[:, 1:4] * b - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_NOTRANS, GEMM_NOTRANS, a.rows, b.cols, 3, 2.0,
          (a.data + 1), a.cols, b.data, b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_ldb(void) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Matrix a = MAT_FROM_ARRAY(2, 3, ARRAY(1, 2, 3, 4, 5, 6));

    // [[0, 1, 2, 3, 4, 5],
    //  [4, 5, 6, 7, 8, 9],
    //  [8, 9, 10, 11, 12, 13]]
    Matrix b = MAT_FROM_ARRAY(
        3, 6, ARRAY(0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    Matrix c = MAT_FROM_ARRAY(2, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8));

    // c = 2.0 * a * b[:, 1:5] - 3.0 * c
    sgemm(GEMM_ROW_MAJOR, GEMM_NOTRANS, GEMM_NOTRANS, a.rows, 4, a.cols, 2.0,
          a.data, a.cols, (b.data + 1), b.cols, -3.0, c.data, c.cols);

    // [[73, 82, 91, 100],
    //  [151, 178, 205, 232]]
    Matrix expected =
        MAT_FROM_ARRAY(2, 4, ARRAY(73, 82, 91, 100, 151, 178, 205, 232));

    return compare_matrix(__func__, &c, &expected);
}

static size_t test_sgemm_ldc(void) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Matrix a = MAT_FROM_ARRAY(2, 3, ARRAY(1, 2, 3, 4, 5, 6));

    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8],
    //  [9, 10, 11, 12]]
    Matrix b =
        MAT_FROM_ARRAY(3, 4, ARRAY(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    // [[0, 1, 2, 3, 4, 5],
    //  [4, 5, 6, 7, 8, 9]]
    Matrix c = MAT_FROM_ARRAY(2, 6, ARRAY(0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9));

    // c = 2.0 * a * b - 3.0 * c[:, 1:5]
    sgemm(GEMM_ROW_MAJOR, GEMM_NOTRANS, GEMM_NOTRANS, a.rows, b.cols, a.cols,
          2.0, a.data, a.cols, b.data, b.cols, -3.0, (c.data + 1), c.cols);

    // [[0, 73, 82, 91, 100, 5],
    //  [4, 151, 178, 205, 232, 9]]
    Matrix expected = MAT_FROM_ARRAY(
        2, 6, ARRAY(0, 73, 82, 91, 100, 5, 4, 151, 178, 205, 232, 9));

    return compare_matrix(__func__, &c, &expected);
}

int main(void) {
    size_t n_diffs = 0;
    n_diffs += test_sgemm();
    n_diffs += test_sgemm_trans_a();
    n_diffs += test_sgemm_trans_b();
    n_diffs += test_sgemm_trans_ab();
    n_diffs += test_sgemm_lda();
    n_diffs += test_sgemm_ldb();
    n_diffs += test_sgemm_ldc();

    if (n_diffs > 0) {
        // Test failed
        return 1;
    }

    return 0;
}
