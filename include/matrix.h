#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

#define ARRAY(...) \
    { __VA_ARGS__ }

#define MAT_FROM_ARRAY(rows, cols, array) \
    (Matrix) { (rows), (cols), (float[(rows) * (cols)]) array }

#define MAT_ZEROS(rows, cols)                          \
    (Matrix) {                                         \
        (rows), (cols), (float[(rows) * (cols)]) { 0 } \
    }

void rand_seed(const unsigned int seed);

void mat_rand_norm(const Matrix* mat);

void mat_print(const Matrix* mat);

#endif  // MATRIX_H
