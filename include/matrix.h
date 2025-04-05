#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Row-major 2-d matrix
typedef struct Matrix {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

// Macro for 1-d array initialization
#define ARRAY(...) \
    { __VA_ARGS__ }

// Local matrix initialized by row-major 1-d array
// Use ARRAY macro as: `MAT_FROM_ARRAY(m, k, ARRAY(...))`
#define MAT_FROM_ARRAY(rows, cols, array) \
    (Matrix) { (rows), (cols), (float[(rows) * (cols)]) array }

// Local matrix initialized by 0
#define MAT_ZEROS(rows, cols)                          \
    (Matrix) {                                         \
        (rows), (cols), (float[(rows) * (cols)]) { 0 } \
    }

// Set a seed for PRNG
void rand_seed(const unsigned int seed);

// Randomize matrix elements
void mat_rand(const Matrix* mat);

// Print a matrix
void mat_print(const Matrix* mat);

#endif  // MATRIX_H
