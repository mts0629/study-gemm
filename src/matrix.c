#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

void rand_seed(const unsigned int seed) { srand(seed); }

void mat_rand_norm(const Matrix* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] = (float)rand() / RAND_MAX - 0.5f;
        }
    }
}

void mat_print(const Matrix* mat) {
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            printf("%.3f", mat->data[i * mat->cols + j]);
            if (j < mat->cols - 1) {
                putchar(',');
            }
        }
        putchar('\n');
    }
}
