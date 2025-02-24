#include <stdio.h>

#include "gemm.h"

void print_matrix(const float* array, const size_t m, const size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.3f", array[i * m + j]);
            if (j < n - 1) {
                putchar(',');
            }
        }
        putchar('\n');
    }
}

int main(void) {
    float a[2 * 3] = {
        1, 2, 3,
        4, 5, 6
    };

    float b[3 * 2] = {
        9, 10,
        11,12,
        13, 14
    };

    float c[2 * 2];

    gemm(a, 2, 3, b, 2, c);

    print_matrix(c, 2, 2);

    return 0;
}
