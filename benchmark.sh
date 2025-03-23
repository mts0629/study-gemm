#!/bin/bash -e

# Type of GEMM implementation
impls=(
    "NAIVE"
    "CHANGE_LOOP_ORDER"
    "LOOP_UNROLLING"
    "CACHE_BLOCKING"
    "OPENMP"
)

# Measure elapsed time for each implementation type
echo "Impl.,AxB[ms],(A^T)xB[ms],Ax(B^T)[ms],(A^T)x(B^T)[ms]"
for impl in ${impls[@]}; do
    make build IMPL="${impl}" >> /dev/null

    # Break if the test result is wrong
    ./build/verification_test
    if [ $? -ne 0 ]; then
        break
    fi

    echo "${impl},$(./build/benchmark_test)"
done
