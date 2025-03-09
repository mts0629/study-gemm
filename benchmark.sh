#!/bin/bash -e

# Type of GEMM implementation
impls=(
    "NAIVE"
    "CHANGE_LOOP_ORDER"
)

# Measure elapsed time for each implementation type
for impl in ${impls[@]}; do
    make build IMPL="${impl}" >> /dev/null

    # Break if the test result is wrong
    ./build/verification_test
    if [ $? -ne 0 ]; then
        break
    fi

    echo "${impl},$(./build/benchmark_test)"
done
