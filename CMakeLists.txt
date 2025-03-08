cmake_minimum_required(VERSION 3.21)

project(
    gemm
    LANGUAGES C
)

set(C_STANDARD 99)
set(C_STANDARD_REQUIRED ON)

add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)

add_executable(
    benchmark_test
    test/benchmark_test.c
    src/gemm.c
    src/matrix.c
    src/timer.c
)

target_include_directories(
    benchmark_test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(
    verification_test
    test/verification_test.c
    src/gemm.c
    src/matrix.c
)

target_include_directories(
    verification_test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
