cmake_minimum_required(VERSION 3.21)

project(
    gemm
    LANGUAGES C
)

set(C_STANDARD 99)
set(C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_executable(
    test
    test/test.c
    src/gemm.c
    src/matrix.c
    src/timer.c
)

target_include_directories(
    test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
