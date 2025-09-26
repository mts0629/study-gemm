cmake_minimum_required(VERSION 3.21)

project(
    gemm
    LANGUAGES C
)

set(CMAKE_C_COMPILER gcc)
set(C_STANDARD 99)
set(C_STANDARD_REQUIRED ON)

# Macro to switch implementation
if(IMPL STREQUAL "CHANGE_LOOP_ORDER")
    add_compile_definitions(CHANGE_LOOP_ORDER)
elseif(IMPL STREQUAL "LOOP_UNROLLING")
    add_compile_definitions(LOOP_UNROLLING)
elseif(IMPL STREQUAL "CACHE_BLOCKING")
    add_compile_definitions(CACHE_BLOCKING)
elseif(IMPL STREQUAL "OPENMP")
    add_compile_definitions(OPENMP)
else()
    # Naive implementation
    set(IMPL "NAIVE")
    add_compile_definitions(NAIVE)
endif()

add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)
if(IMPL STREQUAL "OPENMP")
    add_compile_options(-fopenmp)
    link_libraries(-lgomp)
endif()

# Benchmark test
add_executable(
    benchmark_test
    ${PROJECT_SOURCE_DIR}/test/benchmark_test.c
    ${PROJECT_SOURCE_DIR}/src/gemm.c
    ${PROJECT_SOURCE_DIR}/src/matrix.c
    ${PROJECT_SOURCE_DIR}/src/timer.c
)

target_include_directories(
    benchmark_test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Verification test
add_executable(
    verification_test
    ${PROJECT_SOURCE_DIR}/test/verification_test.c
    ${PROJECT_SOURCE_DIR}/src/gemm.c
    ${PROJECT_SOURCE_DIR}/src/matrix.c
)

target_include_directories(
    verification_test
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Get assembly output of gemm implementation
# Get properties
get_directory_property(OPTS COMPILE_OPTIONS)
get_directory_property(DEF COMPILE_DEFINITIONS)
get_target_property(INCDIR benchmark_test INCLUDE_DIRECTORIES)

add_custom_target(
    assembly
    COMMAND ${CMAKE_C_COMPILER} ${OPTS} -S -g -I${INCDIR} -D${DEF} ${PROJECT_SOURCE_DIR}/src/gemm.c -o gemm_${IMPL}.s
)
