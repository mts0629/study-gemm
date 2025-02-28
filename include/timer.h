#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

struct timespec get_elapsed_time(struct timespec* start, struct timespec* end);

static inline long cvt2nsec(struct timespec time) {
    return time.tv_sec * 1000000000 + time.tv_nsec;
}

static inline double cvt2usec(struct timespec time) {
    return (double)(time.tv_sec * 1000000000 + time.tv_nsec) / 1000;
}

static inline double cvt2msec(struct timespec time) {
    return (double)(time.tv_sec * 1000000000 + time.tv_nsec) / 1000000;
}

static inline double cvt2sec(struct timespec time) {
    return (double)(time.tv_sec * 1000000000 + time.tv_nsec) / 1000000000;
}

#define MEASURE_WALL_TIME(wall_time, func)            \
    {                                                 \
        struct timespec start, end;                   \
        clock_gettime(CLOCK_MONOTONIC, &start);       \
        (func);                                       \
        clock_gettime(CLOCK_MONOTONIC, &end);         \
        (wall_time) = get_elapsed_time(&start, &end); \
    }

#define MEASURE_AVG_WALL_TIME(avg_wall_time, func, n)             \
    {                                                             \
        struct timespec start, end;                               \
        clock_gettime(CLOCK_MONOTONIC, &start);                   \
        for (int i = 0; i < (n); ++i) {                           \
            (func);                                               \
        }                                                         \
        clock_gettime(CLOCK_MONOTONIC, &end);                     \
        struct timespec elapsed = get_elapsed_time(&start, &end); \
        unsigned long total_nsec =                                \
            elapsed.tv_sec * 1000000000 + elapsed.tv_nsec;        \
        total_nsec /= (n);                                        \
        (avg_wall_time).tv_sec = total_nsec / 1000000000;         \
        (avg_wall_time).tv_nsec = total_nsec % 1000000000;        \
    }

#endif
