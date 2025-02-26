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

#define GET_WALL_TIME(func, cvt_func, wall_time)                  \
    {                                                             \
        struct timespec start, end;                               \
        clock_gettime(CLOCK_MONOTONIC, &start);                   \
        (func);                                                   \
        clock_gettime(CLOCK_MONOTONIC, &end);                     \
        (wall_time) = (cvt_func)(get_elapsed_time(&start, &end)); \
    }

#endif
