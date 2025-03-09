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

#define MEASURE_WALL_TIME(wall_time, func)        \
    {                                             \
        struct timespec s_, e_;                   \
        clock_gettime(CLOCK_MONOTONIC, &s_);      \
        (func);                                   \
        clock_gettime(CLOCK_MONOTONIC, &e_);      \
        (wall_time) = get_elapsed_time(&s_, &e_); \
    }

#define MEASURE_AVG_WALL_TIME(avg_wall_time, func, n)            \
    {                                                            \
        struct timespec s_, e_;                                  \
        clock_gettime(CLOCK_MONOTONIC, &s_);                     \
        for (int i_ = 0; i_ < (n); ++i_) {                       \
            (func);                                              \
        }                                                        \
        clock_gettime(CLOCK_MONOTONIC, &e_);                     \
        struct timespec d_ = get_elapsed_time(&s_, &e_);         \
        unsigned long ns_ = d_.tv_sec * 1000000000 + d_.tv_nsec; \
        ns_ /= (n);                                              \
        (avg_wall_time).tv_sec = ns_ / 1000000000;               \
        (avg_wall_time).tv_nsec = ns_ % 1000000000;              \
    }

#endif
