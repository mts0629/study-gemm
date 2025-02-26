#include "timer.h"

struct timespec get_elapsed_time(struct timespec* start, struct timespec* end) {
    struct timespec elapsed;
    elapsed.tv_sec = end->tv_sec - start->tv_sec;
    elapsed.tv_nsec = end->tv_nsec - start->tv_nsec;

    if (start->tv_nsec > end->tv_nsec) {
        elapsed.tv_sec -= 1;
        elapsed.tv_nsec += 1000000000;
    }

    return elapsed;
}
