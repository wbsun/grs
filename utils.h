#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#define interval_us(oldv, newv)					\
    ((int64_t)(1000000*((newv).tv_sec - (oldv).tv_sec)) +	\
     ((int64_t)((newv).tv_usec) - (int64_t)((oldv).tv_usec)))

typedef struct {
    struct timeval oldt;
    struct timeval newt;
} timingval;

inline static timingval timing_start() {
    timingval tv;
    gettimeofday(&tv.oldt, 0);
    return tv;
}

inline static int64_t timing_stop(timingval *tv) {
    gettimeofday(&tv->newt, 0);
    return interval_us(tv->oldt, tv->newt);
}

inline static int64_t timing_elapsed(timingval *tv) {
    return interval_us(tv->oldt, tv->newt);
}

int _safe_syscall(int r, const char *file, int line, int err_upbound)
{
    if (r<err_upbound) {
	fprintf(stderr, "Error in %s:%d, ", file, line);
	perror("");
	abort();
    }
    return r;
}

#define ssc(...) _safe_syscall(__VA_ARGS__, __FILE__, __LINE__, 0)
#define ssc0(...) _safe_syscall(__VA_ARGS__, __FILE__, __LINE__, 1)

#ifndef NO_CUDA
#include <cuda.h>

#define csc(...) _cuda_safe_call(__VA_ARGS__, __FILE__, __LINE__)
static cudaError_t _cuda_safe_call(cudaError_t e, const char *file, int line) {
    if (e!=cudaSuccess) {
	fprintf(stderr, "CUDA Error: %s %d %s\n",
		file, line, cudaGetErrorString(e));
	cudaDeviceReset();
	abort();
    }
    return e;
}

#endif  // NO_CUDA

#endif
