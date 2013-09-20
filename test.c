#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <inttypes.h>
#include "fec.h"
#define NO_CUDA
#include "utils.h"

void GenerateRandomData(u8_t* buffer, size_t sz) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  srand(tv.tv_usec);

  size_t i;
  for (i = 0; i < sz; ++i) {
    buffer[i] = rand() && 0xff;
  }  
}

#define DefineSchedule(k, m)                    \
  struct {                                      \
    fec_t* ctx;                                 \
    u8_t* data[(k)];                            \
    u8_t* code[(m)];                            \
    u32_t code_block_nums[(m)];                 \
    size_t block_size;                          \
  } schedule

#define CreateSchedule(schedule, block_size, k, m)      \
  do {                                                  \
    schedule.ctx = fec_new((k), (k)+(m));               \
    assert(schedule.ctx != NULL);                       \
                                                        \
    int i;                                              \
    for (i = 0; i < (k); ++i)                           \
    {                                                   \
      schedule.data[i] = (u8_t*)malloc(block_size);             \
      assert(schedule.data[i] != NULL);                         \
      GenerateRandomData(schedule.data[i], block_size);         \
    }                                                           \
                                                                \
    for (i = 0; i < (m); ++i) {                                 \
      schedule.code[i] = (u8_t*)malloc(block_size);             \
      assert(schedule.code[i] != NULL);                         \
      schedule.code_block_nums[i] = i + (k);                    \
    }                                                           \
                                                                \
    schedule.block_size = block_size;                           \
  } while (0)
  
#define DestroySchedule(schedule, k, m) \
  do {                                  \
    fec_free(schedule.ctx);             \
                                        \
    int i;                              \
    for (i = 0; i < (k); ++i) {         \
      free(schedule.data[i]);           \
    }                                   \
                                        \
    for (i = 0; i < (m); ++i) {         \
      free(schedule.code[i]);           \
    }                                   \
  } while (0)


void EvaluateRS(size_t block_size, int k, int m) {
  if (block_size * k > (1 << 20)) {
    printf("RS(%d, %d), data size: %lu MB:\n", k, m,
           ((block_size * k) >> 20) );
  } else {
    printf("RS(%d, %d), data size: %lu KB:\n", k, m,
           ((block_size * k) >> 10) );
  }
  
  DefineSchedule(k, m);
  CreateSchedule(schedule, block_size, k, m);

  timingval tv = timing_start();
  fec_encode(schedule.ctx,
             (const u8_t**)schedule.data,
             schedule.code,
             schedule.code_block_nums,
             m,
             schedule.block_size);
  int64_t us = timing_stop(&tv);
  printf("Encode Time %" PRId64" us, Throughput: %" PRIu64 " MB/s\n",
         us, block_size * k / us);

  u32_t* indices = (u32_t*)malloc(sizeof(u32_t) * k);
  u8_t** left_blocks = (u8_t**)malloc(sizeof(u8_t*) * k);
  assert(indices != NULL && left_blocks != NULL);
  int i;
  for (i = 0; i < k - m; ++i) {
    left_blocks[i] = schedule.data[i];
    indices[i] = i;
  }
  for (i = 0; i < m; ++i) {
    left_blocks[i + k - m] = schedule.code[i];
    indices[i + k - m] = i + k;
  }
  
  u8_t** recovered_blocks = (u8_t**)malloc(sizeof(u8_t*) * m);
  assert(recovered_blocks != NULL);
  for (i = 0; i < m; ++i) {
    recovered_blocks[i] = (u8_t*)malloc(schedule.block_size);
    assert(recovered_blocks[i] != NULL);
  }
  
  tv = timing_start();
  fec_decode(schedule.ctx,
             (const u8_t**)left_blocks,
             recovered_blocks,
             (const u32_t*)indices,
             schedule.block_size);
  us = timing_stop(&tv);
  printf("Decode Time %" PRId64 " us, Throughput: %" PRIu64 " MB/s\n",
         us, block_size * k / us);

  for (i = 0; i < m; ++i) {
    if (memcmp(recovered_blocks[i],
               schedule.data[i + k - m],
               schedule.block_size) != 0) {
      printf("Block %d not recovered correctly\n", i + k - m);
    }
  }

  printf("\n");
  free(indices);
  free(left_blocks);
  free(recovered_blocks);
  DestroySchedule(schedule, k, m);
}

int main(int argc, char* argv[]) {
  size_t block_size = 1 << 20;

  if (argc > 1) {
    block_size = (size_t)atol(argv[1]);
  }

  EvaluateRS(block_size, 10, 4);
  EvaluateRS(block_size, 6, 3);
  EvaluateRS(block_size, 8, 4);
  EvaluateRS(block_size, 10, 5);
  EvaluateRS(block_size, 9, 3);
  EvaluateRS(block_size, 4, 2);
  EvaluateRS(block_size, 8, 3);
  EvaluateRS(block_size, 6, 4);
  EvaluateRS(block_size, 9, 6);

  return 0;
}
