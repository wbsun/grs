#include <cuda.h>
#include "fec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "utils.h"

static int g_num_threads = 128;

__constant__ u8_t d_gf_mul_table[256][256];

#define d_gf_mul(x, y) d_gf_mul_table[x][y];

#define RS_MAX_K 10

typedef struct {
  u32_t magic;
  u16_t k, n, m;   // k: data blocks; n: total blocks; m: check blocks.
  u8_t* enc_matrix;
  u8_t* dec_matrices[1024];  
} RSContext;

typedef struct {                                      
  u8_t* h_data; 
  u8_t* d_data;
  u8_t* h_code;
  u8_t* d_code;
  RSContext* h_ctx;
  RSContext* d_ctx;
  u8_t* h_enc_mat; 
  u8_t* d_enc_mat; 
  size_t block_size; 
  fec_t* fec;
} GRSSchedule;

__global__ void RSEncode(RSContext* ctx, u8_t* data, size_t sz, u8_t* code) {
  int myid = threadIdx.x + blockIdx.x * blockDim.x;
  int row = myid / sz;
  int mywork = myid % sz;
  u8_t* p = ctx->enc_matrix + (row + ctx->k) * ctx->k;
  u8_t* d = data + mywork;
  
  u8_t parity = 0;
  for (int j = 0; j < ctx->k; d+=sz, ++j) {
    parity ^= d_gf_mul(p[j], *d);
  }
  code[sz * row + mywork] = parity;
}

__global__ void RSDecode(RSContext* ctx, u8_t* good_blocks, size_t sz,
                         u8_t* recovered_blocks, const u32_t* index,
                         u32_t index_bitmap) {
}

#include "fec.c"

void InitGRS() {
  if (fec_initialized == 0)
    init_fec();
  
  csc( cudaMemcpyToSymbol(d_gf_mul_table, gf_mul_table, 256*256) );
}

void GenerateRandomData(u8_t* buffer, size_t sz) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  srand(tv.tv_usec);

  size_t i;
  for (i = 0; i < sz; ++i) {
    buffer[i] = rand() && 0xff;
  }  
}

GRSSchedule* CreateGRSSchedule(size_t sz, int k, int m) {
  GRSSchedule* schedule = (GRSSchedule*)malloc(sizeof(GRSSchedule));
  schedule->fec = fec_new(k, k+m);

  csc( cudaHostAlloc(&schedule->h_data, k*sz, cudaHostAllocWriteCombined) );
  csc( cudaMalloc(&schedule->d_data, k*sz) );
  csc( cudaMallocHost(&schedule->h_code, m*sz) );
  csc( cudaMalloc(&schedule->d_code, m*sz) );
  csc( cudaMallocHost(&schedule->h_ctx, sizeof(RSContext)) );
  csc( cudaMalloc(&schedule->d_ctx, sizeof(RSContext)) );
  csc( cudaMallocHost(&schedule->h_enc_mat, k*(k+m)) );
  csc( cudaMalloc(&schedule->d_enc_mat, k*(k+m)) );

  // GenerateRandomData(schedule->h_data, k*sz);
  
  schedule->h_ctx->magic = schedule->fec->magic;
  schedule->h_ctx->k = k; 
  schedule->h_ctx->n = k+m;
  schedule->h_ctx->m = m;
  schedule->h_ctx->enc_matrix = schedule->d_enc_mat;
  csc( cudaMemcpy(schedule->d_ctx, schedule->h_ctx, sizeof(RSContext), cudaMemcpyHostToDevice) );
  csc( cudaDeviceSynchronize() );
  
  schedule->block_size = sz;
  memcpy(schedule->h_enc_mat, schedule->fec->enc_matrix, k*(k+m));
  csc( cudaMemcpy(schedule->d_enc_mat, schedule->h_enc_mat, k*(k+m), cudaMemcpyHostToDevice) );
  csc( cudaDeviceSynchronize() );

  return schedule;
}

void DestroyGRSSchedule(GRSSchedule* schedule) {
  csc( cudaFreeHost(schedule->h_data) );
  csc( cudaFreeHost(schedule->h_code) );
  csc( cudaFreeHost(schedule->h_ctx) );
  csc( cudaFreeHost(schedule->h_enc_mat) );
  csc( cudaFree(schedule->d_data) );
  csc( cudaFree(schedule->d_code) );
  csc( cudaFree(schedule->d_ctx) );
  csc( cudaFree(schedule->d_enc_mat) );
  fec_free(schedule->fec);
  free(schedule);
}

void EvaluateGRS(size_t block_size, int k, int m, int nstream, bool device_sync) {
  if (block_size * k > (1 << 20)) {
    printf("RS(%d, %d), %d streams, data size: %lu MB:\n", k, m, nstream,
           ((block_size * k) >> 20) );
  } else {
    printf("RS(%d, %d), %d streams, data size: %lu KB:\n", k, m, nstream,
           ((block_size * k) >> 10) );
  }
  
  GRSSchedule **schedules = new GRSSchedule*[nstream];
  cudaStream_t *streams = new cudaStream_t[nstream];
  for (int i = 0; i < nstream; ++i) {
      csc( cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking) );
    schedules[i] = CreateGRSSchedule(block_size, k, m);
  }

  int num_threads = g_num_threads;
  int num_grids = block_size * m / num_threads;
  printf("Start encoding...\n");

  timingval tv = timing_start();
  for (int i = 0; i < nstream; ++i) {
      csc( cudaMemcpyAsync(schedules[i]->d_data, schedules[i]->h_data, block_size*k,
			   cudaMemcpyHostToDevice, streams[i]) );
    RSEncode<<<dim3(num_grids, 1), dim3(num_threads, 1), 0, streams[i]>>>(
        schedules[i]->d_ctx,
        schedules[i]->d_data,
        block_size,
        schedules[i]->d_code);
    csc( cudaMemcpyAsync(schedules[i]->h_code, schedules[i]->d_code, block_size*m,
                         cudaMemcpyDeviceToHost, streams[i]) );
  }
  if (device_sync) {
      csc( cudaDeviceSynchronize() );
  } else {
      for (int i = 0; i < nstream; ++i) {
	  csc( cudaStreamSynchronize(streams[i]) );
      }
  }
  int64_t us = timing_stop(&tv);
  printf("Encode Time %lld us, Throughput: %llu MB/s\n\n",
         us/nstream, block_size * k * nstream / us);

  for (int i = 0; i < nstream; ++i) {
    csc( cudaStreamDestroy(streams[i]) );
    DestroyGRSSchedule(schedules[i]);
  }
  free(streams);
  free(schedules);
}

int main(int argc, char* argv[]) {
    csc( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
  size_t block_size = 1 << 20;
  int nstreams = 4;
  bool device_sync = false;

  switch(argc) {
    case 5:
      g_num_threads = atoi(argv[4]);
    case 4:
      device_sync = (atoi(argv[3]) == 1? true:false);
    case 3:
      nstreams = atoi(argv[2]);
    case 2:
      block_size = (size_t)atol(argv[1])<<10;
    default:
      break;
  }

  EvaluateGRS(block_size, 10, 4, nstreams, device_sync);
  EvaluateGRS(block_size, 6, 3,  nstreams, device_sync);
  EvaluateGRS(block_size, 8, 4,  nstreams, device_sync);
  EvaluateGRS(block_size, 10, 5, nstreams, device_sync);
  EvaluateGRS(block_size, 9, 3,  nstreams, device_sync);
  EvaluateGRS(block_size, 4, 2,  nstreams, device_sync);
  EvaluateGRS(block_size, 8, 3,  nstreams, device_sync);
  EvaluateGRS(block_size, 6, 4,  nstreams, device_sync);
  EvaluateGRS(block_size, 9, 6,  nstreams, device_sync);

  return 0;
}
