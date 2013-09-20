#include <cuda.h>
#include "utils.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <net/if.h>
#include <pthread.h>
#include <assert.h>
#include <iostream>

#include "grskv.h"
#include "fec.h"

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

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  pthread_mutex_t *done_mutex;
  pthread_cond_t *done_condition;
  volatile int *done_counter;
  uint8_t **buffers;
  int num_buffers;
  size_t buffer_size;
  int id;
  int sock;
} SenderData;

static int g_num_memory_nodes = NUM_MEMORY_NODES;

int ConnectToMemoryNode(const char* address, const char* nic) {
  int sock = ssc(socket(AF_INET, SOCK_STREAM, 0));

  struct sockaddr_in node_addr;
  memset(&node_addr, 0, sizeof(node_addr));
  node_addr.sin_family = AF_INET;
  node_addr.sin_port = htons(MEMORY_NODE_PORT);
  node_addr.sin_addr.s_addr = inet_addr(address);

  char nic_name[IFNAMSIZ+1];
  strcpy(nic_name, nic);
  ssc(setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, nic_name, strlen(nic_name)));
  ssc(connect(sock, (struct sockaddr*)&node_addr, sizeof(node_addr)));
  return sock;
}

void SendData(int sock, uint8_t* buf, size_t sz, bool sendall) {
  size_t sent = 0;
  const size_t blk_size = 128*1024;
  do {
    if (sendall) {
      sent += ssc0(write(sock, buf + sent, sz - sent));
    } else {
      ssc0(write(sock, buf + sent, blk_size));
      sent += blk_size;
    }
  } while (sz > sent);
}

#define REPORT_ROUNDS 5

void RunClient(int k, int m, int block_size) {
  if (block_size * k > (1 << 20)) {
    printf("RS(%d, %d), data size: %d MB:\n", k, m,
           ((block_size * k) >> 20) );
  } else {
    printf("RS(%d, %d), data size: %d KB:\n", k, m,
           ((block_size * k) >> 10) );
  }
  
  GRSSchedule* schedule = CreateGRSSchedule(block_size, k, m);
  uint8_t** blocks = new uint8_t*[k+m];
  for (int i = 0; i < k; ++i) {
    blocks[i] = schedule->h_data + i * block_size;
  }
  for (int i = 0; i < m; ++i) {
    blocks[i + k] = schedule->h_code + i * block_size;
  }

  std::cout << "Done GRS Schedule settings." << std::endl;

  int *socks = new int[g_num_memory_nodes];
  for (int i = 0; i < g_num_memory_nodes; ++i) {
    socks[i] = ConnectToMemoryNode(MEMORY_NODES[i],
                                   CLIENT_NIC_TO_NODE[i]);
  }

  int num_threads = 256;
  int num_grids = block_size * m / num_threads;

  int64_t code_us = 0;
  int64_t net_us = 0;
  int64_t sync_us = 0;
  long rounds = 0;
  while (true) {
    // memset(schedule->h_data, 0, block_size * k);
    
    timingval tv = timing_start();
    if (rounds < 0) {
      csc( cudaMemcpyAsync(schedule->d_data, schedule->h_data, block_size*k,
                           cudaMemcpyHostToDevice, 0) );
      RSEncode<<<dim3(num_grids, 1), dim3(num_threads, 1), 0, 0>>>(
          schedule->d_ctx,
          schedule->d_data,
          block_size,
          schedule->d_code);
      csc( cudaMemcpyAsync(schedule->h_code, schedule->d_code, block_size*m,
                           cudaMemcpyDeviceToHost, 0) );
      csc( cudaStreamSynchronize(0) );
      code_us += timing_stop(&tv);
    }
    tv = timing_start();
    for (int i = 0; i < 1 /*k + m*/; ++i) {
      SendData(socks[i%g_num_memory_nodes], blocks[i], block_size, false);
    }
    net_us += timing_stop(&tv);

    ++rounds;
    if (rounds == REPORT_ROUNDS) {      
      printf("Throughput: %ld MB/s, Code: %ld%%, Sync: %ld%%, Network: %ld%%\n",
             (block_size * 1 * REPORT_ROUNDS) / (code_us + sync_us + net_us),
             (code_us * 100) / (code_us + sync_us + net_us),
             (sync_us * 100) / (code_us + sync_us + net_us),
             (net_us * 100) / (code_us + sync_us + net_us));
      // printf("Encoding: %ld MB/s\n", (block_size * k * 5) / code_us );
      code_us = 0;
      sync_us = 0;
      net_us = 0;
      rounds = 0;
    }
  }
}

int main(int argc, char* argv[]) {
  csc( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
  int k = 10, m = 4;
  int block_size = 1 << 20;
  
  switch (argc) {
    case 5:
      g_num_memory_nodes = atoi(argv[4]);
      std::cout << "Use " << g_num_memory_nodes << " of " << NUM_MEMORY_NODES
                << " memory nodes." << std::endl;
      assert(g_num_memory_nodes <= NUM_MEMORY_NODES
             && g_num_memory_nodes > 0);
    case 4:
      block_size = atoi(argv[3]) << 10;
    case 3:
      k = atoi(argv[1]);
      m = atoi(argv[2]);
      break;
    default:
      std::cout << "Usage: " << argv[0] <<
          " k m block_size(KB) num_memory_nodes" << std::endl;
      return 0;
  }

  RunClient(k, m, block_size);
  
  return 0;
}
