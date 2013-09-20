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
#include <sched.h>

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
  const u8_t** data_bufs;
  u8_t **code_bufs;
  u32_t *code_block_nums;
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

  csc( cudaHostAlloc(&schedule->h_data, k*sz, cudaHostAllocDefault) );
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

  schedule->data_bufs = new const u8_t*[k];
  schedule->code_bufs = new u8_t*[m];
  schedule->code_block_nums = new u32_t[m];
  for ( int i = 0; i < k; ++i) {
    schedule->data_bufs[i] = schedule->h_data + i * sz;
  }
  for (int i = 0; i < m; ++i) {
    schedule->code_bufs[i] = schedule->h_code + i * sz;
    schedule->code_block_nums[i] = i + k;
  }

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
static bool g_skip_encoding = false;
static bool g_pin_thread = false;
static bool g_use_cpu = false;

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
  ssc(setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&WRITE_BUF_LEN,
                 (int)sizeof(WRITE_BUF_LEN)));
  ssc(connect(sock, (struct sockaddr*)&node_addr, sizeof(node_addr)));
  return sock;
}

void SendData(int sock, uint8_t* buf, size_t sz,
              uint8_t* sendbuf, size_t blk_sz) {
  if (blk_sz < sz && blk_sz > 0) {
    size_t sent = 0;
    int flag = MSG_MORE;
    size_t len = blk_sz;
    size_t remain = sz;
    do {
      if (remain <= blk_sz) {
        flag = 0;
        len = remain;
      }
      if (sendbuf != NULL) {
        memcpy(sendbuf, buf + sent, len);
        ssc0(send(sock, sendbuf, len, flag));
      } else {
        ssc0(send(sock, buf + sent, len, flag));
      }
      sent += len;
      remain -= len;
    } while (remain > 0);
  } else {
    ssc0(send(sock, buf, sz, 0));
  }
}

void* Sender(void* arg) {
  uint8_t* sendbuf = new uint8_t[WRITE_BUF_LEN];
  SenderData* data = (SenderData*)arg;

  if (g_pin_thread) {
    cpu_set_t cpumask;
    int cpu = data->id%((int)sysconf(_SC_NPROCESSORS_ONLN));
    
    CPU_ZERO(&cpumask);
    CPU_SET(cpu, &cpumask);
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpumask) != 0) {
      printf("Failed to pin sender %d to CPU %d", data->id, cpu);
    } else {
      printf("Sender %d pinned to CPU %d\n", data->id, cpu);
    }
  }
    
  size_t total_size = data->num_buffers/g_num_memory_nodes * data->buffer_size;
  if (data->id < data->num_buffers % g_num_memory_nodes) {
    total_size += data->buffer_size;
  }

  if (total_size > (1<<20)) {
    printf("Sender %d sends %lu MB per round\n", data->id, total_size>>20);
  } else {
    printf("Sender %d sends %lu KB per round\n", data->id, total_size>>10);
  }

  while (true) {
    pthread_mutex_lock(&data->mutex);
    pthread_cond_wait(&data->condition, &data->mutex);
    pthread_mutex_unlock(&data->mutex);
    // timingval tv = timing_start();
    for (int i = data->id * data->num_buffers/g_num_memory_nodes;
         i < (data->id + 1) * data->num_buffers/g_num_memory_nodes; ++i) {
      SendData(data->sock, data->buffers[i], data->buffer_size,
               sendbuf, SEND_BUF_LEN);
    }
    if (data->id < data->num_buffers % g_num_memory_nodes) {
      int i = g_num_memory_nodes
          * (data->num_buffers/g_num_memory_nodes)
          + data->id;
      SendData(data->sock, data->buffers[i], data->buffer_size,
               sendbuf, SEND_BUF_LEN);
    }
    // int64_t us = timing_stop(&tv);
    // printf("Sender %d throughput %ld MB/s\n", data->id, total_size / us);

    pthread_mutex_lock(data->done_mutex);
    --(*data->done_counter);
    if ((*data->done_counter) <= 0) {
      pthread_cond_signal(data->done_condition);
    }
    pthread_mutex_unlock(data->done_mutex);
  }
}

#define REPORT_ROUNDS 100

void RunClient(int k, int m, int block_size) {
  if (block_size * k > (1 << 20)) {
    printf("RS(%d, %d), data size: %d MB:\n", k, m,
           ((block_size * k) >> 20) );
  } else {
    printf("RS(%d, %d), data size: %d KB:\n", k, m,
           ((block_size * k) >> 10) );
  }

  if (g_pin_thread) {
    cpu_set_t cpumask;
    int cpu = g_num_memory_nodes%((int)sysconf(_SC_NPROCESSORS_ONLN));
    
    CPU_ZERO(&cpumask);
    CPU_SET(cpu, &cpumask);
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpumask) != 0) {
      printf("Failed to pin client to CPU %d", cpu);
    } else {
      printf("Client pinned to CPU %d\n", cpu);
    }
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

  SenderData* sender_data = new SenderData[g_num_memory_nodes];
  pthread_t* threads = new pthread_t[g_num_memory_nodes];

  volatile int done_counter = g_num_memory_nodes;
  pthread_mutex_t done_mutex;
  pthread_cond_t done_condition;
  pthread_mutex_init(&done_mutex, NULL);
  pthread_cond_init(&done_condition, NULL);

  for (int i = 0; i < g_num_memory_nodes; ++i) {
    pthread_mutex_init(&sender_data[i].mutex, NULL);
    pthread_cond_init(&sender_data[i].condition, NULL);
    sender_data[i].buffers = blocks;
    sender_data[i].num_buffers = k + m;
    sender_data[i].buffer_size = block_size;
    sender_data[i].id = i;
    sender_data[i].sock = ConnectToMemoryNode(MEMORY_NODES[i],
                                              CLIENT_NIC_TO_NODE[i]);
    sender_data[i].done_mutex = &done_mutex;
    sender_data[i].done_condition = &done_condition;
    sender_data[i].done_counter = &done_counter;
    pthread_create(&threads[i], NULL, Sender, (void*)&sender_data[i]);
  }

  std::cout << "Done sender threads." << std::endl;

  int num_threads = 128;
  int num_grids = block_size * m / num_threads;

  int64_t code_us = 0;
  int64_t net_us = 0;
  int64_t sync_us = 0;
  long rounds = 0;
  while (true) {
    memset(schedule->h_data, 0, block_size * k);
    
    timingval tv = timing_start();
    if (!g_skip_encoding) {
      if (g_use_cpu) {
        fec_encode(schedule->fec,
                   schedule->data_bufs,
                   schedule->code_bufs,
                   schedule->code_block_nums,
                   m,
                   block_size);
      } else {
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
      }
    }
    code_us += timing_stop(&tv);

    tv = timing_start();
    for (int i = 0; i < g_num_memory_nodes; ++i) {
      pthread_mutex_lock(&sender_data[i].mutex);
      pthread_cond_signal(&sender_data[i].condition);
      pthread_mutex_unlock(&sender_data[i].mutex);
    }
    sync_us += timing_stop(&tv);

    tv = timing_start();
    pthread_mutex_lock(&done_mutex);
    pthread_cond_wait(&done_condition, &done_mutex);
    done_counter = g_num_memory_nodes;
    pthread_mutex_unlock(&done_mutex);
    net_us += timing_stop(&tv);

    ++rounds;
    if (rounds % REPORT_ROUNDS == 0) {
      
      printf("EBW: %ld MB/s, FBW: %ld MB/s, Code: %ld%%, "
             "Sync: %ld%%, Net: %ld%%\n",
             (block_size * k * REPORT_ROUNDS) / (code_us + sync_us + net_us),
             (block_size * (k+m) * REPORT_ROUNDS) / (code_us + sync_us + net_us),
             (code_us * 100) / (code_us + sync_us + net_us),
             (sync_us * 100) / (code_us + sync_us + net_us),
             (net_us * 100) / (code_us + sync_us + net_us));
      // printf("Encoding: %ld MB/s\n", (block_size * k * REPORT_ROUNDS) / code_us );
      code_us = 0;
      sync_us = 0;
      net_us = 0;
    }
  }
}

int main(int argc, char* argv[]) {
  csc( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
  int k = 10, m = 4;
  int block_size = 1 << 20;
  
  switch (argc) {
    case 7:
      g_use_cpu = (atoi(argv[6])==0? false:true);
    case 6:
      g_skip_encoding = (atoi(argv[5])==0? false:true);
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
          " k m block_size(KB) num_memory_nodes skip_encoding use_cpu" << std::endl;
      return 0;
  }

  RunClient(k, m, block_size);
  
  return 0;
}







