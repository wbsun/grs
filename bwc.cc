#define NO_CUDA
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
#include <inttypes.h>

#include "grskv.h"

int ConnectToMemoryNode(const char* address, const char* nic) {
  int sock = ssc(socket(AF_INET, SOCK_STREAM, 0));

  struct sockaddr_in node_addr;
  memset(&node_addr, 0, sizeof(node_addr));
  node_addr.sin_family = AF_INET;
  node_addr.sin_port = htons(MEMORY_NODE_PORT);
  node_addr.sin_addr.s_addr = inet_addr(address);

  char nic_name[IFNAMSIZ+1];
  strcpy(nic_name, nic);
  ssc(setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, nic_name,
                 strlen(nic_name)));
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

static int BUF_SIZE = 1 << 20;
static const int REPORT_ROUNDS = 10;

int main(int argc, char* argv[]) {
  int nic = atoi(argv[1]);
  BUF_SIZE = atoi(argv[2]) << 10;
  uint8_t *buffer = new uint8_t[BUF_SIZE * REPORT_ROUNDS];
  uint8_t *sendbuf = new uint8_t[WRITE_BUF_LEN];
  int rounds = 0;
  int sock = ConnectToMemoryNode(MEMORY_NODES[nic], CLIENT_NIC_TO_NODE[nic]);
  int64_t us = 0;
  while (true) {
    timingval tv = timing_start();
    SendData(sock, buffer + rounds*BUF_SIZE, BUF_SIZE, sendbuf, WRITE_BUF_LEN);
    us += timing_stop(&tv);
    if (++rounds == REPORT_ROUNDS) {
      printf("Throughput: %ld MB/s, time %ld us\n", (BUF_SIZE*REPORT_ROUNDS)/us, us);
      rounds = 0;
      us = 0;
    }
  }
  return 0;
}
