#define NO_CUDA
#include "utils.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "grskv.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " ip_address" << std::endl;
    return 0;
  }
  
  struct sockaddr_in my_addr;

  int sock = ssc(socket(AF_INET, SOCK_STREAM, 0));
  int optval = 1;
  ssc(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)));
  ssc(setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&READ_BUF_LEN,
                 (int)sizeof(READ_BUF_LEN)));
  memset(&my_addr, 0, sizeof(my_addr));
  my_addr.sin_family = AF_INET;
  my_addr.sin_addr.s_addr = inet_addr(argv[1]);
  my_addr.sin_port = htons(MEMORY_NODE_PORT);  
  ssc(bind(sock, (struct sockaddr*)&my_addr, sizeof(my_addr)));
  ssc(listen(sock, 5));

  uint8_t* buffer = new uint8_t[READ_BUF_LEN];
  if (buffer == NULL) {
    std::cerr << "Out of memory\n";
    abort();
  }
  
  std::cout << "Accepting connection..." << std::endl;  
  int comm_sock = ssc(accept(sock, (struct sockaddr *)NULL, NULL));
  ssc(setsockopt(comm_sock, SOL_SOCKET, SO_RCVBUF, (char*)&READ_BUF_LEN,
                 (int)sizeof(READ_BUF_LEN)));
  std::cout << "Storage client connected to me." << std::endl;

  ssize_t size = 0;
  ssize_t next_milestone = 1<<25;
  int64_t us = 0;
  while (true) {
    timingval tv = timing_start();
    ssize_t rt = read(comm_sock, buffer, READ_BUF_LEN);
    us += timing_stop(&tv);
    if (rt < 0) {
      std::cerr << "Error in " << __FILE__ << ":" << __LINE__ << ", ";
      perror("");
      abort();
    } else if (rt == 0) {
      std::cout << "Done client. Received " << (size >> 20) << "MB data in total."
                << std::endl;
      break;
    } else {
      size += rt;
      if (size >= next_milestone) {
        std::cout << "Got " << (size >> 20) << "MB data, throughput " << (size / us)
                  << " MB/s." << std::endl;
        next_milestone = (size + (1<<26));
      }
    }
  }
  delete[] buffer;
  return 0;
}
