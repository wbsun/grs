#ifndef __SENDER_H__
#define __SENDER_H__

#include <vector>
#include <pthread.h>
#include "data_block_queue.h"

class SenderWorker;

class Sender {
public:
  typedef std::vector<std::pair<std::string, string::string> > AddressNICVector;
  Sender(AddressNICVector& memory_nodes);
  virtual ~Sender();

  bool Initialize();
  void Stop();

 private:
  AddressNICVector address_nics_;
  std::vector<SenderWorker> workers_;
};

class SenderWorker {
 public:
  SenderWorker(int id,
               std::string memory_node_addr,
               std::string nic,
               size_t tcp_send_buffer_size,
               bool do_extra_copy);
  virtual ~SenderWorker();

  bool Initialize();

  pthread_t thread_handle() const { return thread_; }
  int worker_id() const { return worker_id_; }

  DataBlockQueue& GetWorkingQueue() { return working_queue_; }
  DataBlockQueue& GetDoneQueue() { return done_queue_; }

  void Stop();

 private:
  static void* ThreadFunction(void* arg);
  void Work();
  
  int worker_id_;
  std::string memory_node_addr_;
  std::string nic_;
  
  pthread_t thread_;
  bool stop_;

  DataBlockQueue working_queue_;
  DataBlockQueue done_queue_;

  int socket_;
  size_t tcp_send_buffer_size_;
  bool do_extra_copy_;
  uint8_t *copy_buffer_;
};
  

#endif
