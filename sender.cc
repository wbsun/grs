#include "sender.h"
#include "net_utils.h"

Sender::Sender(AddressNICVector& memory_nodes) {
}

Sender::~Sender() {
  
}

bool Sender::Initialize() {
}

void Sender::Stop() {
  
}

SenderWorker::SenderWorker(int id, std::string memory_node_addr,
                           std::string nic, size_t tcp_send_buffer_size,
                           bool do_extra_copy)
    : worker_id_(id),
      memory_node_addr_(memory_node_addr),
      nic_(nic),
      tcp_send_buffer_size_(tcp_send_buffer_size),
      do_extra_copy_(do_extra_copy) {
  if (do_extra_copy_) {
    copy_buffer_ = new uint8_t[tcp_send_buffer_size_];
    assert(copy_buffer_ != NULL);
  } else {
    copy_buffer_ = NULL;
  }  
}

SenderWorker::~SenderWorker() {
  NetUtils::CloseSocket(socket_);
  delete[] copy_buffer_; 
}

bool SenderWorker::Initialize() {
  stop_ = false;
  socket_ = NetUtils::ConnectToMemoryNode(
      memory_node_addr_, nic_, send_buffer_size_);
  ssc(::pthread_create(&thread_, NULL,
                       SenderWorker::ThreadFunction, (void*)this));
  return true;
}

void SenderWorker::Stop() {
  stop_ = true;
  ssc(::pthread_join(thread_, NULL));
}

void* SenderWorker::ThreadFunction(void* arg) {
  SenderWorker* worker = (SenderWorker*)arg;
  worker->Work();
  return NULL;
}

void SenderWorker::Work() {
  while (!stop_) {
    DataBlock block = working_queue_.Pop();
    NetUtils::SendDataBlock(socket_, block, do_extra_copy_, copy_buffer_);
    done_queue_.Push(block);
  }
}
