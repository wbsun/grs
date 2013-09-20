#ifndef __BLOCK_QUEUE_H__
#define __BLOCK_QUEUE_H__

#include <pthread.h>
#include <semaphore.h>
#include <queue>
#include <cstdint>

struct DataBlock {
  uint8_t *data;
  size_t size;
  int block_num;
};

class DataBlockQueue {
 public:
  DataBlockQueue();
  virtual ~DataBlockQueue();

  void Push(DataBlock block);
  DataBlock Pop();

  size_t Size();

 private:
  std::queue<DataBlock> queue_;
  sem_t sema_;
  pthread_spinlock_t lock_;
};

#endif
