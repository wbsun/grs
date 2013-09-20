#include "data_block_queue.h"

DataBlockQueue::DataBlockQueue() {
  ssc(sem_init(&sema_, 0, 0));
  ssc(pthread_spin_init(&lock_, PTHREAD_PROCESS_PRIVATE));
}

DataBlockQueue::~DataBlockQueue() {
  ssc(pthread_spin_destroy(&lock_));
  ssc(sem_destroy(&sema_));
}

void DataBlockQueue::Push(DataBlock block) {
  ssc(pthread_spin_lock(&lock_));
  queue_.push(block);

  // Consderig swap the following two lines, there is just one producer
  // and one consumer, so it is OK to swap them, plus it can avoid
  // unnecessary waiting.
  ssc(sem_post(&sema_));
  ssc(pthread_spin_unlock(&lock_));
}

DataBlock DataBlockQueue::Pop() {
  ssc(sem_wait(&sema_));
  ssc(pthread_spin_lock(&lock_));
  assert(!queue_.empty());
  DataBlock block = queue_.front();
  queue_.pop();
  ssc(pthread_spin_unlock(&lock_));
  return block;
}

size_t DataBlockQueue::Size() {
  ssc(pthread_spin_lock(&lock_));
  size_t sz = queue_.size();
  ssc(pthread_spin_unlock(&lock_));
  return sz;
}
