#ifndef __GRSKV_H__
#define __GRSKV_H__

static const char* const MEMORY_NODES[] = {
  "10.1.1.2",
  "10.1.3.2",
  "10.1.2.2",
  "10.1.6.2",
  "10.1.5.2",
  "10.1.4.2"
};

static const char* const CLIENT_NIC_TO_NODE[] = {
  "eth1",
  "eth2",
  "eth4",
  "eth7",
  "eth6",
  "eth5"
};

static const int NUM_MEMORY_NODES = sizeof(MEMORY_NODES)/sizeof(char*);

static const short MEMORY_NODE_PORT = 10000;

static const int READ_BUF_LEN = 1 << 24;

static const int WRITE_BUF_LEN = 1 << 22;

static const int SEND_BUF_LEN = 1 << 17;

#endif
