/**
 * zfec -- fast forward error correction library with Python interface
 *
 * See README.rst for documentation.
 */
#ifndef ZFEC_H
#define ZFEC_H
#include <stddef.h>
#include <stdint.h>

typedef uint8_t u8_t;
typedef uint16_t u16_t;
typedef uint32_t u32_t;
typedef uint64_t u64_t;

typedef struct {
  u32_t magic;
  u16_t k, n;                     /* parameters of the code */
  u8_t* enc_matrix;
} fec_t;

/**
 * param k the number of blocks required to reconstruct
 * param m the total number of blocks created
 */
fec_t* fec_new(u16_t k, u16_t m);
void fec_free(fec_t* p);

/**
 * @param src the "primary blocks" i.e. the chunks of the input data
 * @param fecs buffers into which the secondary blocks will be written
 * @param block_nums the numbers of the desired check blocks (the id >= k)
          which fec_encode() will produce and store into the buffers of
          the fecs parameter
 * @param num_block_nums the length of the block_nums array
 * @param sz size of a packet in bytes
 */
void fec_encode(const fec_t* code,
                const u8_t** src,
                u8_t** fecs,
                const u32_t* block_nums,
                size_t num_block_nums,
                size_t sz);

/**
 * @param inpkts an array of packets (size k);
          If a primary block, i, is present then it must be at index i.
          Secondary blocks can appear anywhere.
 * @param outpkts an array of buffers into which the reconstructed
          output packets will be written (only packets which are not
          present in the inpkts input will be reconstructed and written
          to outpkts)
 * @param index an array of the blocknums of the packets in inpkts
 * @param sz size of a packet in bytes
 */
void fec_decode(const fec_t* code,
                const u8_t** inpkts,
                u8_t** outpkts,
                const u32_t* index,
                size_t sz);


#ifndef alloca
#define alloca(x) __builtin_alloca(x)
#else
#include <alloca.h>
#endif

#endif
