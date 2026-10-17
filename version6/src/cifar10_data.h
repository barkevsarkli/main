#ifndef CIFAR10_DATA_H
#define CIFAR10_DATA_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include "mapped_file.h"

// CIFAR-10 loader over read-only file views (mapped by default, see mapped_file.h).
//
// A record is 3073 bytes: 1 label byte then 3072 pixel bytes, planar 1024 R, 1024 G,
// 1024 B, each plane 32 rows of 32 pixels. That is already CHW order (C = 3, R first), so
// the convolution reads the bytes as-is; tests/test_loader.cpp checks the plane order
// against the documented format.
//
// Images are indexed [0, 50000) = the five training batches in file order, and
// [50000, 60000) = the official test batch.  main.cpp uses the second range as the test
// set and splits the first into train/val with the fixed SPLIT_SEED permutation.

#define CIFAR10_PIXELS      3072u   // 32 * 32 * 3
#define CIFAR10_RECORD      3073u   // 1 label byte + CIFAR10_PIXELS
#define CIFAR10_PER_BATCH   10000u
#define CIFAR10_N_TRAIN     50000u
#define CIFAR10_N_TEST      10000u

class Cifar10Data
{
public:
    // `dir` is the --data prefix; expects <dir>/CIFAR10/train_images/data_batch_{1..5}.bin
    // and <dir>/CIFAR10/test_images/test_batch.bin.
    explicit Cifar10Data(const std::string& dir, bool use_mmap = true);

    Cifar10Data(const Cifar10Data&) = delete;
    Cifar10Data& operator=(const Cifar10Data&) = delete;

    uint32_t size()        const { return CIFAR10_N_TRAIN + CIFAR10_N_TEST; }
    uint32_t train_count() const { return CIFAR10_N_TRAIN; }   // indices [0, 50000)
    uint32_t test_count()  const { return CIFAR10_N_TEST; }    // indices [50000, 60000)

    uint8_t label(uint32_t i) const { return *record(i); }

    // Normalises image i into the internal scratch buffer and returns it.  The pointer is
    // valid until the next call -- the training loop uses one sample at a time.
    const float* normalized(uint32_t i) const;

    // Raw pixel bytes of image i (planar RGB).
    const uint8_t* raw(uint32_t i) const { return record(i) + 1; }

private:
    void map_batch(const std::string& path, uint32_t expected_records, int slot, bool use_mmap);
    void validate_labels() const;

    // Start of record i, which is its label byte.
    const uint8_t* record(uint32_t i) const
    {
        const uint32_t slot = (i < CIFAR10_N_TRAIN) ? (i / CIFAR10_PER_BATCH) : 5u;
        const uint32_t r    = (i < CIFAR10_N_TRAIN) ? (i % CIFAR10_PER_BATCH) : (i - CIFAR10_N_TRAIN);
        return maps[slot].data() + (size_t)r * CIFAR10_RECORD;
    }

    MappedFile maps[6];                     // 5 training batches, then the test batch
    float      lut[256];                    // lut[b] == b / 255.0f
    mutable std::vector<float> scratch;     // CIFAR10_PIXELS floats
};

#endif
