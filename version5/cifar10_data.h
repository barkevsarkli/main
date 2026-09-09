#ifndef CIFAR10_DATA_H
#define CIFAR10_DATA_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

// CIFAR-10 loader for version5.
//
// The batch files are mapped read-only with mmap rather than read into a per-process
// buffer.  This matters on an 8 GB machine: a private 185 MB copy per worker meant six
// workers needed 1.1 GB of resident memory, and at width 512 that could not be held --
// free memory fell to 61 MB, the machine sustained ~52 MB/s of swap-in traffic, and the
// workers ran at 26% of the CPU they should have.  Mapped file pages are shared between
// every worker by the unified buffer cache, so the six of them now cost one 180 MB
// footprint between them instead of six.  Being file-backed and clean, those pages are
// also evicted straight back to disk and never consume swap.
//
// The bytes handed to the network are unchanged, so results are bit-identical to the
// buffered loader.
//
// A record is 3073 bytes: 1 label byte then 3072 pixel bytes, planar 1024 R, 1024 G,
// 1024 B.  A fully connected first layer is invariant to any fixed permutation of its
// inputs, so the planar order is used as-is.
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
    explicit Cifar10Data(const std::string& dir);
    ~Cifar10Data();

    Cifar10Data(const Cifar10Data&) = delete;
    Cifar10Data& operator=(const Cifar10Data&) = delete;

    uint32_t size()        const { return CIFAR10_N_TRAIN + CIFAR10_N_TEST; }
    uint32_t train_count() const { return CIFAR10_N_TRAIN; }   // indices [0, 50000)
    uint32_t test_count()  const { return CIFAR10_N_TEST; }    // indices [50000, 60000)

    uint8_t label(uint32_t i) const { return *record(i); }

    // Normalises image i into the internal scratch buffer and returns it.  The pointer is
    // valid until the next call -- the training loop uses one sample at a time.
    const float* normalized(uint32_t i) const;

    // Raw pixel bytes of image i (planar RGB), for sanity checks and BMP dumps.
    const uint8_t* raw(uint32_t i) const { return record(i) + 1; }

    // Writes image i as a 32x32 BMP.  Used to eyeball channel order, which no numeric
    // check can catch.
    void display_image(uint32_t i, const std::string& filename) const;

private:
    struct Mapping { const uint8_t* base = nullptr; size_t len = 0; };

    void map_batch(const std::string& path, uint32_t expected_records, int slot);
    void validate_labels() const;

    // Start of record i, which is its label byte.
    const uint8_t* record(uint32_t i) const
    {
        const uint32_t slot = (i < CIFAR10_N_TRAIN) ? (i / CIFAR10_PER_BATCH) : 5u;
        const uint32_t r    = (i < CIFAR10_N_TRAIN) ? (i % CIFAR10_PER_BATCH) : (i - CIFAR10_N_TRAIN);
        return maps[slot].base + (size_t)r * CIFAR10_RECORD;
    }

    Mapping maps[6];                        // 5 training batches, then the test batch
    float   lut[256];                       // lut[b] == b / 255.0f
    mutable std::vector<float> scratch;     // CIFAR10_PIXELS floats
};

#endif
