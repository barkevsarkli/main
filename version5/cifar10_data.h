#ifndef CIFAR10_DATA_H
#define CIFAR10_DATA_H

#include <string>
#include <vector>
#include <cstdint>

// CIFAR-10 loader for version5.
//
// Deliberately does not reuse version5's MNIST `Image` struct.  That struct keeps a
// parallel `float normalized_pixels[]` next to the raw bytes, which costs 3936 B per
// 28x28 image (~236 MB for MNIST).  The same layout for 32x32x3 images would be
// 3072 + 3072*4 = 15.4 kB each, about 922 MB per worker -- more than an 8 GB machine
// can run several of.  Here the pixels are kept as raw bytes (~185 MB for all 60 000
// images) and normalised one sample at a time into a scratch buffer via a 256-entry
// lookup table, which produces exactly the same floats as `pixel / 255.0f` without
// paying for 3072 divisions per sample.
//
// Pixel order is the on-disk order: planar 1024 R, then 1024 G, then 1024 B.  A fully
// connected first layer is invariant to any fixed permutation of its inputs, so the
// bytes are not re-interleaved.
//
// Images are concatenated as [0, 50000) = the five training batches in file order,
// [50000, 60000) = the official test batch.  main.cpp uses the second range as the
// test set and splits the first into train/val with the fixed SPLIT_SEED permutation.

#define CIFAR10_PIXELS      3072u   // 32 * 32 * 3
#define CIFAR10_RECORD      3073u   // 1 label byte + CIFAR10_PIXELS
#define CIFAR10_N_TRAIN     50000u
#define CIFAR10_N_TEST      10000u

class Cifar10Data
{
public:
    // `dir` is the --data prefix; expects <dir>/CIFAR10/train_images/data_batch_{1..5}.bin
    // and <dir>/CIFAR10/test_images/test_batch.bin.
    explicit Cifar10Data(const std::string& dir);

    uint32_t size()       const { return n_images; }
    uint32_t train_count() const { return CIFAR10_N_TRAIN; }   // indices [0, 50000)
    uint32_t test_count()  const { return CIFAR10_N_TEST; }    // indices [50000, 60000)

    uint8_t label(uint32_t i) const { return labels[i]; }

    // Normalises image i into the internal scratch buffer and returns it.  The pointer
    // is valid until the next call -- the training loop uses one sample at a time.
    const float* normalized(uint32_t i) const;

    // Raw bytes of image i (planar RGB), for sanity checks and BMP dumps.
    const uint8_t* raw(uint32_t i) const { return &pixels[(size_t)i * CIFAR10_PIXELS]; }

    // Writes image i as a 32x32 BMP.  Used to eyeball channel order, which no numeric
    // check can catch.
    void display_image(uint32_t i, const std::string& filename) const;

private:
    void load_batch(const std::string& path, uint32_t expected_records, uint32_t offset);

    std::vector<uint8_t>  pixels;   // n_images * CIFAR10_PIXELS
    std::vector<uint8_t>  labels;   // n_images
    uint32_t              n_images = 0;

    float                 lut[256];         // lut[b] == b / 255.0f
    mutable std::vector<float> scratch;     // CIFAR10_PIXELS floats
};

#endif
