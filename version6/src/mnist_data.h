#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include "mapped_file.h"

// MNIST loader over a read-only file view (mapped by default, see mapped_file.h).
//
// The floats handed to the network are pixel / 255.0f, produced from a 256-entry lookup
// table so the value is identical without paying for a division per pixel. Image layout
// is CHW with C = 1: 28 rows of 28 pixels, row-major, exactly the IDX byte order.
//
// IDX format: big-endian header (magic, count, rows, cols) then rows*cols bytes per
// image. Labels: big-endian header (magic, count) then one byte per label.

#define MNIST_PIXELS      784u    // 28 * 28
#define MNIST_IMG_HEADER  16u
#define MNIST_LBL_HEADER  8u

class MnistData
{
public:
    MnistData(const std::string& images_path, const std::string& labels_path, bool use_mmap = true);

    MnistData(const MnistData&) = delete;
    MnistData& operator=(const MnistData&) = delete;

    uint32_t size() const { return n_images; }

    uint8_t label(uint32_t i) const { return lbl_base[MNIST_LBL_HEADER + i]; }

    // Normalises image i into the internal scratch buffer and returns it. Valid until the
    // next call -- the training loop uses one sample at a time.
    const float* normalized(uint32_t i) const;

    const uint8_t* raw(uint32_t i) const
    {
        return img_base + MNIST_IMG_HEADER + (size_t)i * MNIST_PIXELS;
    }

private:
    MappedFile img_map, lbl_map;
    const uint8_t* img_base = nullptr;
    const uint8_t* lbl_base = nullptr;
    uint32_t n_images = 0;

    float lut[256];                         // lut[b] == b / 255.0f
    mutable std::vector<float> scratch;     // MNIST_PIXELS floats
};

#endif
