#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

// MNIST loader that maps the IDX files instead of copying them into each process.
//
// The original `Data` class in data.cpp allocates an Image per digit holding both the raw
// bytes and a parallel float array, 3936 bytes each, so 60 000 images cost 236 MB of
// private memory per worker. At 190 workers that is 45 GB. The CIFAR-10 loader already
// moved to mmap for the same reason; this does the same for MNIST, so a worker's dataset
// footprint is shared page cache rather than private pages, and file-backed clean pages
// never touch swap.
//
// The floats handed to the network are unchanged: pixel / 255.0f, produced here from a
// 256-entry lookup table so the value is identical without paying for a division per
// pixel. mnist_equiv_check.cpp proves the equivalence against the original loader over
// all 60 000 images before any of this is used for results.
//
// IDX format: big-endian header (magic, count, rows, cols) then rows*cols bytes per
// image. Labels: big-endian header (magic, count) then one byte per label.

#define MNIST_PIXELS      784u    // 28 * 28
#define MNIST_IMG_HEADER  16u
#define MNIST_LBL_HEADER  8u

class MnistData
{
public:
    MnistData(const std::string& images_path, const std::string& labels_path);
    ~MnistData();

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
    struct Mapping { const uint8_t* base = nullptr; size_t len = 0; };

    Mapping map_file(const std::string& path);

    Mapping img_map, lbl_map;
    const uint8_t* img_base = nullptr;
    const uint8_t* lbl_base = nullptr;
    uint32_t n_images = 0;

    float lut[256];                         // lut[b] == b / 255.0f
    mutable std::vector<float> scratch;     // MNIST_PIXELS floats
};

#endif
