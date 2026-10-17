#include "cifar10_data.h"
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

Cifar10Data::Cifar10Data(const std::string& dir, bool use_mmap)
{
    for (int b = 0; b < 256; ++b)
        lut[b] = b / 255.0f;
    scratch.resize(CIFAR10_PIXELS);

    const fs::path root = fs::path(dir) / "CIFAR10";
    for (int b = 1; b <= 5; ++b)
        map_batch((root / "train_images" / ("data_batch_" + std::to_string(b) + ".bin")).string(),
                  CIFAR10_PER_BATCH, b - 1, use_mmap);
    map_batch((root / "test_images" / "test_batch.bin").string(), CIFAR10_N_TEST, 5, use_mmap);

    validate_labels();
}

void Cifar10Data::map_batch(const std::string& path, uint32_t expected_records, int slot, bool use_mmap)
{
    maps[slot].open(path, use_mmap);
    // A CIFAR-10 batch is a headerless run of fixed-size records, so the file length is
    // the only integrity check available before reading.
    const size_t want = (size_t)expected_records * CIFAR10_RECORD;
    if (maps[slot].size() != want)
        throw std::runtime_error("Cifar10Data | " + path + " is " + std::to_string(maps[slot].size()) +
                                 " bytes, expected " + std::to_string(want));
}

void Cifar10Data::validate_labels() const
{
    for (uint32_t i = 0; i < size(); ++i) {
        const uint8_t l = label(i);
        if (l > 9)
            throw std::runtime_error("Cifar10Data | image " + std::to_string(i) +
                                     " has label " + std::to_string((int)l));
    }
}

const float* Cifar10Data::normalized(uint32_t i) const
{
    const uint8_t* p = raw(i);
    float* d = scratch.data();
    for (uint32_t j = 0; j < CIFAR10_PIXELS; ++j)
        d[j] = lut[p[j]];
    return d;
}
