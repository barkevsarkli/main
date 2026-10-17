#include "cifar10_data.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

Cifar10Data::Cifar10Data(const std::string& dir)
{
    for (int b = 0; b < 256; ++b)
        lut[b] = b / 255.0f;
    scratch.resize(CIFAR10_PIXELS);

    for (int b = 1; b <= 5; ++b)
        map_batch(dir + "/CIFAR10/train_images/data_batch_" + std::to_string(b) + ".bin",
                  CIFAR10_PER_BATCH, b - 1);
    map_batch(dir + "/CIFAR10/test_images/test_batch.bin", CIFAR10_N_TEST, 5);

    validate_labels();

    std::cout << "CIFAR-10 mapped successfully" << std::endl;
    std::cout << "Train batches: " << CIFAR10_N_TRAIN << std::endl;
    std::cout << "Test batch: " << CIFAR10_N_TEST << std::endl;
}

Cifar10Data::~Cifar10Data()
{
    for (auto& m : maps)
        if (m.base) munmap(const_cast<uint8_t*>(m.base), m.len);
}

void Cifar10Data::map_batch(const std::string& path, uint32_t expected_records, int slot)
{
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("map_batch() | Failed to open file: " + path);

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("map_batch() | fstat failed on " + path);
    }

    // A CIFAR-10 batch is a headerless run of fixed-size records, so the file length is
    // the only integrity check available before reading.
    const size_t want = (size_t)expected_records * CIFAR10_RECORD;
    if ((size_t)st.st_size != want) {
        close(fd);
        throw std::runtime_error("map_batch() | " + path + " is " + std::to_string((long long)st.st_size) +
                                 " bytes, expected " + std::to_string(want));
    }

    void* p = mmap(nullptr, want, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);                      // the mapping keeps its own reference to the file
    if (p == MAP_FAILED)
        throw std::runtime_error("map_batch() | mmap failed on " + path);

    // Sequential passes over a shuffled index order, so tell the kernel not to bother
    // with read-ahead heuristics tuned for streaming.
    madvise(p, want, MADV_RANDOM);

    maps[slot].base = static_cast<const uint8_t*>(p);
    maps[slot].len  = want;
}

void Cifar10Data::validate_labels() const
{
    for (uint32_t i = 0; i < size(); ++i) {
        const uint8_t l = label(i);
        if (l > 9)
            throw std::runtime_error("validate_labels() | image " + std::to_string(i) +
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

void Cifar10Data::display_image(uint32_t i, const std::string& filename) const
{
    const int width = 32;
    const int height = 32;
    const int row_padded = (width * 3 + 3) & (~3);
    const int filesize = 14 + 40 + row_padded * height;

    unsigned char bmpfileheader[14] = {
        'B','M',
        0,0,0,0,
        0,0,
        0,0,
        54,0,0,0
    };
    unsigned char bmpinfoheader[40] = {
        40,0,0,0,
        0,0,0,0,
        0,0,0,0,
        1,0,
        24,0
    };

    bmpfileheader[2] = (unsigned char)(filesize      );
    bmpfileheader[3] = (unsigned char)(filesize >>  8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);

    bmpinfoheader[ 4] = (unsigned char)(width       );
    bmpinfoheader[ 5] = (unsigned char)(width  >>  8);
    bmpinfoheader[ 6] = (unsigned char)(width  >> 16);
    bmpinfoheader[ 7] = (unsigned char)(width  >> 24);
    bmpinfoheader[ 8] = (unsigned char)(height      );
    bmpinfoheader[ 9] = (unsigned char)(height >>  8);
    bmpinfoheader[10] = (unsigned char)(height >> 16);
    bmpinfoheader[11] = (unsigned char)(height >> 24);

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("display_image() | Failed to open file: " + filename);

    out.write(reinterpret_cast<char*>(bmpfileheader), 14);
    out.write(reinterpret_cast<char*>(bmpinfoheader), 40);

    const uint8_t* r = raw(i);
    const uint8_t* g = r + 1024;
    const uint8_t* b = r + 2048;
    unsigned char pad[3] = {0, 0, 0};

    // BMP rows run bottom-to-top and store pixels as B,G,R.
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int k = y * width + x;
            unsigned char bgr[3] = { b[k], g[k], r[k] };
            out.write(reinterpret_cast<char*>(bgr), 3);
        }
        out.write(reinterpret_cast<char*>(pad), row_padded - width * 3);
    }
}
