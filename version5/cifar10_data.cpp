#include "cifar10_data.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

Cifar10Data::Cifar10Data(const std::string& dir)
{
    for (int b = 0; b < 256; ++b)
        lut[b] = b / 255.0f;
    scratch.resize(CIFAR10_PIXELS);

    n_images = CIFAR10_N_TRAIN + CIFAR10_N_TEST;
    pixels.resize((size_t)n_images * CIFAR10_PIXELS);
    labels.resize(n_images);

    // [0, 50000) -- the five training batches, in file order.
    for (int b = 1; b <= 5; ++b) {
        std::string path = dir + "/CIFAR10/train_images/data_batch_" + std::to_string(b) + ".bin";
        load_batch(path, 10000u, (uint32_t)(b - 1) * 10000u);
    }
    // [50000, 60000) -- the official test batch.
    load_batch(dir + "/CIFAR10/test_images/test_batch.bin", CIFAR10_N_TEST, CIFAR10_N_TRAIN);

    std::cout << "CIFAR-10 loaded successfully" << std::endl;
    std::cout << "Train batches: " << CIFAR10_N_TRAIN << std::endl;
    std::cout << "Test batch: " << CIFAR10_N_TEST << std::endl;
}

void Cifar10Data::load_batch(const std::string& path, uint32_t expected_records, uint32_t offset)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("load_batch() | Failed to open file: " + path);

    // A CIFAR-10 batch is a headerless run of fixed-size records, so the file length is
    // the only integrity check available before reading.
    file.seekg(0, std::ios::end);
    std::streamoff bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    if (bytes != (std::streamoff)expected_records * CIFAR10_RECORD)
        throw std::runtime_error("load_batch() | " + path + " is " + std::to_string(bytes) +
                                 " bytes, expected " + std::to_string((size_t)expected_records * CIFAR10_RECORD));

    for (uint32_t i = 0; i < expected_records; ++i) {
        int lab = file.get();
        if (lab < 0 || lab > 9)
            throw std::runtime_error("load_batch() | " + path + " record " + std::to_string(i) +
                                     " has label " + std::to_string(lab));
        labels[offset + i] = (uint8_t)lab;
        file.read(reinterpret_cast<char*>(&pixels[(size_t)(offset + i) * CIFAR10_PIXELS]), CIFAR10_PIXELS);
    }
    if (!file)
        throw std::runtime_error("load_batch() | short read on " + path);
}

const float* Cifar10Data::normalized(uint32_t i) const
{
    const uint8_t* p = &pixels[(size_t)i * CIFAR10_PIXELS];
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
