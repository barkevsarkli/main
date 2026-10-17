#include "mnist_data.h"
#include <stdexcept>

static uint32_t be32(const uint8_t* p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

MnistData::MnistData(const std::string& images_path, const std::string& labels_path, bool use_mmap)
{
    for (int b = 0; b < 256; ++b)
        lut[b] = b / 255.0f;
    scratch.resize(MNIST_PIXELS);

    img_map.open(images_path, use_mmap);
    lbl_map.open(labels_path, use_mmap);
    img_base = img_map.data();
    lbl_base = lbl_map.data();

    if (img_map.size() < MNIST_IMG_HEADER || be32(img_base) != 2051u)
        throw std::runtime_error("MnistData | bad image magic in " + images_path);
    if (lbl_map.size() < MNIST_LBL_HEADER || be32(lbl_base) != 2049u)
        throw std::runtime_error("MnistData | bad label magic in " + labels_path);

    n_images = be32(img_base + 4);
    const uint32_t rows = be32(img_base + 8);
    const uint32_t cols = be32(img_base + 12);
    if (rows != 28 || cols != 28)
        throw std::runtime_error("MnistData | expected 28x28, got " + std::to_string(rows) + "x" + std::to_string(cols));

    const uint32_t n_labels = be32(lbl_base + 4);
    if (n_labels != n_images)
        throw std::runtime_error("MnistData | " + std::to_string(n_images) + " images but " +
                                 std::to_string(n_labels) + " labels");

    const size_t want_img = (size_t)MNIST_IMG_HEADER + (size_t)n_images * MNIST_PIXELS;
    const size_t want_lbl = (size_t)MNIST_LBL_HEADER + n_images;
    if (img_map.size() < want_img || lbl_map.size() < want_lbl)
        throw std::runtime_error("MnistData | file shorter than its header claims");

    for (uint32_t i = 0; i < n_images; ++i)
        if (label(i) > 9)
            throw std::runtime_error("MnistData | image " + std::to_string(i) +
                                     " has label " + std::to_string((int)label(i)));
}

const float* MnistData::normalized(uint32_t i) const
{
    const uint8_t* p = raw(i);
    float* d = scratch.data();
    for (uint32_t j = 0; j < MNIST_PIXELS; ++j)
        d[j] = lut[p[j]];
    return d;
}
