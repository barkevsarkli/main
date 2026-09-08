#ifndef DATASET_H
#define DATASET_H

#include <cstdint>
#include "data.h"
#include "cifar10_data.h"

// A uniform read-only view over whichever dataset a run was asked for, so main.cpp's
// training loop and evaluate() do not care which one is loaded.
//
// The MNIST view is a pure pass-through: input(i) returns the very pointer the loop
// used before this abstraction existed (&Image::normalized_pixels[0]), and label(i)
// returns Image::label.  Nothing about the MNIST arithmetic changes.
//
// One-hot targets are produced on demand as (j == label ? 1.0f : 0.0f) rather than read
// from Image::one_hot_encoded_label, which held the same 0/1 values as uint8_t and
// converted to exactly these floats at every use.
struct Dataset
{
    virtual ~Dataset() = default;
    virtual uint32_t       size()       const = 0;
    virtual int            input_size() const = 0;
    virtual const float*   input(uint32_t i) const = 0;
    virtual uint8_t        label(uint32_t i) const = 0;
};

class MnistDataset : public Dataset
{
public:
    // Keeps the exact call version5 has always made: everything in "train", the
    // train/val/test split is done by main.cpp over indices.
    MnistDataset(const std::string& images, const std::string& labels)
        : data(images, labels, 0.0f), imgs(data.get_train_images()), n(data.get_train_size()) {}

    uint32_t     size()       const override { return n; }
    int          input_size() const override { return 28 * 28; }
    const float* input(uint32_t i) const override { return imgs[i].normalized_pixels; }
    uint8_t      label(uint32_t i) const override { return imgs[i].label; }

private:
    Data     data;
    Image*   imgs;
    uint32_t n;
};

class Cifar10Dataset : public Dataset
{
public:
    explicit Cifar10Dataset(const std::string& dir) : data(dir) {}

    uint32_t     size()       const override { return data.size(); }
    int          input_size() const override { return (int)CIFAR10_PIXELS; }
    const float* input(uint32_t i) const override { return data.normalized(i); }
    uint8_t      label(uint32_t i) const override { return data.label(i); }

    const Cifar10Data& raw_data() const { return data; }

private:
    Cifar10Data data;
};

#endif
