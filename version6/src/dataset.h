#ifndef DATASET_H
#define DATASET_H

#include <cstdint>
#include "mnist_data.h"
#include "cifar10_data.h"

// A uniform read-only view over whichever dataset a run was asked for, so main.cpp's
// training loop and evaluate() do not care which one is loaded.
//
// Both views normalise one sample at a time out of a memory-mapped file, so a worker's
// dataset footprint is shared page cache rather than private pages.  The floats are
// unchanged: mnist_equiv_check.cpp compares the mapped MNIST loader against the original
// buffered one over all 60 000 images and finds zero differences in labels, raw bytes or
// normalised floats.
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
    MnistDataset(const std::string& images, const std::string& labels) : data(images, labels) {}

    uint32_t     size()       const override { return data.size(); }
    int          input_size() const override { return (int)MNIST_PIXELS; }
    const float* input(uint32_t i) const override { return data.normalized(i); }
    uint8_t      label(uint32_t i) const override { return data.label(i); }

private:
    MnistData data;
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
