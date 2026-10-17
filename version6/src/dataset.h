#ifndef DATASET_H
#define DATASET_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "mnist_data.h"
#include "cifar10_data.h"

// A uniform read-only view over whichever dataset a run was asked for, so the training
// loop does not care which one is loaded. Inputs are CHW: MNIST 1x28x28, CIFAR-10
// 3x32x32. A dense first layer sees the same floats flattened in that order.
struct Dataset
{
    virtual ~Dataset() = default;
    virtual uint32_t       size()       const = 0;
    virtual int            input_size() const = 0;
    virtual int            channels()   const = 0;
    virtual int            height()     const = 0;
    virtual int            width()      const = 0;
    virtual const float*   input(uint32_t i) const = 0;
    virtual uint8_t        label(uint32_t i) const = 0;
};

class MnistDataset : public Dataset
{
public:
    MnistDataset(const std::string& images, const std::string& labels, bool use_mmap)
        : data(images, labels, use_mmap) {}

    uint32_t     size()       const override { return data.size(); }
    int          input_size() const override { return (int)MNIST_PIXELS; }
    int          channels()   const override { return 1; }
    int          height()     const override { return 28; }
    int          width()      const override { return 28; }
    const float* input(uint32_t i) const override { return data.normalized(i); }
    uint8_t      label(uint32_t i) const override { return data.label(i); }

private:
    MnistData data;
};

class Cifar10Dataset : public Dataset
{
public:
    Cifar10Dataset(const std::string& dir, bool use_mmap) : data(dir, use_mmap) {}

    uint32_t     size()       const override { return data.size(); }
    int          input_size() const override { return (int)CIFAR10_PIXELS; }
    int          channels()   const override { return 3; }
    int          height()     const override { return 32; }
    int          width()      const override { return 32; }
    const float* input(uint32_t i) const override { return data.normalized(i); }
    uint8_t      label(uint32_t i) const override { return data.label(i); }

    const Cifar10Data& raw_data() const { return data; }

private:
    Cifar10Data data;
};

// The fixed train/val/test split shared by every run (version5 protocol, SPLIT_SEED 12345).
//   MNIST:    test = first 10k of one permutation of the 60k training file, val = next 5k,
//             train = the remaining 45k.
//   CIFAR-10: val = first 5k of a permutation of the 50k training images, train = the
//             remaining 45k, test = the official test batch (indices 50000..59999).
// train_eval holds the fixed 10k subset used for clean training accuracy.
struct Split
{
    std::vector<uint32_t> train, val, test, train_eval;
};

#define SPLIT_SEED 12345u
#define N_TEST 10000
#define N_VAL   5000

std::unique_ptr<Dataset> load_dataset(const std::string& name, const std::string& data_dir, bool use_mmap);
Split make_split(const std::string& name, const Dataset& ds);

#endif
