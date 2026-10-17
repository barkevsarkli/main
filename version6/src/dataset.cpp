#include "dataset.h"
#include "rng_util.h"
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

std::unique_ptr<Dataset> load_dataset(const std::string& name, const std::string& data_dir, bool use_mmap)
{
    if (name == "mnist") {
        const fs::path d(data_dir);
        return std::unique_ptr<Dataset>(new MnistDataset(
            (d / "train-images-idx3-ubyte" / "train-images-idx3-ubyte").string(),
            (d / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte").string(), use_mmap));
    }
    if (name == "cifar10")
        return std::unique_ptr<Dataset>(new Cifar10Dataset(data_dir, use_mmap));
    throw std::runtime_error("unknown dataset: " + name);
}

// Identical to version5/main.cpp, including the order of RNG use: one mt19937(SPLIT_SEED)
// for the split permutation and one mt19937(SPLIT_SEED + 1) for the clean-accuracy subset.
Split make_split(const std::string& name, const Dataset& ds)
{
    Split s;
    if (name == "mnist") {
        const uint32_t N = ds.size();
        std::vector<uint32_t> perm(N);
        std::iota(perm.begin(), perm.end(), 0u);
        { std::mt19937 split_rng(SPLIT_SEED); detu::shuffle(perm, split_rng); }
        s.test .assign(perm.begin(),                  perm.begin() + N_TEST);
        s.val  .assign(perm.begin() + N_TEST,         perm.begin() + N_TEST + N_VAL);
        s.train.assign(perm.begin() + N_TEST + N_VAL, perm.end());
    } else {
        std::vector<uint32_t> perm(CIFAR10_N_TRAIN);
        std::iota(perm.begin(), perm.end(), 0u);
        { std::mt19937 split_rng(SPLIT_SEED); detu::shuffle(perm, split_rng); }
        s.val  .assign(perm.begin(),         perm.begin() + N_VAL);
        s.train.assign(perm.begin() + N_VAL, perm.end());
        s.test.resize(CIFAR10_N_TEST);
        std::iota(s.test.begin(), s.test.end(), CIFAR10_N_TRAIN);   // the official test batch
    }

    // Clean training accuracy uses a fixed 10 000-image subset of the training split, drawn
    // once with SPLIT_SEED + 1 (version5 protocol).
    s.train_eval = s.train;
    { std::mt19937 sub_rng(SPLIT_SEED + 1u); detu::shuffle(s.train_eval, sub_rng); }
    if (s.train_eval.size() > (size_t)N_TEST) s.train_eval.resize(N_TEST);
    return s;
}
