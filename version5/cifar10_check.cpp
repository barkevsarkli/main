// Standalone correctness checks for the CIFAR-10 loader.  Not part of ./main.
//
//   clang++ -std=c++17 -O2 -o cifar10_check cifar10_check.cpp cifar10_data.cpp && ./cifar10_check ..
//
// Checks the things that silently corrupt an experiment rather than crash it: record
// stride, label range, class balance, normalisation range, and that the fixed-seed split
// is deterministic and disjoint.  Also writes one image to BMP -- a planar/interleaved
// channel mix-up passes every numeric check but is obvious in the picture.

#include "cifar10_data.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define SPLIT_SEED 12345u
#define N_VAL 5000

static int failures = 0;

static void check(bool cond, const std::string& what)
{
    std::cout << (cond ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!cond) ++failures;
}

static std::vector<uint32_t> train_val_split(std::vector<uint32_t>& val)
{
    std::vector<uint32_t> perm(CIFAR10_N_TRAIN);
    std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937 rng(SPLIT_SEED);
    std::shuffle(perm.begin(), perm.end(), rng);
    val.assign(perm.begin(), perm.begin() + N_VAL);
    return std::vector<uint32_t>(perm.begin() + N_VAL, perm.end());
}

int main(int argc, char** argv)
{
    std::string dir = argc > 1 ? argv[1] : "..";
    Cifar10Data data(dir);

    std::cout << "\nrecord counts\n";
    check(data.size() == 60000u, "60000 records loaded");
    check(data.train_count() == 50000u, "50000 in the training batches");
    check(data.test_count() == 10000u, "10000 in the official test batch");

    std::cout << "\nlabels\n";
    int hist[10] = {};
    bool in_range = true;
    for (uint32_t i = 0; i < data.size(); ++i) {
        uint8_t l = data.label(i);
        if (l > 9) { in_range = false; break; }
        hist[l]++;
    }
    check(in_range, "every label is in [0,9]");
    bool balanced = true;
    for (int c = 0; c < 10; ++c) balanced = balanced && hist[c] == 6000;
    check(balanced, "exactly 6000 images per class (catches record-stride errors)");
    std::cout << "        histogram:";
    for (int c = 0; c < 10; ++c) std::cout << " " << hist[c];
    std::cout << "\n";

    int test_hist[10] = {};
    for (uint32_t i = CIFAR10_N_TRAIN; i < data.size(); ++i) test_hist[data.label(i)]++;
    bool test_balanced = true;
    for (int c = 0; c < 10; ++c) test_balanced = test_balanced && test_hist[c] == 1000;
    check(test_balanced, "exactly 1000 per class in the test batch");

    std::cout << "\nnormalisation\n";
    bool range_ok = true, lut_exact = true;
    for (uint32_t i = 0; i < data.size(); i += 97) {           // a strided sample of every batch
        const float* x = data.normalized(i);
        const uint8_t* p = data.raw(i);
        for (uint32_t j = 0; j < CIFAR10_PIXELS; ++j) {
            if (x[j] < 0.0f || x[j] > 1.0f) range_ok = false;
            if (x[j] != p[j] / 255.0f) lut_exact = false;
        }
    }
    check(range_ok, "all normalised pixels in [0,1]");
    check(lut_exact, "lookup table gives bit-identical results to pixel / 255.0f");

    std::cout << "\nsplit\n";
    std::vector<uint32_t> val, train = train_val_split(val);
    check(train.size() == 45000u && val.size() == 5000u, "45000 train / 5000 val");
    std::vector<uint32_t> both = train;
    both.insert(both.end(), val.begin(), val.end());
    std::sort(both.begin(), both.end());
    bool covers = both.size() == CIFAR10_N_TRAIN;
    for (uint32_t i = 0; i < both.size() && covers; ++i) covers = both[i] == i;
    check(covers, "train and val are disjoint and cover exactly [0,50000)");
    bool below_test = *std::max_element(both.begin(), both.end()) < CIFAR10_N_TRAIN;
    check(below_test, "no training index reaches into the test batch [50000,60000)");

    std::vector<uint32_t> val2, train2 = train_val_split(val2);
    check(train2 == train && val2 == val, "the same SPLIT_SEED gives the same split twice");

    std::cout << "\nvisual check\n";
    const char* names[10] = {"airplane", "automobile", "bird", "cat", "deer",
                             "dog", "frog", "horse", "ship", "truck"};
    for (uint32_t i = 0; i < 4; ++i) {
        std::string f = "cifar10_sample_" + std::to_string(i) + "_" + names[data.label(i)] + ".bmp";
        data.display_image(i, f);
        std::cout << "  wrote " << f << "\n";
    }

    std::cout << "\n" << (failures ? "FAILED: " + std::to_string(failures) + " check(s)" : "all checks passed") << "\n";
    return failures ? 1 : 0;
}
