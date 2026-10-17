// Loader equivalence: mapped (mmap / MapViewOfFile) vs --no-mmap (std::ifstream) give
// identical labels and normalised floats for every MNIST and CIFAR-10 image. Also checks
// the documented format details the conv layer relies on: pixel / 255 normalisation,
// CIFAR-10 plane order R, G, B (via the known per-channel training means) and known labels.
//
// Usage: test_loader <data dir>   (the folder holding train-images-idx3-ubyte/ and CIFAR10/)

#include <cmath>
#include <cstring>
#include "check.h"
#include "dataset.h"

int main(int argc, char** argv)
{
    const std::string data = argc > 1 ? argv[1] : "..";

    {
        auto a = load_dataset("mnist", data, true);
        auto b = load_dataset("mnist", data, false);
        CHECK(a->size() == 60000 && b->size() == 60000, "MNIST size");
        size_t diff_f = 0, diff_l = 0;
        std::vector<float> fa(MNIST_PIXELS);
        for (uint32_t i = 0; i < a->size(); ++i) {
            std::memcpy(fa.data(), a->input(i), MNIST_PIXELS * sizeof(float));
            if (std::memcmp(fa.data(), b->input(i), MNIST_PIXELS * sizeof(float))) ++diff_f;
            if (a->label(i) != b->label(i)) ++diff_l;
        }
        CHECK(diff_f == 0 && diff_l == 0, "MNIST mmap vs ifstream: %zu float / %zu label diffs", diff_f, diff_l);
        CHECK(a->label(0) == 5, "MNIST first training label is 5");
        const float* x = a->input(0);
        bool range = true;
        for (uint32_t j = 0; j < MNIST_PIXELS; ++j) range = range && x[j] >= 0.0f && x[j] <= 1.0f;
        CHECK(range, "MNIST range");
        std::printf("MNIST: 60000 images identical across loaders\n");
    }
    {
        auto a = load_dataset("cifar10", data, true);
        auto b = load_dataset("cifar10", data, false);
        CHECK(a->size() == 60000, "CIFAR size");
        size_t diff_f = 0, diff_l = 0;
        std::vector<float> fa(CIFAR10_PIXELS);
        double plane[3] = {0, 0, 0};
        for (uint32_t i = 0; i < a->size(); ++i) {
            std::memcpy(fa.data(), a->input(i), CIFAR10_PIXELS * sizeof(float));
            if (std::memcmp(fa.data(), b->input(i), CIFAR10_PIXELS * sizeof(float))) ++diff_f;
            if (a->label(i) != b->label(i)) ++diff_l;
            if (i < CIFAR10_N_TRAIN)
                for (int c = 0; c < 3; ++c)
                    for (int k = 0; k < 1024; ++k) plane[c] += fa[c * 1024 + k];
        }
        CHECK(diff_f == 0 && diff_l == 0, "CIFAR mmap vs ifstream: %zu float / %zu label diffs", diff_f, diff_l);
        // Known labels: data_batch_1 starts with a frog (6), test_batch with a cat (3).
        CHECK(a->label(0) == 6, "CIFAR first training label %d", a->label(0));
        CHECK(a->label(CIFAR10_N_TRAIN) == 3, "CIFAR first test label %d", a->label(CIFAR10_N_TRAIN));
        // Published per-channel means of the 50k training images: R 0.4914, G 0.4822, B 0.4465.
        const double n = 50000.0 * 1024;
        const double m[3] = {plane[0] / n, plane[1] / n, plane[2] / n};
        std::printf("CIFAR-10 plane means: %.4f %.4f %.4f (expected R 0.4914, G 0.4822, B 0.4465)\n", m[0], m[1], m[2]);
        CHECK(std::fabs(m[0] - 0.4914) < 1e-3 && std::fabs(m[1] - 0.4822) < 1e-3 && std::fabs(m[2] - 0.4465) < 1e-3,
              "CIFAR plane order is not R, G, B");
        auto& cd = static_cast<Cifar10Dataset&>(*a).raw_data();
        CHECK(a->input(17)[5] == cd.raw(17)[5] / 255.0f, "normalisation");
        std::printf("CIFAR-10: 60000 images identical across loaders\n");
    }
    return finish("test_loader");
}
