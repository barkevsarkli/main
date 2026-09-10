// Proves the mmap MNIST loader hands the network exactly the same bytes and floats as the
// original buffered `Data` class, over all 60,000 images. This is what licenses replacing
// the loader without re-verifying every stored result: the arithmetic downstream cannot
// tell the two apart.
//
//   clang++ -std=c++17 -O2 -o mnist_equiv_check mnist_equiv_check.cpp mnist_data.cpp data.cpp
//   ./mnist_equiv_check ..
#include "mnist_data.h"
#include "data.h"
#include <iostream>
#include <cstring>

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "..";
    const std::string img = dir + "/train-images-idx3-ubyte/train-images-idx3-ubyte";
    const std::string lbl = dir + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    std::cout.setstate(std::ios::failbit);          // silence both loaders
    Data      old_loader(img, lbl, 0.0f);
    MnistData new_loader(img, lbl);
    std::cout.clear();

    Image*   old_imgs = old_loader.get_train_images();
    uint32_t n_old    = old_loader.get_train_size();

    if (n_old != new_loader.size()) {
        std::cout << "FAIL: image count " << n_old << " vs " << new_loader.size() << "\n";
        return 1;
    }

    size_t bad_label = 0, bad_pixel = 0, bad_float = 0;
    for (uint32_t i = 0; i < n_old; ++i) {
        if (old_imgs[i].label != new_loader.label(i)) ++bad_label;
        if (std::memcmp(old_imgs[i].pixels, new_loader.raw(i), MNIST_PIXELS) != 0) ++bad_pixel;

        const float* nf = new_loader.normalized(i);
        for (uint32_t j = 0; j < MNIST_PIXELS; ++j) {
            // Bit-for-bit, not approximate: any difference at all would propagate.
            if (old_imgs[i].normalized_pixels[j] != nf[j]) { ++bad_float; break; }
        }
    }

    std::cout << "images compared      : " << n_old << "\n";
    std::cout << "label mismatches     : " << bad_label << "\n";
    std::cout << "raw byte mismatches  : " << bad_pixel << "\n";
    std::cout << "float mismatches     : " << bad_float << "\n";

    const bool ok = !bad_label && !bad_pixel && !bad_float;
    std::cout << (ok ? "\nIDENTICAL - the mmap loader is a drop-in replacement\n"
                     : "\nDIFFERENT - do not use the mmap loader\n");
    return ok ? 0 : 1;
}
