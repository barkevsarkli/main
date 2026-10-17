// kernel_dump -- build every fixed kernel bank, calibrate its per-filter gains and write
//   kernels/<dataset>_<family>_k<K>.json   (filters, gains, tau; read by the trainer)
//   kernels/<family>_k<K>.csv               (the L2-normalised 2-D filters, for inspection)
//
// Gains (PROMPT.md section 7): the pre-activation std of every filter, measured on the
// calibration set (the first 1 000 indices of the training split -- never validation or
// test images), is scaled to tau_K. tau_K is the mean, over seeds 1-5 and the 8 channels,
// of the pre-activation std of the `random`-mode conv with the same K and dataset, using
// exactly the conv weights a split-mode run with that seed draws first.
//
// Usage: kernel_dump [--data ..] [--out kernels] [--datasets mnist,cifar10] [--ks 3,5,7,9]
//                    [--kernel-seed 20261018] [--calib 1000] [--no-mmap]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "conv.h"
#include "dataset.h"
#include "kernels.h"
#include "rng_util.h"
#include "util.h"

static std::vector<std::string> split_list(const std::string& s)
{
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}

// Per-channel population std of the pre-activation (bias 0) over the calibration images.
static std::vector<double> channel_std(const std::vector<double>& W, int K, const Dataset& ds,
                                       const std::vector<uint32_t>& calib)
{
    const int C = N_BANK_FILTERS;
    Conv<double> conv(ds.channels(), ds.height(), ds.width(), C, K, 1, std::vector<Act>(C, ACT_LINEAR));
    conv.W = W;
    std::vector<double> s(C, 0.0), s2(C, 0.0), x(ds.input_size());
    const int HW = ds.height() * ds.width();
    for (uint32_t i : calib) {
        const float* in = ds.input(i);
        for (size_t k = 0; k < x.size(); ++k) x[k] = in[k];
        conv.forward(x.data());
        for (int c = 0; c < C; ++c)
            for (int k = 0; k < HW; ++k) { const double z = conv.z[(size_t)c * HW + k]; s[c] += z; s2[c] += z * z; }
    }
    const double n = (double)calib.size() * HW;
    std::vector<double> sd(C);
    for (int c = 0; c < C; ++c) { const double m = s[c] / n; sd[c] = std::sqrt(std::max(0.0, s2[c] / n - m * m)); }
    return sd;
}

int main(int argc, char** argv)
{
    std::string data = "..", out = "kernels", datasets = "mnist,cifar10", ks = "3,5,7,9";
    uint32_t kseed = DEFAULT_KERNEL_SEED;
    int n_calib = 1000;
    bool mmap = true;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(1); } return argv[++i]; };
        if (k == "--data") data = val();
        else if (k == "--out") out = val();
        else if (k == "--datasets") datasets = val();
        else if (k == "--ks") ks = val();
        else if (k == "--kernel-seed") kseed = (uint32_t)std::stoul(val());
        else if (k == "--calib") n_calib = std::stoi(val());
        else if (k == "--no-mmap") mmap = false;
        else { std::cerr << "unknown argument: " << k << "\n"; return 1; }
    }
    std::filesystem::create_directories(out);

    // 2-D filter CSVs (dataset independent).
    for (const std::string& ksz : split_list(ks)) {
        const int K = std::stoi(ksz);
        for (const char* fam : KERNEL_FAMILIES) {
            std::vector<double> bank = make_bank(fam, K, kseed);
            std::ostringstream s;
            s << "family,k,filter,row,col,value\n";
            char buf[48];
            for (int j = 0; j < N_BANK_FILTERS; ++j)
                for (int r = 0; r < K; ++r)
                    for (int c = 0; c < K; ++c) {
                        std::snprintf(buf, sizeof buf, "%.9g", (double)(float)bank[(size_t)j * K * K + r * K + c]);
                        s << fam << "," << K << "," << j << "," << r << "," << c << "," << buf << "\n";
                    }
            util::write_atomic((std::filesystem::path(out) / (std::string(fam) + "_k" + std::to_string(K) + ".csv")).string(), s.str());
        }
    }

    for (const std::string& dsname : split_list(datasets)) {
        auto ds = load_dataset(dsname, data, mmap);
        Split sp = make_split(dsname, *ds);
        std::vector<uint32_t> calib(sp.train.begin(), sp.train.begin() + n_calib);
        const int cin = ds->channels();

        for (const std::string& ksz : split_list(ks)) {
            const int K = std::stoi(ksz);
            // tau_K from the random-mode conv init of seeds 1-5 (split RNG, init stream).
            double tau = 0;
            for (uint32_t seed = 1; seed <= 5; ++seed) {
                std::mt19937 g(detu::stream_seed(seed, detu::STREAM_INIT));
                const float limit = std::sqrt(1.0f / (cin * K * K));
                std::vector<double> W((size_t)N_BANK_FILTERS * cin * K * K);
                for (auto& w : W) w = detu::uniform_sym(g, limit);
                for (double sd : channel_std(W, K, *ds, calib)) tau += sd;
            }
            tau /= 5.0 * N_BANK_FILTERS;
            std::printf("%s K=%d tau=%.6g\n", dsname.c_str(), K, tau);

            for (const char* fam : KERNEL_FAMILIES) {
                BankFile b;
                b.dataset = dsname; b.family = fam; b.k = K;
                b.kernel_seed = std::string(fam) == "random_frozen" ? kseed : 0;
                b.calib_images = n_calib;
                b.tau = tau;
                for (double x : make_bank(fam, K, kseed)) b.filters.push_back((float)x);
                std::vector<float> ones(N_BANK_FILTERS, 1.0f);
                std::vector<float> W1 = bank_weights(b.filters, ones, K, cin);
                b.measured_std = channel_std(std::vector<double>(W1.begin(), W1.end()), K, *ds, calib);
                for (double sd : b.measured_std) b.gains.push_back((float)(tau / sd));
                util::Fnv1a h;
                for (float x : b.filters) h.f32(x);
                b.filter_hash = h.hex();
                save_bank(bank_path(out, dsname, fam, K), b);
                std::printf("  %-14s gains:", fam);
                for (float gn : b.gains) std::printf(" %.4g", gn);
                std::printf("\n");
            }
        }
    }
    return 0;
}
