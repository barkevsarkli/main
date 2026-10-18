// Kernel bank checks: shapes, unit L2 norm, zero sum (sobel, laplacian, gabor), K = 3 Sobel
// equals the classic kernel up to sign/scale, Gaussians non-negative, stored banks load
// and match make_bank, gains finite and positive, gallery PNGs exist.
//
// Usage: test_kernels <kernels dir>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include "check.h"
#include "kernels.h"

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "kernels";
    for (int K : {3, 5, 7, 9})
        for (const char* fam : KERNEL_FAMILIES) {
            const std::string f(fam);
            std::vector<double> b = make_bank(f, K, DEFAULT_KERNEL_SEED);
            CHECK((int)b.size() == 8 * K * K, "%s k%d size", fam, K);
            for (int j = 0; j < 8; ++j) {
                double s = 0, n2 = 0, mn = 1e9;
                for (int i = 0; i < K * K; ++i) { const double v = b[(size_t)j * K * K + i]; s += v; n2 += v * v; mn = std::min(mn, v); }
                CHECK(std::fabs(n2 - 1) < 1e-12, "%s k%d filter %d not unit norm", fam, K, j);
                if (f == "sobel" || f == "laplacian" || f == "gabor")
                    CHECK(std::fabs(s) < 1e-12, "%s k%d filter %d sum %.3g", fam, K, j, s);
                if (f == "gauss") CHECK(mn >= 0, "gauss k%d filter %d negative", K, j);
            }
            // Distinct filters (no duplicates within a bank).
            for (int j = 0; j < 8; ++j)
                for (int l = j + 1; l < 8; ++l) {
                    double d = 0;
                    for (int i = 0; i < K * K; ++i) d += std::fabs(b[(size_t)j * K * K + i] - b[(size_t)l * K * K + i]);
                    CHECK(d > 1e-6, "%s k%d filters %d and %d identical", fam, K, j, l);
                }
            for (const char* ds : {"mnist", "cifar10"}) {
                const std::string p = bank_path(dir, ds, f, K);
                if (!std::filesystem::exists(p)) { CHECK(false, "missing %s", p.c_str()); continue; }
                BankFile bf = load_bank(p);
                CHECK(bf.k == K && bf.family == f && bf.dataset == ds, "%s header", p.c_str());
                double md = 0;
                for (size_t i = 0; i < b.size(); ++i) md = std::max(md, std::fabs((double)bf.filters[i] - b[i]));
                CHECK(md < 1e-7, "%s filters differ from make_bank by %.3g", p.c_str(), md);
                for (float g : bf.gains) CHECK(std::isfinite(g) && g > 0, "%s gain %g", p.c_str(), g);
                CHECK(bf.tau > 0 && std::isfinite(bf.tau), "%s tau", p.c_str());
            }
        }
    // Classic Sobel at K = 3, theta = 0: [[-1,0,1],[-2,0,2],[-1,0,1]] up to sign and scale.
    {
        std::vector<double> b = make_bank("sobel", 3, 0);
        const double classic[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        const double s = b[0] / classic[0];
        double md = 0;
        for (int i = 0; i < 9; ++i) md = std::max(md, std::fabs(b[i] - s * classic[i]));
        CHECK(md < 1e-12, "K=3 Sobel differs from the classic kernel (%.3g)", md);
    }
    for (const char* ds : {"mnist", "cifar10"}) {
        const auto p = std::filesystem::path(dir) / (std::string("gallery_") + ds + ".png");
        CHECK(std::filesystem::exists(p), "missing %s", p.string().c_str());
    }
    return finish("test_kernels");
}
