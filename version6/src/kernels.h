#ifndef KERNELS_H
#define KERNELS_H

#include <cstdint>
#include <string>
#include <vector>

// Fixed kernel banks for --kernel-mode bank_init / frozen (PROMPT.md section 7).
//
// Every bank has exactly 8 two-dimensional K x K filters (K odd, radius r = (K-1)/2,
// coordinates u (column) and v (row) in -r..r, base scale sigma_K = K/6):
//
//   gauss          isotropic Gaussians, sigma_j = sigma_K (0.5 + 0.25 j), j = 0..7
//   sharpen        delta + alpha (delta - G_sigma), alpha in {0.5,1,2,4} x sigma in
//                  {sigma_K, 1.5 sigma_K}; filter j = 2*alpha_index + sigma_index
//   sobel          generalised Sobel G_theta = cos(theta) Gx + sin(theta) Gy, theta_j = 2 pi j/8
//   laplacian      zero-sum LoG at sigma_K {0.5,0.75,1,1.25}; filters 0-3 on-centre
//                  (positive centre), 4-7 off-centre (same sigmas, negated)
//   gabor          theta in {0, pi/4, pi/2, 3pi/4} x psi in {0, pi/2}; filter j = 2*theta_index
//                  + psi_index; zero mean
//   random_frozen  i.i.d. N(0,1) from mt19937(kernel_seed) via detu::normal_pair
//
// Every filter is then L2-normalised. Construction is in double precision; the stored
// bank (kernels/<dataset>_<family>_k<K>.json) holds the filters rounded to float and the
// per-filter gains, and runs read that file instead of rebuilding, so libm differences
// between platforms can never change a run's kernels.

static const char* const KERNEL_FAMILIES[] = {"gauss", "sharpen", "sobel", "laplacian", "gabor", "random_frozen"};
static const int N_BANK_FILTERS = 8;
static const uint32_t DEFAULT_KERNEL_SEED = 20261018u;

// 8 * K * K values, filter-major then row-major, each filter L2-normalised.
std::vector<double> make_bank(const std::string& family, int K, uint32_t kernel_seed);

// Wavelength used by the gabor family. PROMPT.md specifies max(2, (K+1)/2); at K = 3 that
// gives lambda = 2 and the odd-phase (psi = pi/2) filters at theta = 0 and pi/2 are
// identically zero (sin(pi u) = 0 at integer u), so K = 3 uses 2.5 instead. See
// DECISIONS.md.
double gabor_lambda(int K);

struct BankFile
{
    std::string dataset, family;
    int k = 0;
    uint32_t kernel_seed = 0;
    int calib_images = 0;
    double tau = 0;                    // target pre-activation std
    std::vector<float> filters;        // 8 * K * K, L2-normalised (pre-gain)
    std::vector<float> gains;          // 8
    std::vector<double> measured_std;  // 8, std of each un-gained filter on the calibration set
    std::string filter_hash;           // FNV-1a over `filters`
};

std::string bank_path(const std::string& dir, const std::string& dataset, const std::string& family, int K);
void        save_bank(const std::string& path, const BankFile& b);
BankFile    load_bank(const std::string& path);

// Conv weights from a bank: W[c, ci, :, :] = f_c * (g_c / sqrt(cin)), computed in double
// and rounded once to float. cin = 1 (MNIST) or 3 (CIFAR-10: the same 2-D filter on every
// colour channel). Gains of 1 give the un-gained bank used for calibration.
std::vector<float> bank_weights(const std::vector<float>& filters, const std::vector<float>& gains, int K, int cin);

#endif
