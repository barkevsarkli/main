// Production conv (src/conv.h) against the plain reference (tests/conv_reference.h).
//
//  (a) float production vs double reference, error relative to the sum of |terms| of
//      each output (the scale floating-point rounding acts on): <= 1e-6 per element;
//  (b) double production vs double reference with the same measure: <= 1e-12;
//  (c) the production forward/backward run twice gives bit-identical results.
// Covers both input shapes, K in {3,5,7,9}, P in {1,2,4}, mixed channel masks.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include "check.h"
#include "conv.h"
#include "conv_reference.h"
#include "rng_util.h"

template <class T>
static void run_case(int cin, int H, int K, int P, uint32_t seed, double tol, double& worst_f, double& worst_b)
{
    const int C = 8;
    std::vector<Act> acts = {ACT_RELU, ACT_TANH, ACT_LEAKY_RELU, ACT_SIGMOID, ACT_RELU, ACT_TANH, ACT_TANH, ACT_RELU};
    Conv<T> pc(cin, H, H, C, K, P, acts);
    ConvRef<double> rc(cin, H, H, C, K, P, acts);
    std::mt19937 g(seed);
    const float lim = std::sqrt(3.0f / (cin * K * K));
    for (size_t i = 0; i < pc.W.size(); ++i) { const float w = detu::uniform_sym(g, lim); pc.W[i] = T(w); rc.W[i] = w; }
    for (int c = 0; c < C; ++c) { const float bb = detu::uniform_sym(g, 0.1f); pc.b[c] = T(bb); rc.b[c] = bb; }
    std::vector<T> x((size_t)cin * H * H);
    std::vector<double> xd(x.size());
    for (size_t i = 0; i < x.size(); ++i) { const float v = detu::uniform01(g); x[i] = T(v); xd[i] = v; }
    std::vector<T> dp((size_t)C * (H / P) * (H / P));
    std::vector<double> dpd(dp.size());
    for (size_t i = 0; i < dp.size(); ++i) { const float v = detu::uniform_sym(g, 2.0f); dp[i] = T(v); dpd[i] = v; }

    pc.forward(x.data());
    rc.forward(xd.data());
    // Same summation order as the reference in the same precision: bit-identical forward.
    {
        ConvRef<T> rt(cin, H, H, C, K, P, acts);
        rt.W = pc.W; rt.b = pc.b;
        rt.forward(x.data());
        CHECK(std::memcmp(rt.z.data(), pc.z.data(), rt.z.size() * sizeof(T)) == 0,
              "forward not bit-identical to the same-order reference (cin=%d K=%d)", cin, K);
    }
    for (size_t i = 0; i < rc.z.size(); ++i) {
        const double e = std::fabs((double)pc.z[i] - rc.z[i]) / std::max(rc.abs_z[i], 1e-30);
        worst_f = std::max(worst_f, e);
    }
    // Pooled values and arg-max positions must agree exactly unless two candidates differ
    // by less than the rounding error (never with random continuous inputs).
    int arg_mismatch = 0;
    for (size_t q = 0; q < rc.argmax.size(); ++q) arg_mismatch += (pc.argmax[q] != rc.argmax[q]);
    CHECK(arg_mismatch == 0, "cin=%d K=%d P=%d: %d arg-max mismatches", cin, K, P, arg_mismatch);

    std::vector<T> W0 = pc.W, b0 = pc.b;
    const T lr = T(0.5);
    pc.backward(dp.data(), lr);
    rc.backward(xd.data(), dpd.data());
    for (size_t k = 0; k < rc.dW.size(); ++k) {
        // production applied W -= lr * dW; recover its dW from its own buffer
        const double e = std::fabs((double)pc.dW[k] - rc.dW[k]) / std::max(rc.abs_dW[k], 1e-30);
        worst_b = std::max(worst_b, e);
        CHECK(pc.W[k] == W0[k] - lr * pc.dW[k], "update rule");
    }
    for (int c = 0; c < C; ++c) {
        const double e = std::fabs((double)pc.db[c] - rc.db[c]) / std::max(rc.abs_db[c], 1e-30);
        worst_b = std::max(worst_b, e);
    }
    CHECK(worst_f <= tol, "forward rel err %.3g > %.3g (cin=%d K=%d P=%d)", worst_f, tol, cin, K, P);
    CHECK(worst_b <= tol, "backward rel err %.3g > %.3g (cin=%d K=%d P=%d)", worst_b, tol, cin, K, P);

    // Bit reproducibility of the production path.
    Conv<T> pc2(cin, H, H, C, K, P, acts);
    pc2.W = W0; pc2.b = b0;
    pc2.forward(x.data());
    pc2.backward(dp.data(), lr);
    CHECK(std::memcmp(pc2.W.data(), pc.W.data(), pc.W.size() * sizeof(T)) == 0, "not bit-reproducible");
}

int main()
{
    double wf32 = 0, wb32 = 0, wf64 = 0, wb64 = 0;
    uint32_t seed = 1;
    for (int shape = 0; shape < 2; ++shape) {
        const int cin = shape ? 3 : 1, H = shape ? 32 : 28;
        for (int K : {3, 5, 7, 9})
            for (int P : {1, 2, 4}) {
                run_case<float>(cin, H, K, P, seed, 1e-6, wf32, wb32);
                run_case<double>(cin, H, K, P, seed, 1e-12, wf64, wb64);
                ++seed;
            }
    }
    std::printf("worst relative error (vs sum|terms|): float fwd %.3g bwd %.3g | double fwd %.3g bwd %.3g\n",
                wf32, wb32, wf64, wb64);
    return finish("test_conv_reference");
}
