// Finite-difference gradient check of the production code path in double precision
// (Model<double>, the same templates training runs in float), clip disabled.
//
// Analytic gradient: one backward() with lr = 1 from state S0 gives S1 = S0 - grad.
// Numerical gradient: central differences (L(theta + e) - L(theta - e)) / 2e, e = 1e-6.
// A parameter passes if |num - ana| <= 1e-4 * max(|num|, |ana|) + 1e-9. The absolute
// floor is the round-off of the central difference itself (machine eps * |L| / e is about
// 5e-10 for L near ln 10), which dominates for the many gradients of order 1e-7.
// Also checks that the update is exactly -lr * gradient for lr = 0.01 (relative 1e-9).
//
// Coverage: cnn on 1x28x28 and 3x32x32, K in {3, 9}, P in {1, 2, 4}, every activation in
// the conv and hidden layers, mixed masks in both layers; mlp1 and mlp2 likewise; CE and MSE.

#include <cmath>
#include <random>
#include "check.h"
#include "model.h"
#include "train_util.h"

static double g_worst = 0;
static long g_params = 0;

static double loss_of(Model<double>& m, const std::vector<float>& x, int label, bool ce)
{
    const double* o = m.forward(x.data());
    // version5's MSE back-propagates d = 2 (o - y), the gradient of the *sum* of squared
    // errors, while it logs their mean; differentiate the sum so the two agree.
    return sample_loss(o, label, ce, (double*)nullptr) * (ce ? 1.0 : OUTPUT_SIZE);
}

static void check_model(const ModelSpec& spec, bool ce, uint32_t seed, const char* name)
{
    Model<double> m(spec);
    std::mt19937 g(seed);
    m.init_weights(g);
    // Non-zero biases so that every bias gradient path is exercised away from symmetry.
    for (auto* p : m.params())
        if (p->size() <= 128) for (auto& v : *p) v = detu::uniform_sym(g, 0.05f);
    m.set_clip(INFINITY);

    const int D = spec.in_c * spec.in_h * spec.in_w;
    std::vector<float> x(D);
    for (auto& v : x) v = detu::uniform01(g);
    const int label = (int)(seed % 10);

    std::vector<std::vector<double>> S0, S1, S2;
    m.save_state(S0);
    double d[OUTPUT_SIZE];
    const double* o = m.forward(x.data());
    sample_loss(o, label, ce, d);
    m.backward(d, 1.0);
    m.save_state(S1);

    m.load_state(S0);
    o = m.forward(x.data());
    sample_loss(o, label, ce, d);
    m.backward(d, 0.01);
    m.save_state(S2);

    const double eps = 1e-6;
    for (size_t t = 0; t < S0.size(); ++t) {
        const bool frozen_conv = m.conv && !m.conv_trainable && t < 2;
        const size_t n = S0[t].size();
        const size_t n_check = std::min<size_t>(n, 24);
        for (size_t j = 0; j < n_check; ++j) {
            const size_t k = (n <= 24) ? j : (size_t)(((uint64_t)j * 2654435761u + seed) % n);
            const double ana = S0[t][k] - S1[t][k];
            if (frozen_conv) { CHECK(ana == 0.0, "%s: frozen conv changed", name); continue; }
            m.load_state(S0);
            auto p = m.params();
            (*p[t])[k] = S0[t][k] + eps;
            const double lp = loss_of(m, x, label, ce);
            (*p[t])[k] = S0[t][k] - eps;
            const double lm = loss_of(m, x, label, ce);
            const double num = (lp - lm) / (2 * eps);
            const double scale = std::max(std::fabs(num), std::fabs(ana));
            const double rel = std::fabs(num - ana) / (scale + 1e-5);   // reported; ~relative for |g| >> 1e-5
            g_worst = std::max(g_worst, rel);
            ++g_params;
            CHECK(std::fabs(num - ana) <= 1e-4 * scale + 1e-9, "%s: tensor %zu index %zu numeric %.10g analytic %.10g rel %.3g", name, t, k, num, ana, rel);
            const double upd = (S0[t][k] - S2[t][k]) / 0.01;
            CHECK(std::fabs(upd - ana) <= 1e-9 * std::max(1e-12, std::fabs(ana)) + 1e-15,
                  "%s: update != -lr * grad (%.12g vs %.12g)", name, upd, ana);
        }
    }
}

int main()
{
    const Act all[4] = {ACT_RELU, ACT_LEAKY_RELU, ACT_TANH, ACT_SIGMOID};
    uint32_t seed = 11;
    // CNN: every (K, P, input shape) with mixed masks built from each activation pair.
    for (int shape = 0; shape < 2; ++shape)
        for (int K : {3, 9})
            for (int P : {1, 2, 4})
                for (int ai = 0; ai < 4; ++ai) {
                    ModelSpec s;
                    s.arch = "cnn";
                    s.in_c = shape ? 3 : 1; s.in_h = s.in_w = shape ? 32 : 28;
                    s.hidden = 6; s.conv_k = K; s.conv_channels = 4; s.pool = P;
                    const Act a1 = all[ai], a2 = all[(ai + 1) % 4];
                    s.conv_mask = {a1, a2, a1, a2};
                    s.hid_masks = {{a2, a1, a2, a1, a2, a1}};
                    s.kernel_mode = "random";
                    char name[96];
                    std::snprintf(name, sizeof name, "cnn %dx%d K=%d P=%d %s+%s", s.in_c, s.in_h, K, P,
                                  act_to_string(a1).c_str(), act_to_string(a2).c_str());
                    check_model(s, true, seed++, name);
                }
    // Frozen conv: no conv gradient, dense gradients still exact.
    {
        ModelSpec s;
        s.arch = "cnn"; s.hidden = 6; s.conv_k = 5; s.conv_channels = 4; s.pool = 2;
        s.conv_mask = {ACT_TANH, ACT_RELU, ACT_TANH, ACT_RELU};
        s.hid_masks = {{ACT_TANH, ACT_RELU, ACT_TANH, ACT_RELU, ACT_TANH, ACT_RELU}};
        s.kernel_mode = "frozen";
        check_model(s, true, seed++, "cnn frozen");
    }
    // MLPs, homogeneous and mixed, CE and MSE.
    for (const char* arch : {"mlp1", "mlp2"})
        for (int ai = 0; ai < 4; ++ai)
            for (bool ce : {true, false}) {
                ModelSpec s;
                s.arch = arch; s.hidden = 7;
                s.out_act = ce ? ACT_LINEAR : ACT_SIGMOID;
                std::vector<Act> mk(7);
                for (int i = 0; i < 7; ++i) mk[i] = (i % 2) ? all[(ai + 2) % 4] : all[ai];
                s.hid_masks.assign(std::string(arch) == "mlp2" ? 2 : 1, mk);
                char name[64];
                std::snprintf(name, sizeof name, "%s %s %s", arch, act_to_string(all[ai]).c_str(), ce ? "ce" : "mse");
                check_model(s, ce, seed++, name);
            }
    std::printf("checked %ld parameters, worst |num-ana| / (max(|num|,|ana|) + 1e-5) = %.3g\n", g_params, g_worst);
    return finish("test_gradcheck");
}
