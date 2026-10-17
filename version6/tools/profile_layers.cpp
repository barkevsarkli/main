// profile_layers -- seconds per epoch spent in each layer of the production trainer path.
//
// Times every layer's forward and backward separately with steady_clock around the same
// calls the trainer makes (Model::forward / backward unrolled), on N training samples, and
// scales to one epoch: 45 000 training steps + 15 000 evaluation forwards (val 5k + clean
// training 10k; the test set is evaluated only once per run).
//
// Usage: profile_layers [--dataset cifar10] [--hidden 128] [--k 9] [--mode random]
//                       [--samples 3000] [--data ..] [--act1 tanh --act2 relu --ratio 0.5]

#include <chrono>
#include <cstdio>
#include <string>
#include "dataset.h"
#include "kernels.h"
#include "model.h"
#include "train_util.h"

typedef std::chrono::steady_clock Clock;
static double sec(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv)
{
    std::string ds_name = "cifar10", data = "..", mode = "random", a1 = "tanh", a2 = "relu", arch = "cnn";
    int H = 128, K = 9, N = 3000;
    float ratio = 0.5f;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--dataset") ds_name = v; else if (k == "--hidden") H = std::stoi(v);
        else if (k == "--k") K = std::stoi(v); else if (k == "--mode") mode = v;
        else if (k == "--samples") N = std::stoi(v); else if (k == "--data") data = v;
        else if (k == "--act1") a1 = v; else if (k == "--act2") a2 = v; else if (k == "--ratio") ratio = std::stof(v);
        else if (k == "--arch") arch = v;
        else { std::fprintf(stderr, "unknown %s\n", k.c_str()); return 1; }
    }
    auto ds = load_dataset(ds_name, data, true);
    Split sp = make_split(ds_name, *ds);
    std::mt19937 mr(1);
    ModelSpec s;
    s.arch = arch; s.in_c = ds->channels(); s.in_h = ds->height(); s.in_w = ds->width();
    s.hidden = H; s.conv_k = K; s.conv_channels = 8; s.pool = 2; s.kernel_mode = mode;
    s.conv_mask = make_mask(8, act_from_string(a1), act_from_string(a2), ratio, "interleave", mr);
    s.hid_masks.assign(arch == "mlp2" ? 2 : 1, make_mask(H, act_from_string(a1), act_from_string(a2), ratio, "interleave", mr));
    Model<float> m(s);
    std::mt19937 g(detu::stream_seed(1, detu::STREAM_INIT));
    m.init_weights(g);
    const float lr = 0.003f;
    const size_t L = m.dense.size();
    std::vector<double> t_cf(1, 0), t_cb(1, 0), t_df(L, 0), t_db(L, 0);
    double t_loss = 0, t_load = 0;
    float d[OUTPUT_SIZE];
    auto t_all0 = Clock::now();
    for (int n = 0; n < N; ++n) {
        const uint32_t i = sp.train[n % sp.train.size()];
        auto t0 = Clock::now();
        const float* x = ds->input(i);
        auto t1 = Clock::now(); t_load += sec(t0, t1);
        const float* h = x;
        if (m.conv) { auto a = Clock::now(); m.conv->forward(h); h = m.conv->pooled.data(); t_cf[0] += sec(a, Clock::now()); }
        for (size_t l = 0; l < L; ++l) { auto a = Clock::now(); m.dense[l].forward(h); h = m.dense[l].out.data(); t_df[l] += sec(a, Clock::now()); }
        auto a = Clock::now();
        sample_loss(h, ds->label(i), true, d);
        t_loss += sec(a, Clock::now());
        const bool train_conv = m.conv && m.conv_trainable;
        const float* gr = d;
        for (int l = (int)L - 1; l >= 0; --l) {
            auto b = Clock::now();
            m.dense[l].backward(gr, lr, l > 0 || train_conv);
            gr = m.dense[l].d_in.data();
            t_db[l] += sec(b, Clock::now());
        }
        if (train_conv) { auto b = Clock::now(); m.conv->backward(gr, lr); t_cb[0] += sec(b, Clock::now()); }
    }
    const double total = sec(t_all0, Clock::now());
    // epoch = 45k training steps + 15k evaluation forwards
    const double tr = 45000.0 / N, ev = 15000.0 / N;
    std::printf("profile %s %s H=%d K=%d mode=%s %s+%s r=%.2f, %d samples (scaled to one epoch: 45k steps + 15k eval forwards)\n",
                arch.c_str(), ds_name.c_str(), H, K, mode.c_str(), a1.c_str(), a2.c_str(), ratio, N);
    double sum = 0;
    auto row = [&](const char* name, double fwd, double bwd) {
        const double e = fwd * (tr + ev) + bwd * tr;
        sum += e;
        std::printf("  %-10s forward %7.2f s  backward %7.2f s  -> %7.2f s/epoch\n", name, fwd * (tr + ev), bwd * tr, e);
    };
    if (m.conv) row("conv", t_cf[0], t_cb[0]);
    for (size_t l = 0; l < L; ++l) row(l + 1 == L ? "output" : ("hidden" + std::to_string(l + 1)).c_str(), t_df[l], t_db[l]);
    row("loss+load", t_loss + t_load, 0);
    std::printf("  total %.2f s/epoch (measured loop %.2f s for %d samples)\n", sum, total, N);
    return 0;
}
