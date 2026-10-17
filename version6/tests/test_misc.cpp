// Small unit tests:
//  * --abort-on-collapse logic on synthetic loss/accuracy sequences (triggers, does not
//    trigger, resets on a good epoch);
//  * dead-unit and saturation counting of the diagnostics on a hand-built network where
//    the answer is known;
//  * RNG alignment inside the model: conv kernel mode and activation masks do not change
//    the dense initial weights for a given init stream.

#include <cmath>
#include <map>
#include <sstream>
#include "check.h"
#include "kernels.h"
#include "model.h"
#include "stats.h"
#include "train_util.h"

// Four fixed 1x28x28 "images" with known content.
struct FakeData : Dataset
{
    std::vector<std::vector<float>> img;
    FakeData()
    {
        for (int i = 0; i < 4; ++i) {
            std::vector<float> x(784);
            for (int k = 0; k < 784; ++k) x[k] = (float)((k * 7 + i * 13) % 17) / 16.0f;   // in [0, 1]
            img.push_back(x);
        }
    }
    uint32_t size() const override { return 4; }
    int input_size() const override { return 784; }
    int channels() const override { return 1; }
    int height() const override { return 28; }
    int width() const override { return 28; }
    const float* input(uint32_t i) const override { return img[i].data(); }
    uint8_t label(uint32_t i) const override { return (uint8_t)i; }
};

static std::map<std::string, std::string> parse_stats(const std::string& text, int epoch)
{
    std::map<std::string, std::string> m;
    std::istringstream s(text);
    std::string line;
    while (std::getline(s, line)) {
        std::vector<std::string> f;
        std::stringstream ls(line);
        std::string t;
        while (std::getline(ls, t, ',')) f.push_back(t);
        if (f.size() == 6 && f[1] == std::to_string(epoch)) m[f[2] + "/" + f[3] + "/" + f[4]] = f[5];
    }
    return m;
}

int main()
{
    // ---- collapse watch ----
    {
        const double ln10 = std::log(10.0);
        CollapseWatch off(0);
        bool fired = false;
        for (int e = 0; e < 10; ++e) fired = fired || off.update(ln10, 10.0);
        CHECK(!fired, "N=0 must never fire");

        CollapseWatch w(3);
        CHECK(!w.update(ln10 + 0.005, 10.0), "epoch 1");
        CHECK(!w.update(ln10 - 0.009, 11.0), "epoch 2");
        CHECK(w.update(ln10, 9.8), "third consecutive collapsed epoch must fire");

        CollapseWatch r(3);
        r.update(ln10, 10.0); r.update(ln10, 10.0);
        CHECK(!r.update(ln10, 50.0), "accuracy above 15%% resets");
        CHECK(!r.update(ln10, 10.0) && !r.update(ln10, 10.0), "count restarts");
        CHECK(r.update(ln10, 10.0), "fires after three again");

        CollapseWatch l(2);
        CHECK(!l.update(ln10 + 0.02, 10.0) && !l.update(ln10 + 0.02, 10.0), "loss 0.02 above ln 10 is not collapse");
        CHECK(!l.update(ln10, 15.0) && !l.update(ln10, 15.0), "15.0%% is not below 15%%");
    }

    // ---- dead / saturated unit counting ----
    {
        ModelSpec s;
        s.arch = "mlp1"; s.hidden = 4;
        s.hid_masks = {{ACT_RELU, ACT_RELU, ACT_TANH, ACT_TANH}};
        Model<float> m(s);
        std::mt19937 g(1);
        m.init_weights(g);
        auto& H = m.dense[0];
        // Inputs are in [0, 1] with positive sum, so:
        //   unit 0: W = -1, b = -1   -> z < 0 always       -> dead ReLU
        //   unit 1: W = +1, b = +1   -> z > 0 always       -> alive
        //   unit 2: W = +1, b = 0    -> z ~ 400            -> tanh saturated on every image
        //   unit 3: W = 0,  b = 0.1  -> h = tanh(0.1)       -> never saturated
        for (int j = 0; j < 784; ++j) { H.W[j] = -1; H.W[784 + j] = 1; H.W[2 * 784 + j] = 1; H.W[3 * 784 + j] = 0; }
        H.b = {-1, 1, 0, 0.1f};
        FakeData ds;
        StatsCollector st("t", {0, 1, 2, 3});
        st.attach(m);
        st.record_probe(0, m, ds);
        auto v = parse_stats(st.text(), 0);
        CHECK(v["hidden1/relu/dead_units"] == "1", "dead_units = %s", v["hidden1/relu/dead_units"].c_str());
        CHECK(v["hidden1/relu/nearly_dead_units"] == "1", "nearly_dead = %s", v["hidden1/relu/nearly_dead_units"].c_str());
        CHECK(v["hidden1/relu/frac_neg"] == "0.5", "frac_neg = %s", v["hidden1/relu/frac_neg"].c_str());
        CHECK(v["hidden1/tanh/sat_units"] == "1", "sat_units = %s", v["hidden1/tanh/sat_units"].c_str());
        CHECK(v["hidden1/tanh/frac_sat"] == "0.5", "frac_sat = %s", v["hidden1/tanh/frac_sat"].c_str());
        CHECK(v.count("hidden1/all/z_mean") == 1, "mixed layer has an `all` group");
        CHECK(v["output/linear/n_pred_classes"] != "", "output metrics present");

        // Conv channels: a channel with W = -1, b = -1 is dead on every position and image.
        ModelSpec c;
        c.arch = "cnn"; c.hidden = 3; c.conv_k = 3; c.conv_channels = 2; c.pool = 2;
        c.conv_mask = {ACT_RELU, ACT_RELU};
        c.hid_masks = {{ACT_RELU, ACT_RELU, ACT_RELU}};
        Model<float> mc(c);
        mc.init_weights(g);
        for (int k = 0; k < 9; ++k) { mc.conv->W[k] = -1; mc.conv->W[9 + k] = 1; }
        mc.conv->b = {-1, 1};
        StatsCollector sc("t", {0, 1, 2, 3});
        sc.attach(mc);
        sc.record_probe(0, mc, ds);
        auto vc = parse_stats(sc.text(), 0);
        CHECK(vc["conv/relu/dead_units"] == "1", "conv dead_units = %s", vc["conv/relu/dead_units"].c_str());
    }

    // ---- RNG alignment inside the model ----
    {
        std::string ref;
        for (const char* mode : {"random", "bank_init", "frozen"})
            for (int mix = 0; mix < 3; ++mix) {
                ModelSpec s;
                s.arch = "cnn"; s.hidden = 16; s.conv_k = 5; s.conv_channels = 8; s.pool = 2; s.kernel_mode = mode;
                const Act a = mix == 0 ? ACT_RELU : ACT_TANH, b = mix == 2 ? ACT_SIGMOID : ACT_RELU;
                s.conv_mask = {a, b, a, b, a, b, a, b};
                s.hid_masks = {std::vector<Act>(16, b)};
                Model<float> m(s);
                std::mt19937 g(detu::stream_seed(42, detu::STREAM_INIT));
                m.init_weights(g);
                if (std::string(mode) != "random")
                    m.set_conv_weights(bank_weights(std::vector<float>(8 * 25, 0.2f), std::vector<float>(8, 1.0f), 5, 1));
                const std::string h = m.hash_dense();
                if (ref.empty()) ref = h;
                CHECK(h == ref, "dense init differs for mode %s mix %d", mode, mix);
            }
    }
    return finish("test_misc");
}
