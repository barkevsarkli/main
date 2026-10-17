#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "conv.h"
#include "net.h"
#include "util.h"

#define OUTPUT_SIZE 10

// Architectures (--arch):
//   mlp2  D -> H -> H -> 10       the version5 network
//   mlp1  D -> H -> 10            depth-matched baseline for the CNN
//   cnn   input -> Conv2D -> act (per-channel mask) -> MaxPool -> flatten
//               -> Dense H (per-neuron mask) -> Dense 10 (linear) -> softmax CE
struct ModelSpec
{
    std::string arch = "cnn";
    int in_c = 1, in_h = 28, in_w = 28;
    int hidden = 128;
    int conv_k = 5, conv_channels = 8, pool = 2;
    std::string kernel_mode = "random";          // random | bank_init | frozen
    std::vector<Act> conv_mask;                  // C entries (cnn only)
    std::vector<std::vector<Act>> hid_masks;     // one per hidden dense layer
    Act out_act = ACT_LINEAR;                    // linear (CE) or sigmoid (MSE)
    bool legacy = false;                         // version5 init and summation order
};

template <class T>
class Model
{
public:
    explicit Model(const ModelSpec& s) : spec(s)
    {
        const int D = s.in_c * s.in_h * s.in_w;
        int feat = D;
        if (s.arch == "cnn") {
            if ((int)s.conv_mask.size() != s.conv_channels) throw std::runtime_error("conv mask size");
            conv.reset(new Conv<T>(s.in_c, s.in_h, s.in_w, s.conv_channels, s.conv_k, s.pool, s.conv_mask));
            feat = conv->out_size();
            conv_trainable = (s.kernel_mode != "frozen");
        } else if (s.arch != "mlp1" && s.arch != "mlp2") {
            throw std::runtime_error("unknown arch: " + s.arch);
        }
        const size_t n_hidden = (s.arch == "mlp2") ? 2 : 1;
        if (s.hid_masks.size() != n_hidden) throw std::runtime_error("hidden mask count does not match arch");
        dense.reserve(n_hidden + 1);                 // Dense keeps pointers into its neighbours
        int fan = feat;
        for (size_t L = 0; L < n_hidden; ++L) {
            dense.emplace_back(fan, s.hidden, s.hid_masks[L]);
            fan = s.hidden;
        }
        dense.emplace_back(fan, OUTPUT_SIZE, std::vector<Act>(OUTPUT_SIZE, s.out_act));
        for (auto& d : dense) d.sequential_sum = s.legacy;
        if (!std::is_same<T, float>::value) xbuf.resize(D);
    }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Initial weights, all from one stream, in layer order: conv (if any) first, then the
    // dense layers. Every kernel mode draws the random conv weights so that the dense
    // layers get the same draws whatever happens to the conv layer afterwards.
    void init_weights(std::mt19937& rng)
    {
        if (conv) {
            const float limit = std::sqrt(1.0f / conv->fan_in());
            for (auto& w : conv->W) w = T(detu::uniform_sym(rng, limit));
        }
        for (auto& d : dense) {
            if (spec.legacy) d.init_legacy(rng);
            else d.init_split(rng);
        }
    }

    // Overwrite the conv weights with a fixed bank (bank_init / frozen); bias stays 0.
    void set_conv_weights(const std::vector<float>& W)
    {
        if (!conv || W.size() != conv->W.size()) throw std::runtime_error("set_conv_weights: shape mismatch");
        for (size_t i = 0; i < W.size(); ++i) conv->W[i] = T(W[i]);
        std::fill(conv->b.begin(), conv->b.end(), T(0));
    }

    const T* forward(const float* x)
    {
        const T* h;
        if (std::is_same<T, float>::value) h = reinterpret_cast<const T*>(x);
        else { for (size_t i = 0; i < xbuf.size(); ++i) xbuf[i] = T(x[i]); h = xbuf.data(); }
        if (conv) { conv->forward(h); h = conv->pooled.data(); }
        for (auto& d : dense) { d.forward(h); h = d.out.data(); }
        return h;
    }

    const T* logits() const { return dense.back().out.data(); }

    // One SGD step from dL/d(logits). Dense layers compute their input gradient with the
    // pre-update weights and update in the same pass (version5); the first trainable layer
    // skips the input gradient. A frozen conv layer is skipped entirely.
    void backward(const T* d_logits, T lr)
    {
        const bool train_conv = conv && conv_trainable;
        const T* g = d_logits;
        for (int L = (int)dense.size() - 1; L >= 0; --L) {
            const bool need_d_in = (L > 0) || train_conv;
            dense[L].backward(g, lr, need_d_in);
            g = dense[L].d_in.data();
        }
        if (train_conv) conv->backward(g, lr);
    }

    void set_clip(T c)
    {
        if (conv) conv->clip = c;
        for (auto& d : dense) d.clip = c;
    }

    // All parameter tensors in a fixed order: conv W, conv b, then W, b of each dense layer.
    std::vector<std::vector<T>*> params()
    {
        std::vector<std::vector<T>*> p;
        if (conv) { p.push_back(&conv->W); p.push_back(&conv->b); }
        for (auto& d : dense) { p.push_back(&d.W); p.push_back(&d.b); }
        return p;
    }

    void save_state(std::vector<std::vector<T>>& s)
    {
        auto p = params();
        s.resize(p.size());
        for (size_t i = 0; i < p.size(); ++i) s[i] = *p[i];
    }

    void load_state(const std::vector<std::vector<T>>& s)
    {
        auto p = params();
        for (size_t i = 0; i < p.size(); ++i) *p[i] = s[i];
    }

    size_t n_params_total()
    {
        size_t n = 0;
        for (auto* v : params()) n += v->size();
        return n;
    }

    size_t n_params_trainable()
    {
        size_t n = n_params_total();
        if (conv && !conv_trainable) n -= conv->W.size() + conv->b.size();
        return n;
    }

    // FNV-1a 64 over float32 little-endian values.
    std::string hash_all()   { util::Fnv1a h; for (auto* v : params()) h.floats(*v); return h.hex(); }
    std::string hash_dense() { util::Fnv1a h; for (auto& d : dense) { h.floats(d.W); h.floats(d.b); } return h.hex(); }
    std::string hash_conv()  { if (!conv) return "none"; util::Fnv1a h; h.floats(conv->W); h.floats(conv->b); return h.hex(); }

    ModelSpec spec;
    std::unique_ptr<Conv<T>> conv;
    std::vector<Dense<T>> dense;
    bool conv_trainable = false;

private:
    std::vector<T> xbuf;
};

#endif
