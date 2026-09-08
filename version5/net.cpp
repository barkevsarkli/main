#include "net.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Act act_from_string(const std::string& s)
{
    if (s == "relu")       return ACT_RELU;
    if (s == "leaky_relu") return ACT_LEAKY_RELU;
    if (s == "tanh")       return ACT_TANH;
    if (s == "sigmoid")    return ACT_SIGMOID;
    if (s == "linear")     return ACT_LINEAR;
    throw std::runtime_error("unknown activation: " + s);
}

std::string act_to_string(Act a)
{
    switch (a) {
        case ACT_RELU:       return "relu";
        case ACT_LEAKY_RELU: return "leaky_relu";
        case ACT_TANH:       return "tanh";
        case ACT_SIGMOID:    return "sigmoid";
        default:             return "linear";
    }
}

static inline float activate(Act a, float z)
{
    switch (a) {
        case ACT_RELU:       return z > 0.0f ? z : 0.0f;
        case ACT_LEAKY_RELU: return z > 0.0f ? z : LEAKY_SLOPE * z;
        case ACT_TANH:       return std::tanh(z);
        case ACT_SIGMOID:
            if (z >= 0.0f) { float e = std::exp(-z); return 1.0f / (1.0f + e); }
            else           { float e = std::exp(z);  return e / (1.0f + e); }
        default:             return z;
    }
}

// Derivative expressed in terms of the *output* y (as in v2), which is exact
// for all functions used here.
static inline float activate_derivative(Act a, float y)
{
    switch (a) {
        case ACT_RELU:       return y > 0.0f ? 1.0f : 0.0f;
        case ACT_LEAKY_RELU: return y > 0.0f ? 1.0f : LEAKY_SLOPE;
        case ACT_TANH:       return 1.0f - y * y;
        case ACT_SIGMOID:    return y * (1.0f - y);
        default:             return 1.0f;
    }
}

Layer::Layer(int n_in_, int n_out_, const std::vector<Act>& acts, std::mt19937& rng)
    : n_in(n_in_), n_out(n_out_),
      W((size_t)n_in_ * n_out_), b(n_out_, 0.0f), out(n_out_, 0.0f), d_in(n_in_, 0.0f),
      act(acts), in(nullptr)
{
    if ((int)act.size() != n_out)
        throw std::runtime_error("Layer: activation mask size does not match layer size");

    // Kaiming-uniform with bound sqrt(1/fan_in) -- PyTorch's nn.Linear default.
    float limit = std::sqrt(1.0f / n_in);
    std::uniform_real_distribution<float> U(-limit, limit);
    for (auto& w : W) w = U(rng);
}

void Layer::forward(const float* inputs)
{
    in = inputs;
    for (int i = 0; i < n_out; ++i) {
        const float* w = &W[(size_t)i * n_in];
        float z = b[i];
        for (int j = 0; j < n_in; ++j) z += inputs[j] * w[j];
        out[i] = activate(act[i], z);
    }
}

void Layer::backward(const float* d_values, float lr, bool compute_d_inputs)
{
    if (compute_d_inputs) std::fill(d_in.begin(), d_in.end(), 0.0f);

    for (int i = 0; i < n_out; ++i) {
        float d = d_values[i] * activate_derivative(act[i], out[i]);
        if (d >  GRAD_CLIP) d =  GRAD_CLIP;
        if (d < -GRAD_CLIP) d = -GRAD_CLIP;

        float* w = &W[(size_t)i * n_in];
        const float step = lr * d;

        if (compute_d_inputs) {
            for (int j = 0; j < n_in; ++j) {
                d_in[j] += w[j] * d;        // uses pre-update weight
                w[j]    -= step * in[j];
            }
        } else {
            for (int j = 0; j < n_in; ++j) w[j] -= step * in[j];
        }
        b[i] -= step;
    }
}

std::vector<Act> make_mask(int n, Act act1, Act act2, float ratio,
                           const std::string& layout, std::mt19937& rng)
{
    if (ratio < 0.0f || ratio > 1.0f) throw std::runtime_error("ratio must be in [0,1]");
    std::vector<Act> m(n, act1);
    int n2 = (int)std::lround(ratio * n);   // number of act2 neurons

    if (layout == "interleave") {
        // Spread act2 as evenly as possible.  For ratio 0.5 this is exactly
        // the v2 rule: odd indices get act2.
        int placed = 0;
        for (int i = 0; i < n && placed < n2; ++i) {
            int should = (int)std::floor((i + 1) * ratio + 1e-6f);
            if (should > placed) { m[i] = act2; ++placed; }
        }
    } else if (layout == "block") {
        for (int i = n - n2; i < n; ++i) m[i] = act2;
    } else if (layout == "random") {
        for (int i = n - n2; i < n; ++i) m[i] = act2;
        std::shuffle(m.begin(), m.end(), rng);
    } else {
        throw std::runtime_error("unknown layout: " + layout);
    }
    return m;
}
