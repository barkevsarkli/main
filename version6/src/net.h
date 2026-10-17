#ifndef NET_H
#define NET_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "rng_util.h"

// Per-neuron activation functions.  Every neuron in a layer carries its own
// Act value, so a layer can mix functions in any pattern (even/odd, blocks,
// random masks, or anything an assignment method produces later).
enum Act { ACT_RELU = 0, ACT_LEAKY_RELU = 1, ACT_TANH = 2, ACT_SIGMOID = 3, ACT_LINEAR = 4 };

Act         act_from_string(const std::string& s);
std::string act_to_string(Act a);

// Slope used by leaky ReLU on the negative side, in forward and backward (version5).
#define LEAKY_SLOPE 0.1f

// Element-wise clip on d(loss)/d(pre-activation), kept from v2 as a safety
// net against exploding gradients.  version6 applies it at every pre-activation,
// including every conv output position.
#define GRAD_CLIP 10.0f

template <class T> inline T activate(Act a, T z)
{
    switch (a) {
        case ACT_RELU:       return z > T(0) ? z : T(0);
        case ACT_LEAKY_RELU: return z > T(0) ? z : T(LEAKY_SLOPE) * z;
        case ACT_TANH:       return std::tanh(z);
        case ACT_SIGMOID:
            if (z >= T(0)) { T e = std::exp(-z); return T(1) / (T(1) + e); }
            else           { T e = std::exp(z);  return e / (T(1) + e); }
        default:             return z;
    }
}

// Derivative expressed in terms of the *output* y (as in v2/v5), which is exact
// for all functions used here.
template <class T> inline T activate_derivative(Act a, T y)
{
    switch (a) {
        case ACT_RELU:       return y > T(0) ? T(1) : T(0);
        case ACT_LEAKY_RELU: return y > T(0) ? T(1) : T(LEAKY_SLOPE);
        case ACT_TANH:       return T(1) - y * y;
        case ACT_SIGMOID:    return y * (T(1) - y);
        default:             return T(1);
    }
}

// Training-time gradient statistics for one layer, accumulated during the epoch's SGD
// steps when diagnostics are on (a null pointer otherwise). Pure reads of values the
// backward pass computes anyway, so they cannot change training.
struct GradStats
{
    double   sum_abs = 0;     // sum of |dL/dz| before the clip
    double   max_abs = 0;
    uint64_t n = 0;           // number of dL/dz elements seen
    uint64_t clipped = 0;     // elements with |dL/dz| > clip
    void reset() { *this = GradStats(); }
    inline void add(double g, double clip)
    {
        const double a = std::fabs(g);
        sum_abs += a;
        if (a > max_abs) max_abs = a;
        if (a > clip) ++clipped;
        ++n;
    }
};

// Build a per-neuron mask for a layer of size n.
//   ratio  = fraction of neurons that receive act2 (0 -> all act1, 1 -> all act2)
//   layout = "interleave" (act2 spread evenly; ratio 0.5 == v2's even/odd),
//            "block"      (act1 first, act2 last),
//            "random"     (seeded shuffle of the block mask; the only layout that draws)
std::vector<Act> make_mask(int n, Act act1, Act act2, float ratio,
                           const std::string& layout, std::mt19937& rng);

// --conv-mask-offset 1: rotate the mask by one position, m'[i] = m[(i + 1) % n]. At
// ratio 0.5 with the interleave layout this is exactly the complementary pattern (the
// channels that had act1 get act2 and vice versa); at other ratios it keeps the counts.
std::vector<Act> rotate_mask(const std::vector<Act>& m);

// A fully connected layer. Batch-size-1 SGD as in version5: backward computes d_inputs
// with the *current* weights, then applies the update in the same pass.
//
// Templated on the scalar type so that tests/test_gradcheck.cpp can run the very same
// code in double precision; training uses T = float.
template <class T>
class Dense
{
public:
    Dense(int n_in_, int n_out_, const std::vector<Act>& acts)
        : n_in(n_in_), n_out(n_out_), W((size_t)n_in_ * n_out_), b(n_out_, T(0)),
          z(n_out_, T(0)), out(n_out_, T(0)), d_in(n_in_, T(0)), act(acts)
    {
        if ((int)act.size() != n_out)
            throw std::runtime_error("Dense: activation mask size does not match layer size");
    }

    // version5 init: U(-1/sqrt(fan_in), +1/sqrt(fan_in)) via std::uniform_real_distribution.
    // Kept only for --rng-mode legacy, where it must reproduce version5's stored results.
    void init_legacy(std::mt19937& rng)
    {
        float limit = std::sqrt(1.0f / n_in);
        std::uniform_real_distribution<float> U(-limit, limit);
        for (auto& w : W) w = T(U(rng));
    }

    // Same distribution from raw 32-bit draws (detu::uniform_sym), identical on every
    // standard library. Biases stay 0.
    void init_split(std::mt19937& rng)
    {
        float limit = std::sqrt(1.0f / n_in);
        for (auto& w : W) w = T(detu::uniform_sym(rng, limit));
    }

    void forward(const T* inputs)
    {
        in = inputs;
        for (int i = 0; i < n_out; ++i) {
            const T* w = &W[(size_t)i * n_in];
            T s;
            if (sequential_sum) {
                // version5 order: bias first, then inputs one by one.
                s = b[i];
                for (int j = 0; j < n_in; ++j) s += inputs[j] * w[j];
            } else {
                s = b[i] + dot_blocked(inputs, w, n_in);
            }
            z[i] = s;
            out[i] = activate(act[i], s);
        }
    }

    void backward(const T* d_values, T lr, bool compute_d_inputs = true)
    {
        if (compute_d_inputs) std::fill(d_in.begin(), d_in.end(), T(0));

        for (int i = 0; i < n_out; ++i) {
            T d = d_values[i] * activate_derivative(act[i], out[i]);
            if (gstats) gstats->add((double)d, (double)clip);
            if (d >  clip) d =  clip;
            if (d < -clip) d = -clip;

            T* w = &W[(size_t)i * n_in];
            const T step = lr * d;

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

    // Fixed-order dot product: sixteen running partial sums over j mod 16, combined as a
    // balanced tree, then the tail added in order. The order never depends on the
    // platform or the vector width, so the result is reproducible everywhere, and the
    // independent chains let the compiler vectorise without reassociating.
    static T dot_blocked(const T* x, const T* w, int n)
    {
        T a[16];
        for (int l = 0; l < 16; ++l) a[l] = T(0);
        int j = 0;
        for (; j + 16 <= n; j += 16)
            for (int l = 0; l < 16; ++l) a[l] += x[j + l] * w[j + l];
        for (int step = 8; step >= 1; step /= 2)
            for (int l = 0; l < step; ++l) a[l] = a[l] + a[l + step];
        T s = a[0];
        for (; j < n; ++j) s += x[j] * w[j];
        return s;
    }

    int n_in, n_out;
    std::vector<T> W, b;            // W is n_out rows of n_in
    std::vector<T> z, out, d_in;    // pre-activation, post-activation, dL/d(input)
    std::vector<Act> act;
    const T* in = nullptr;

    bool sequential_sum = false;    // true only for --rng-mode legacy (version5 numerics)
    T clip = T(GRAD_CLIP);
    GradStats* gstats = nullptr;
};

#endif
