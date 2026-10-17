#include "net.h"
#include "rng_util.h"
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
        detu::shuffle(m, rng);
    } else {
        throw std::runtime_error("unknown layout: " + layout);
    }
    return m;
}

std::vector<Act> rotate_mask(const std::vector<Act>& m)
{
    const size_t n = m.size();
    std::vector<Act> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = m[(i + 1) % n];
    return r;
}
