#ifndef RNG_UTIL_H
#define RNG_UTIL_H

#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

// Platform-independent random numbers.
//
// std::shuffle is NOT specified to consume its generator in any particular way, so
// libstdc++ (GNU/Linux) and libc++ (macOS) produce different permutations from the same
// std::mt19937 state. version5 hit this in its first sweep: results produced on a Linux
// host could not be topped up on this Mac, because the same SPLIT_SEED gave a different
// train/val/test partition -- confirmed by comparing the per-class composition of the
// stored test sets -- and two widths had to be re-run in full.
//
// std::mt19937 itself is fully specified, and so is std::uniform_int_distribution's
// *range*, but not the exact draw sequence of the distribution either. So this does the
// Fisher-Yates walk directly on raw generator output with unbiased rejection sampling,
// which depends on nothing but the standard's definition of mt19937.
//
// version6 extends the same rule to every new random quantity: uniform floats, normals and
// the per-purpose seeds below are built from raw 32-bit draws only. std::uniform_real_
// distribution survives in one place, the `legacy` weight init, because version5's stored
// results depend on it (see DECISIONS.md).
//
// Consequence: a run is reproducible on any platform and any standard library. Results
// from different machines can be pooled, which is what the whole paired design needs.

namespace detu {

// Uniform in [0, n) from raw mt19937 output, rejecting the short tail so every value is
// equally likely. mt19937 yields 32 bits in [0, 2^32).
inline uint32_t bounded(std::mt19937& g, uint32_t n)
{
    if (n <= 1) return 0;
    // Largest multiple of n that fits in 2^32; draws at or above it are rejected so every
    // value in [0, n) is equally likely.  The limit is kept in 64 bits on purpose: when n
    // divides 2^32 exactly (every power of two, and n = 2 happens on the last swap of
    // every shuffle) the limit *is* 2^32, and truncating it to uint32_t would make it 0,
    // rejecting every draw and hanging forever.
    const uint64_t limit = 0x100000000ull - (0x100000000ull % n);
    uint32_t x;
    do { x = (uint32_t)g(); } while ((uint64_t)x >= limit);
    return x % n;
}

// Fisher-Yates, walking downward. Fixed for all time: changing the direction or the
// bounded() scheme changes every seeded result in the project.
template <class T>
inline void shuffle(std::vector<T>& v, std::mt19937& g)
{
    for (size_t i = v.size(); i > 1; --i) {
        const uint32_t j = bounded(g, (uint32_t)i);
        std::swap(v[i - 1], v[j]);
    }
}

// Uniform float in [0, 1) from the top 24 bits of one draw: k * 2^-24 with k in
// [0, 2^24). Every value is exactly representable, so no rounding happens anywhere.
inline float uniform01(std::mt19937& g)
{
    return (float)((uint32_t)g() >> 8) * (1.0f / 16777216.0f);
}

// Uniform float in [-limit, limit). 2u - 1 is exact for u = k * 2^-24, so the only
// rounding is the final multiplication -- one IEEE operation, identical on every platform.
inline float uniform_sym(std::mt19937& g, float limit)
{
    const float u = uniform01(g);
    return limit * (2.0f * u - 1.0f);
}

// Box-Muller normal pair in double precision. u1 is in (0, 1] so the log is finite.
// The result goes through std::log / std::cos / std::sin, whose last bit is not specified
// across C libraries; it is used only to build the random_frozen kernel bank, which is
// generated once and stored (kernels/*.json), so runs never recompute it.
inline void normal_pair(std::mt19937& g, double& z0, double& z1)
{
    const double u1 = ((double)((uint32_t)g() >> 8) + 1.0) * (1.0 / 16777216.0);
    const double u2 = (double)((uint32_t)g() >> 8) * (1.0 / 16777216.0);
    const double r  = std::sqrt(-2.0 * std::log(u1));
    const double t  = 6.283185307179586476925286766559 * u2;
    z0 = r * std::cos(t);
    z1 = r * std::sin(t);
}

// SplitMix64 finaliser (Steele, Lea & Flood 2014), the standard 64-bit mixer.
inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Independent per-purpose seed for --rng-mode split:
//   stream_seed(seed, id) = high 32 bits of splitmix64((uint32(seed) << 32) | id)
// with id 1 = initial weights, 2 = activation masks, 3 = data order.
enum Stream : uint32_t { STREAM_INIT = 1, STREAM_MASK = 2, STREAM_ORDER = 3 };

inline uint32_t stream_seed(uint32_t seed, uint32_t stream_id)
{
    return (uint32_t)(splitmix64(((uint64_t)seed << 32) | (uint64_t)stream_id) >> 32);
}

}  // namespace detu

#endif
