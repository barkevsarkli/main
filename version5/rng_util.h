#ifndef RNG_UTIL_H
#define RNG_UTIL_H

#include <cstdint>
#include <vector>
#include <random>

// Platform-independent shuffling.
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

}  // namespace detu

#endif
