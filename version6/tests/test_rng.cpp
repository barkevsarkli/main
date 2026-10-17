// Portable random numbers: mt19937 conformance, SplitMix64 reference values, uniform and
// Box-Muller generators, stream-seed independence, shuffle validity.

#include <cmath>
#include <set>
#include "check.h"
#include "rng_util.h"

int main()
{
    // The C++ standard fixes the 10000th output of a default-seeded mt19937.
    { std::mt19937 g; for (int i = 0; i < 9999; ++i) g(); CHECK(g() == 4123659995u, "mt19937 conformance"); }

    // SplitMix64 reference: the first output of the generator with state 0.
    CHECK(detu::splitmix64(0) == 0xE220A8397B1DCDAFull, "splitmix64(0)");
    CHECK(detu::splitmix64(0x9E3779B97F4A7C15ull) == 0x6E789E6AA1B965F4ull, "splitmix64 second output");

    // Stream seeds: documented formula and no collisions over seeds 0..9999 x 3 streams.
    CHECK(detu::stream_seed(7, 2) == (uint32_t)(detu::splitmix64((7ull << 32) | 2ull) >> 32), "formula");
    {
        std::set<uint32_t> s;
        for (uint32_t seed = 0; seed < 10000; ++seed)
            for (uint32_t id = 1; id <= 3; ++id) s.insert(detu::stream_seed(seed, id));
        CHECK(s.size() == 30000, "stream seed collisions: %zu unique of 30000", s.size());
    }

    // uniform01: exact multiples of 2^-24 in [0, 1); uniform_sym bounded; moments.
    {
        std::mt19937 g(123);
        double s = 0, s2 = 0;
        const int n = 1000000;
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const float u = detu::uniform01(g);
            ok = ok && u >= 0.0f && u < 1.0f && std::ldexp((double)u, 24) == std::floor(std::ldexp((double)u, 24));
            s += u; s2 += (double)u * u;
        }
        CHECK(ok, "uniform01 range/granularity");
        CHECK(std::fabs(s / n - 0.5) < 2e-3, "uniform01 mean %.5f", s / n);
        CHECK(std::fabs(s2 / n - s * s / n / n - 1.0 / 12) < 2e-3, "uniform01 variance");
        std::mt19937 h(5);
        bool okb = true;
        for (int i = 0; i < 100000; ++i) { const float w = detu::uniform_sym(h, 0.25f); okb = okb && w >= -0.25f && w < 0.25f; }
        CHECK(okb, "uniform_sym range");
    }

    // Box-Muller: mean 0, variance 1, small skew/excess kurtosis.
    {
        std::mt19937 g(20261018);
        const int n = 1000000;
        double m1 = 0, m2 = 0, m3 = 0, m4 = 0;
        for (int i = 0; i < n / 2; ++i) {
            double a, b;
            detu::normal_pair(g, a, b);
            for (double z : {a, b}) { m1 += z; m2 += z * z; m3 += z * z * z; m4 += z * z * z * z; }
        }
        m1 /= n; m2 /= n; m3 /= n; m4 /= n;
        CHECK(std::fabs(m1) < 5e-3, "normal mean %.4f", m1);
        CHECK(std::fabs(m2 - 1) < 5e-3, "normal var %.4f", m2);
        CHECK(std::fabs(m3) < 2e-2, "normal skew %.4f", m3);
        CHECK(std::fabs(m4 - 3) < 3e-2, "normal kurtosis %.4f", m4);
    }

    // Shuffle is a permutation and depends only on the generator state.
    {
        std::vector<uint32_t> v(1000);
        for (uint32_t i = 0; i < 1000; ++i) v[i] = i;
        std::vector<uint32_t> w = v;
        std::mt19937 g1(9), g2(9);
        detu::shuffle(v, g1);
        detu::shuffle(w, g2);
        CHECK(v == w, "shuffle reproducible");
        std::set<uint32_t> s(v.begin(), v.end());
        CHECK(s.size() == 1000, "shuffle is a permutation");
    }
    return finish("test_rng");
}
