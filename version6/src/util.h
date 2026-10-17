#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace util {

// FNV-1a 64-bit, fed byte by byte. Values are serialised little-endian explicitly so the
// hash of the same numbers is the same on every platform.
struct Fnv1a
{
    uint64_t h = 0xcbf29ce484222325ull;
    void byte(uint8_t b) { h ^= b; h *= 0x100000001b3ull; }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) byte((uint8_t)(v >> (8 * i))); }
    void f32(float f) { uint32_t v; std::memcpy(&v, &f, 4); u32(v); }
    template <class T> void floats(const std::vector<T>& v) { for (T x : v) f32((float)x); }
    void u32s(const std::vector<uint32_t>& v) { for (uint32_t x : v) u32(x); }
    std::string hex() const
    {
        static const char* d = "0123456789abcdef";
        std::string s(16, '0');
        for (int i = 0; i < 16; ++i) s[15 - i] = d[(h >> (4 * i)) & 0xf];
        return s;
    }
};

template <class T> inline std::string join(const std::vector<T>& v, const char* sep = "|")
{
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s << sep; s << v[i]; }
    return s.str();
}

// RFC 4180 quoting, only when needed (provenance strings may hold spaces or commas).
inline std::string csv_field(const std::string& s)
{
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}

// Write `content` to `path` atomically: write a sibling temp file, then rename over the
// target, so a killed process never leaves a half-written row or stats file behind.
bool write_atomic(const std::string& path, const std::string& content);

}  // namespace util

#endif
