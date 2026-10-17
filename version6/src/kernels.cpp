#include "kernels.h"
#include "rng_util.h"
#include "util.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

static const double PI = 3.141592653589793238462643383279502884;

namespace {

typedef std::vector<double> Filter;   // K * K, row-major: index (v + r) * K + (u + r)

double binom(int n, int k)
{
    double c = 1;
    for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
    return c;
}

Filter gaussian(int K, double sigma, bool sum_to_one)
{
    const int r = (K - 1) / 2;
    Filter f(K * K);
    double s = 0;
    for (int v = -r; v <= r; ++v)
        for (int u = -r; u <= r; ++u) {
            const double g = std::exp(-(u * u + v * v) / (2 * sigma * sigma));
            f[(v + r) * K + (u + r)] = g;
            s += g;
        }
    if (sum_to_one) for (double& x : f) x /= s;
    return f;
}

void subtract_mean(Filter& f)
{
    double m = 0;
    for (double x : f) m += x;
    m /= (double)f.size();
    for (double& x : f) x -= m;
}

void l2_normalise(Filter& f, const std::string& what)
{
    double n = 0;
    for (double x : f) n += x * x;
    n = std::sqrt(n);
    if (!(n > 1e-8)) throw std::runtime_error("make_bank: degenerate (all-zero) filter in " + what);
    for (double& x : f) x /= n;
}

}  // namespace

double gabor_lambda(int K)
{
    return K == 3 ? 2.5 : std::max(2.0, (K + 1) / 2.0);
}

std::vector<double> make_bank(const std::string& family, int K, uint32_t kernel_seed)
{
    if (K < 3 || K % 2 == 0) throw std::runtime_error("make_bank: K must be odd and >= 3");
    const int r = (K - 1) / 2;
    const double sK = K / 6.0;
    std::vector<Filter> bank;

    if (family == "gauss") {
        for (int j = 0; j < 8; ++j) bank.push_back(gaussian(K, sK * (0.5 + 0.25 * j), true));

    } else if (family == "sharpen") {
        const double alphas[4] = {0.5, 1, 2, 4};
        const double sigmas[2] = {sK, 1.5 * sK};
        for (double a : alphas)
            for (double s : sigmas) {
                Filter g = gaussian(K, s, true), f(K * K);
                for (int i = 0; i < K * K; ++i) {
                    const double delta = (i == r * K + r) ? 1.0 : 0.0;
                    f[i] = delta + a * (delta - g[i]);
                }
                bank.push_back(f);
            }

    } else if (family == "sobel") {
        std::vector<double> s(K), a(K - 2), d(K, 0.0);
        for (int i = 0; i < K; ++i) s[i] = binom(K - 1, i);
        for (int i = 0; i < K - 2; ++i) a[i] = binom(K - 3, i);
        // d = a * [-1, 0, 1] (full convolution): d[n] = -a[n] + a[n-2]
        for (int n = 0; n < K; ++n) {
            if (n < K - 2) d[n] -= a[n];
            if (n - 2 >= 0) d[n] += a[n - 2];
        }
        Filter gx(K * K), gy(K * K);
        for (int v = 0; v < K; ++v)
            for (int u = 0; u < K; ++u) {
                gx[v * K + u] = s[v] * d[u];
                gy[v * K + u] = s[u] * d[v];
            }
        for (int j = 0; j < 8; ++j) {
            const double th = 2 * PI * j / 8;
            Filter f(K * K);
            for (int i = 0; i < K * K; ++i) f[i] = std::cos(th) * gx[i] + std::sin(th) * gy[i];
            bank.push_back(f);
        }

    } else if (family == "laplacian") {
        const double mult[4] = {0.5, 0.75, 1.0, 1.25};
        std::vector<Filter> on;
        for (double m : mult) {
            const double s = sK * m;
            Filter f(K * K);
            for (int v = -r; v <= r; ++v)
                for (int u = -r; u <= r; ++u) {
                    const double q = u * u + v * v;
                    const double log_ = (q - 2 * s * s) / (s * s * s * s) * std::exp(-q / (2 * s * s));
                    f[(v + r) * K + (u + r)] = -log_;        // on-centre: positive centre
                }
            subtract_mean(f);
            on.push_back(f);
        }
        for (auto& f : on) bank.push_back(f);
        for (auto f : on) { for (double& x : f) x = -x; bank.push_back(f); }

    } else if (family == "gabor") {
        const double lambda = gabor_lambda(K), sigma = 0.56 * lambda, gamma = 0.5;
        const double thetas[4] = {0, PI / 4, PI / 2, 3 * PI / 4};
        const double psis[2] = {0, PI / 2};
        for (double th : thetas)
            for (double psi : psis) {
                Filter f(K * K);
                for (int v = -r; v <= r; ++v)
                    for (int u = -r; u <= r; ++u) {
                        const double up = u * std::cos(th) + v * std::sin(th);
                        const double vp = -u * std::sin(th) + v * std::cos(th);
                        f[(v + r) * K + (u + r)] =
                            std::exp(-(up * up + gamma * gamma * vp * vp) / (2 * sigma * sigma)) *
                            std::cos(2 * PI * up / lambda + psi);
                    }
                subtract_mean(f);
                bank.push_back(f);
            }

    } else if (family == "random_frozen") {
        std::mt19937 g(kernel_seed);
        const size_t n = (size_t)8 * K * K;
        std::vector<double> z;
        while (z.size() < n) {                       // both values of every pair, in order
            double a, b;
            detu::normal_pair(g, a, b);
            z.push_back(a);
            z.push_back(b);
        }
        z.resize(n);
        for (int j = 0; j < 8; ++j) bank.emplace_back(z.begin() + j * K * K, z.begin() + (j + 1) * K * K);

    } else {
        throw std::runtime_error("unknown kernel family: " + family);
    }

    std::vector<double> out;
    for (size_t j = 0; j < bank.size(); ++j) {
        l2_normalise(bank[j], family + " k" + std::to_string(K) + " filter " + std::to_string(j));
        out.insert(out.end(), bank[j].begin(), bank[j].end());
    }
    return out;
}

std::string bank_path(const std::string& dir, const std::string& dataset, const std::string& family, int K)
{
    return (std::filesystem::path(dir) / (dataset + "_" + family + "_k" + std::to_string(K) + ".json")).string();
}

std::vector<float> bank_weights(const std::vector<float>& filters, const std::vector<float>& gains, int K, int cin)
{
    const int KK = K * K;
    std::vector<float> W((size_t)N_BANK_FILTERS * cin * KK);
    const double rs = 1.0 / std::sqrt((double)cin);
    for (int c = 0; c < N_BANK_FILTERS; ++c) {
        const double scale = (double)gains[c] * rs;
        for (int ci = 0; ci < cin; ++ci)
            for (int k = 0; k < KK; ++k)
                W[((size_t)c * cin + ci) * KK + k] = (float)((double)filters[(size_t)c * KK + k] * scale);
    }
    return W;
}

// ---- JSON (only the flat shape written by save_bank) ---------------------------------

static std::string fmt9(double x)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", x);
    return buf;
}

static std::string fmt17(double x)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", x);
    return buf;
}

void save_bank(const std::string& path, const BankFile& b)
{
    std::ostringstream s;
    s << "{\n";
    s << "  \"dataset\": \"" << b.dataset << "\",\n";
    s << "  \"family\": \"" << b.family << "\",\n";
    s << "  \"k\": " << b.k << ",\n";
    s << "  \"kernel_seed\": " << b.kernel_seed << ",\n";
    s << "  \"calib_images\": " << b.calib_images << ",\n";
    s << "  \"tau\": " << fmt17(b.tau) << ",\n";
    s << "  \"filter_hash\": \"" << b.filter_hash << "\",\n";
    s << "  \"measured_std\": [";
    for (size_t i = 0; i < b.measured_std.size(); ++i) s << (i ? ", " : "") << fmt17(b.measured_std[i]);
    s << "],\n  \"gains\": [";
    for (size_t i = 0; i < b.gains.size(); ++i) s << (i ? ", " : "") << fmt9(b.gains[i]);
    s << "],\n  \"filters\": [";
    for (size_t i = 0; i < b.filters.size(); ++i) {
        if (i % (size_t)(b.k * b.k) == 0) s << (i ? ",\n    " : "\n    ");
        else s << ", ";
        s << fmt9(b.filters[i]);
    }
    s << "\n  ]\n}\n";
    if (!util::write_atomic(path, s.str())) throw std::runtime_error("cannot write " + path);
}

namespace {

struct JsonFlat
{
    std::map<std::string, std::string> str;
    std::map<std::string, std::vector<double>> arr;
};

JsonFlat parse_flat_json(const std::string& t, const std::string& path)
{
    JsonFlat j;
    size_t i = 0;
    auto fail = [&](const char* m) { throw std::runtime_error(std::string("bad bank json (") + m + "): " + path); };
    auto ws = [&]() { while (i < t.size() && std::isspace((unsigned char)t[i])) ++i; };
    auto str = [&]() {
        if (t[i] != '"') fail("expected string");
        size_t e = t.find('"', i + 1);
        if (e == std::string::npos) fail("unterminated string");
        std::string s = t.substr(i + 1, e - i - 1);
        i = e + 1;
        return s;
    };
    auto num = [&]() {
        const char* b = t.c_str() + i;
        char* e = nullptr;
        double v = std::strtod(b, &e);
        if (e == b) fail("expected number");
        i += (size_t)(e - b);
        return v;
    };
    ws();
    if (i >= t.size() || t[i] != '{') fail("expected {");
    ++i;
    for (;;) {
        ws();
        if (t[i] == '}') break;
        std::string key = str();
        ws();
        if (t[i] != ':') fail("expected :");
        ++i;
        ws();
        if (t[i] == '"') j.str[key] = str();
        else if (t[i] == '[') {
            ++i;
            std::vector<double> v;
            ws();
            if (t[i] != ']')
                for (;;) {
                    ws();
                    v.push_back(num());
                    ws();
                    if (t[i] == ',') { ++i; continue; }
                    if (t[i] == ']') break;
                    fail("expected , or ]");
                }
            ++i;
            j.arr[key] = v;
        } else {
            const size_t b = i;
            num();
            j.str[key] = t.substr(b, i - b);
        }
        ws();
        if (t[i] == ',') { ++i; continue; }
        if (t[i] == '}') break;
        fail("expected , or }");
    }
    return j;
}

}  // namespace

BankFile load_bank(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("kernel bank not found: " + path +
                                     " (generate it with tools/kernel_dump, see README)");
    std::stringstream ss;
    ss << f.rdbuf();
    JsonFlat j = parse_flat_json(ss.str(), path);
    BankFile b;
    try {
        b.dataset = j.str.at("dataset");
        b.family = j.str.at("family");
        b.k = std::stoi(j.str.at("k"));
        b.kernel_seed = (uint32_t)std::stoul(j.str.at("kernel_seed"));
        b.calib_images = std::stoi(j.str.at("calib_images"));
        b.tau = std::stod(j.str.at("tau"));
        b.filter_hash = j.str.at("filter_hash");
        for (double x : j.arr.at("filters")) b.filters.push_back((float)x);
        for (double x : j.arr.at("gains")) b.gains.push_back((float)x);
        b.measured_std = j.arr.at("measured_std");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("bank json is missing a field: " + path);
    }
    if ((int)b.gains.size() != N_BANK_FILTERS || (int)b.filters.size() != N_BANK_FILTERS * b.k * b.k)
        throw std::runtime_error("bank json has the wrong shape: " + path);
    util::Fnv1a h;
    for (float x : b.filters) h.f32(x);
    if (h.hex() != b.filter_hash) throw std::runtime_error("bank json filter_hash mismatch: " + path);
    return b;
}
