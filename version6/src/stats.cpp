#include "stats.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>

namespace {

const int HIST_BINS = 48;
const double HIST_LO = -12.0, HIST_HI = 12.0;

struct GroupAcc
{
    std::vector<float> z;                  // every pre-activation value in the group
    double sh = 0, sh2 = 0;                // sums of h and h^2
    uint64_t neg = 0, sat = 0;             // z < 0;  saturated (tanh |h|>0.99, sigmoid h<.01 or >.99)
    std::array<uint64_t, HIST_BINS + 2> hist{};   // [0] underflow, [1..48] bins, [49] overflow
};

struct UnitAcc { uint64_t n = 0, active = 0, sat = 0; };

bool is_sat(Act a, float h)
{
    if (a == ACT_TANH) return std::fabs(h) > 0.99f;
    if (a == ACT_SIGMOID) return h < 0.01f || h > 0.99f;
    return false;
}

void add_value(GroupAcc& g, UnitAcc& u, Act a, float z, float h)
{
    g.z.push_back(z);
    g.sh += h;
    g.sh2 += (double)h * h;
    if (z < 0) ++g.neg;
    const bool s = is_sat(a, h);
    if (s) { ++g.sat; ++u.sat; }
    int bin;
    if (z < HIST_LO) bin = 0;
    else if (z >= HIST_HI) bin = HIST_BINS + 1;
    else bin = 1 + std::min(HIST_BINS - 1, (int)((z - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS));
    ++g.hist[bin];
    ++u.n;
    if (z > 0) ++u.active;
}

// Nearest-rank quantile on a copy-free buffer (reorders it).
double quantile(std::vector<float>& v, double q)
{
    if (v.empty()) return NAN;
    const size_t k = (size_t)std::llround(q * (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    return v[k];
}

std::string fmt(double v)
{
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

}  // namespace

std::vector<StatsCollector::LayerRef> StatsCollector::layers(Model<float>& m)
{
    std::vector<LayerRef> r;
    size_t gi = 0;
    auto g = [&]() -> GradStats* { return gi < gstats.size() ? &gstats[gi++] : nullptr; };
    if (m.conv) r.push_back({"conv", g(), &m.conv->W});
    for (size_t L = 0; L < m.dense.size(); ++L) {
        const bool out = (L + 1 == m.dense.size());
        r.push_back({out ? std::string("output") : "hidden" + std::to_string(L + 1), g(), &m.dense[L].W});
    }
    return r;
}

void StatsCollector::attach(Model<float>& m)
{
    const size_t n = (m.conv ? 1 : 0) + m.dense.size();
    gstats.assign(n, GradStats());
    size_t gi = 0;
    if (m.conv) m.conv->gstats = m.conv_trainable ? &gstats[gi] : nullptr;
    if (m.conv) ++gi;
    for (auto& d : m.dense) d.gstats = &gstats[gi++];
}

void StatsCollector::emit(int epoch, const std::string& layer, const std::string& group,
                          const std::string& metric, double v)
{
    emit_str(epoch, layer, group, metric, fmt(v));
}

void StatsCollector::emit_str(int epoch, const std::string& layer, const std::string& group,
                              const std::string& metric, const std::string& v)
{
    out += run_id; out += ','; out += std::to_string(epoch); out += ',';
    out += layer; out += ','; out += group; out += ','; out += metric; out += ','; out += v; out += '\n';
}

void StatsCollector::record_probe(int epoch, Model<float>& m, const Dataset& ds)
{
    struct LayerAcc
    {
        std::string name;
        std::vector<Act> acts;               // per unit
        std::map<Act, GroupAcc> groups;
        std::vector<UnitAcc> units;
    };
    std::vector<LayerAcc> L;
    if (m.conv) L.push_back({"conv", m.conv->act, {}, std::vector<UnitAcc>(m.conv->C)});
    for (size_t i = 0; i + 1 < m.dense.size(); ++i)
        L.push_back({"hidden" + std::to_string(i + 1), m.dense[i].act, {}, std::vector<UnitAcc>(m.dense[i].n_out)});

    // Output-layer accumulators.
    std::array<double, OUTPUT_SIZE> cls_s{}, cls_s2{};
    std::array<uint64_t, OUTPUT_SIZE> pred_count{};
    double lg_s = 0, lg_s2 = 0, maxp_s = 0, ent_s = 0, loss_s = 0;
    uint64_t correct = 0;

    for (uint32_t idx : probe) {
        const float* o = m.forward(ds.input(idx));
        size_t li = 0;
        if (m.conv) {
            auto& A = L[li++];
            const int HW = m.conv->H * m.conv->Wd;
            for (int c = 0; c < m.conv->C; ++c) {
                const Act a = A.acts[c];
                GroupAcc& g = A.groups[a];
                const float* zc = &m.conv->z[(size_t)c * HW];
                const float* hc = &m.conv->h[(size_t)c * HW];
                for (int k = 0; k < HW; ++k) add_value(g, A.units[c], a, zc[k], hc[k]);
            }
        }
        for (size_t d = 0; d + 1 < m.dense.size(); ++d) {
            auto& A = L[li++];
            const auto& D = m.dense[d];
            for (int u = 0; u < D.n_out; ++u) add_value(A.groups[A.acts[u]], A.units[u], A.acts[u], D.z[u], D.out[u]);
        }

        // Output layer: logits o[0..9] (linear outputs for CE).
        int p = 0;
        float mx = o[0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) if (o[j] > mx) { mx = o[j]; p = j; }
        double Z = 0, prob[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] = std::exp((double)o[j] - mx); Z += prob[j]; }
        double ent = 0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            prob[j] /= Z;
            if (prob[j] > 0) ent -= prob[j] * std::log(prob[j]);
            cls_s[j] += o[j];
            cls_s2[j] += (double)o[j] * o[j];
            lg_s += o[j];
            lg_s2 += (double)o[j] * o[j];
        }
        const int lbl = ds.label(idx);
        ++pred_count[p];
        if (p == lbl) ++correct;
        maxp_s += prob[p];
        ent_s += ent;
        loss_s += -std::log(std::max(prob[lbl], 1e-12));
    }

    for (auto& A : L) {
        const bool mixed = A.groups.size() > 1;
        // Per activation group.
        for (auto& kv : A.groups) {
            const Act a = kv.first;
            GroupAcc& g = kv.second;
            const std::string gname = act_to_string(a);
            int n_units = 0, dead = 0, nearly = 0, sat_units = 0;
            for (size_t u = 0; u < A.units.size(); ++u) {
                if (A.acts[u] != a) continue;
                const UnitAcc& U = A.units[u];
                ++n_units;
                if (U.active == 0) ++dead;
                if ((double)U.active < 0.01 * (double)U.n) ++nearly;
                if ((double)U.sat > 0.95 * (double)U.n) ++sat_units;
            }
            const double n = (double)g.z.size();
            double zs = 0, za = 0;
            for (float z : g.z) { zs += z; za = std::max(za, (double)std::fabs(z)); }
            const double zm = zs / n;
            double zv = 0;
            for (float z : g.z) zv += ((double)z - zm) * ((double)z - zm);
            const double hm = g.sh / n;
            emit(epoch, A.name, gname, "n_units", n_units);
            emit(epoch, A.name, gname, "z_mean", zm);
            emit(epoch, A.name, gname, "z_std", std::sqrt(zv / n));
            emit(epoch, A.name, gname, "z_maxabs", za);
            emit(epoch, A.name, gname, "h_mean", hm);
            emit(epoch, A.name, gname, "h_std", std::sqrt(std::max(0.0, g.sh2 / n - hm * hm)));
            if (a == ACT_RELU || a == ACT_LEAKY_RELU) {
                emit(epoch, A.name, gname, "frac_neg", (double)g.neg / n);
                emit(epoch, A.name, gname, "dead_units", dead);
                emit(epoch, A.name, gname, "nearly_dead_units", nearly);
            } else if (a == ACT_TANH || a == ACT_SIGMOID) {
                emit(epoch, A.name, gname, "frac_sat", (double)g.sat / n);
                emit(epoch, A.name, gname, "sat_units", sat_units);
            }
            emit_str(epoch, A.name, gname, "z_hist", util::join(std::vector<uint64_t>(g.hist.begin(), g.hist.end())));
            if (!mixed) {
                emit(epoch, A.name, gname, "z_p01", quantile(g.z, 0.01));
                emit(epoch, A.name, gname, "z_p50", quantile(g.z, 0.50));
                emit(epoch, A.name, gname, "z_p99", quantile(g.z, 0.99));
            }
        }
        if (mixed) {
            // Group quantiles, then `all`. The group buffers are merged after their own
            // quantiles are taken, and released as they go, to bound peak memory.
            std::vector<float> all;
            GroupAcc tot;
            for (auto& kv : A.groups) {
                GroupAcc& g = kv.second;
                const std::string gname = act_to_string(kv.first);
                emit(epoch, A.name, gname, "z_p01", quantile(g.z, 0.01));
                emit(epoch, A.name, gname, "z_p50", quantile(g.z, 0.50));
                emit(epoch, A.name, gname, "z_p99", quantile(g.z, 0.99));
                tot.sh += g.sh; tot.sh2 += g.sh2;
                for (int b = 0; b < HIST_BINS + 2; ++b) tot.hist[b] += g.hist[b];
                all.insert(all.end(), g.z.begin(), g.z.end());
                std::vector<float>().swap(g.z);
            }
            const double n = (double)all.size();
            double zs = 0, za = 0;
            for (float z : all) { zs += z; za = std::max(za, (double)std::fabs(z)); }
            const double zm = zs / n;
            double zv = 0;
            for (float z : all) zv += ((double)z - zm) * ((double)z - zm);
            const double hm = tot.sh / n;
            emit(epoch, A.name, "all", "n_units", (double)A.units.size());
            emit(epoch, A.name, "all", "z_mean", zm);
            emit(epoch, A.name, "all", "z_std", std::sqrt(zv / n));
            emit(epoch, A.name, "all", "z_maxabs", za);
            emit(epoch, A.name, "all", "h_mean", hm);
            emit(epoch, A.name, "all", "h_std", std::sqrt(std::max(0.0, tot.sh2 / n - hm * hm)));
            emit_str(epoch, A.name, "all", "z_hist", util::join(std::vector<uint64_t>(tot.hist.begin(), tot.hist.end())));
            emit(epoch, A.name, "all", "z_p01", quantile(all, 0.01));
            emit(epoch, A.name, "all", "z_p50", quantile(all, 0.50));
            emit(epoch, A.name, "all", "z_p99", quantile(all, 0.99));
        }
    }

    const double np = (double)probe.size();
    const double nl = np * OUTPUT_SIZE;
    double between = 0;
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        const double mj = cls_s[j] / np;
        between += std::sqrt(std::max(0.0, cls_s2[j] / np - mj * mj));
    }
    int distinct = 0;
    uint64_t largest = 0;
    for (uint64_t c : pred_count) { if (c) ++distinct; largest = std::max(largest, c); }
    const double lm = lg_s / nl;
    emit(epoch, "output", "linear", "logit_mean", lm);
    emit(epoch, "output", "linear", "logit_std", std::sqrt(std::max(0.0, lg_s2 / nl - lm * lm)));
    emit(epoch, "output", "linear", "logit_between_std", between / OUTPUT_SIZE);
    emit(epoch, "output", "linear", "n_pred_classes", distinct);
    emit(epoch, "output", "linear", "max_class_share", (double)largest / np);
    emit(epoch, "output", "linear", "mean_max_prob", maxp_s / np);
    emit(epoch, "output", "linear", "mean_entropy", ent_s / np);
    emit(epoch, "output", "linear", "probe_acc", 100.0 * (double)correct / np);
    emit(epoch, "output", "linear", "probe_loss", loss_s / np);
}

void StatsCollector::begin_epoch(Model<float>& m)
{
    auto L = layers(m);
    w_start.resize(L.size());
    for (size_t i = 0; i < L.size(); ++i) w_start[i] = *L[i].W;
    for (auto& g : gstats) g.reset();
}

void StatsCollector::record_train(int epoch, Model<float>& m)
{
    auto L = layers(m);
    for (size_t i = 0; i < L.size(); ++i) {
        const std::vector<float>& W = *L[i].W;
        double wn = 0, dn = 0, w0 = 0;
        for (size_t k = 0; k < W.size(); ++k) {
            wn += (double)W[k] * W[k];
            if (i < w_start.size()) {
                const double d = (double)W[k] - w_start[i][k];
                dn += d * d;
                w0 += (double)w_start[i][k] * w_start[i][k];
            }
        }
        emit(epoch, L[i].name, "all", "w_norm", std::sqrt(wn));
        if (epoch > 0) emit(epoch, L[i].name, "all", "dw_rel", w0 > 0 ? std::sqrt(dn / w0) : 0.0);
        const bool trained = !(L[i].name == "conv" && !m.conv_trainable);
        if (epoch > 0 && trained && L[i].g && L[i].g->n) {
            const GradStats& g = *L[i].g;
            emit(epoch, L[i].name, "all", "grad_mean_abs", g.sum_abs / (double)g.n);
            emit(epoch, L[i].name, "all", "grad_max_abs", g.max_abs);
            emit(epoch, L[i].name, "all", "clip_frac", (double)g.clipped / (double)g.n);
        }
    }
}
