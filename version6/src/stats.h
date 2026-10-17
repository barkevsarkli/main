#ifndef STATS_H
#define STATS_H

#include <string>
#include <vector>
#include "dataset.h"
#include "model.h"

// Activation / gradient diagnostics (--stats-out, --stats-every). PROMPT.md section 8.
//
// Rules that keep training untouched:
//   * probe measurements are forward-only passes over a fixed probe set (the first 1 000
//     validation indices) after the epoch's updates; the forward pass overwrites only the
//     layers' activation scratch buffers, which the next training step recomputes anyway;
//   * no RNG is used anywhere;
//   * gradient statistics read dL/dz values the backward pass computes regardless.
// tests/test_determinism.py checks that result rows are identical with stats on and off.
//
// Output: long CSV, one row per (run_id, epoch, layer, group, metric) with a `value`
// column; the z histogram is one row whose value holds 50 pipe-separated counts
// (underflow, 48 bins on [-12, 12], overflow).
//
// Layers: conv (post-activation measured before pooling; a unit is a channel), hidden1,
// hidden2 (mlp2 only), output. Groups: one per activation present in the layer's mask,
// plus `all` when the layer mixes activations. Activation-specific metrics are reported
// in the per-activation groups only.
class StatsCollector
{
public:
    StatsCollector(const std::string& run_id, const std::vector<uint32_t>& probe_idx)
        : run_id(run_id), probe(probe_idx) {}

    // Turn gradient accumulation on for every layer of the model.
    void attach(Model<float>& m);

    // Probe measurements (forward-only) for `epoch` (0 = initialisation).
    void record_probe(int epoch, Model<float>& m, const Dataset& ds);

    // Call at the start of every epoch: remembers the weights for ||dW|| / ||W||.
    void begin_epoch(Model<float>& m);

    // Training-time statistics accumulated over the epoch that just finished.
    void record_train(int epoch, Model<float>& m);

    const std::string& text() const { return out; }
    static std::string header() { return "run_id,epoch,layer,group,metric,value\n"; }

private:
    struct LayerRef { std::string name; GradStats* g; std::vector<float>* W; };
    std::vector<LayerRef> layers(Model<float>& m);
    void emit(int epoch, const std::string& layer, const std::string& group, const std::string& metric, double v);
    void emit_str(int epoch, const std::string& layer, const std::string& group, const std::string& metric,
                  const std::string& v);

    std::string run_id;
    std::vector<uint32_t> probe;
    std::string out;
    std::vector<GradStats> gstats;              // one per layer, stable addresses after attach()
    std::vector<std::vector<float>> w_start;    // weights at the start of the epoch
};

#endif
