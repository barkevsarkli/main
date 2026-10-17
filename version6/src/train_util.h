#ifndef TRAIN_UTIL_H
#define TRAIN_UTIL_H

#include <algorithm>
#include <cmath>
#include "model.h"

// Loss of one sample and, if d_loss is non-null, dL/d(output).
//   CE:  softmax over linear outputs, L = -log(max(p_label, 1e-12)), dL/do = p - onehot
//   MSE: sigmoid outputs, L = mean_j (o_j - onehot_j)^2, dL/do = 2 (o - onehot)
// Identical arithmetic to version5's training loop (in float), so --rng-mode legacy
// reproduces its losses bit for bit.
template <class T>
inline double sample_loss(const T* o, int label, bool ce, T* d_loss)
{
    if (ce) {
        T mx = *std::max_element(o, o + OUTPUT_SIZE), Z = 0;
        T prob[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] = std::exp(o[j] - mx); Z += prob[j]; }
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            prob[j] /= Z;
            if (d_loss) d_loss[j] = prob[j] - (j == label ? T(1) : T(0));
        }
        return (double)(-std::log(std::max(prob[label], T(1e-12f))));
    }
    T l = 0;
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        T e = o[j] - (j == label ? T(1) : T(0));
        l += e * e;
        if (d_loss) d_loss[j] = 2 * e;
    }
    return (double)(l / OUTPUT_SIZE);
}

template <class T>
inline bool all_finite(const T* v, int n)
{
    for (int i = 0; i < n; ++i) if (!std::isfinite(v[i])) return false;
    return true;
}

// --abort-on-collapse N: true once N consecutive epochs have mean training loss within
// 0.01 of ln 10 (an input-independent output) and validation accuracy below 15%.
// N = 0 disables it.
struct CollapseWatch
{
    explicit CollapseWatch(int n_) : n(n_) {}
    bool update(double train_loss, double val_acc)
    {
        if (n <= 0) return false;
        const bool collapsed = std::fabs(train_loss - std::log(10.0)) < 0.01 && val_acc < 15.0;
        run = collapsed ? run + 1 : 0;
        return run >= n;
    }
    int n, run = 0;
};

#endif
