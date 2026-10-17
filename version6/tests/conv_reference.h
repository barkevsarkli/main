#ifndef CONV_REFERENCE_H
#define CONV_REFERENCE_H

#include <cmath>
#include <vector>
#include "net.h"

// Plain, readable reference for the conv block: direct loops straight from the
// definitions, no padding buffer, no sparsity tricks. Used only to check src/conv.h.
//
//   z[c,y,x] = b[c] + sum_{ci,ky,kx} W[c,ci,ky,kx] * x[ci, y+ky-r, x+kx-r]   (0 outside)
//   h = act_c(z);   pooled[c,py,px] = max over the P x P window (first max in row-major)
//   backward: g[c,y,x] = clip(d_pooled[c,py,px] * act'(h)) at the arg-max, 0 elsewhere
//             dW[c,ci,ky,kx] = sum_{y,x} g[c,y,x] * x[ci, y+ky-r, x+kx-r];  db[c] = sum g
// `abs_*` hold the matching sums of |term|, the scale against which rounding error is judged.
template <class T>
struct ConvRef
{
    int cin, H, Wd, C, K, P, r, Ho, Wo;
    std::vector<T> W, b;
    std::vector<Act> act;
    std::vector<T> z, h, pooled, abs_z;
    std::vector<int> argmax;
    std::vector<T> dW, db, abs_dW, abs_db;
    T clip = T(GRAD_CLIP);

    ConvRef(int cin_, int h_, int w_, int c_, int k_, int p_, const std::vector<Act>& a)
        : cin(cin_), H(h_), Wd(w_), C(c_), K(k_), P(p_), r((k_ - 1) / 2), Ho(h_ / p_), Wo(w_ / p_),
          W((size_t)c_ * cin_ * k_ * k_), b(c_), act(a) {}

    T in(const T* x, int ci, int y, int xx) const
    {
        if (y < 0 || y >= H || xx < 0 || xx >= Wd) return T(0);
        return x[((size_t)ci * H + y) * Wd + xx];
    }

    void forward(const T* x)
    {
        z.assign((size_t)C * H * Wd, T(0));
        abs_z.assign(z.size(), T(0));
        h.assign(z.size(), T(0));
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int xx = 0; xx < Wd; ++xx) {
                    T s = b[c], a = std::fabs(b[c]);
                    for (int ci = 0; ci < cin; ++ci)
                        for (int ky = 0; ky < K; ++ky)
                            for (int kx = 0; kx < K; ++kx) {
                                const T t = W[(((size_t)c * cin + ci) * K + ky) * K + kx] * in(x, ci, y + ky - r, xx + kx - r);
                                s += t;
                                a += std::fabs(t);
                            }
                    const size_t i = ((size_t)c * H + y) * Wd + xx;
                    z[i] = s;
                    abs_z[i] = a;
                    h[i] = activate(act[c], s);
                }
        pooled.assign((size_t)C * Ho * Wo, T(0));
        argmax.assign(pooled.size(), 0);
        for (int c = 0; c < C; ++c)
            for (int py = 0; py < Ho; ++py)
                for (int px = 0; px < Wo; ++px) {
                    int best = -1;
                    T bv = 0;
                    for (int dy = 0; dy < P; ++dy)
                        for (int dx = 0; dx < P; ++dx) {
                            const int pos = (py * P + dy) * Wd + px * P + dx;
                            const T v = h[(size_t)c * H * Wd + pos];
                            if (best < 0 || v > bv) { bv = v; best = pos; }
                        }
                    pooled[((size_t)c * Ho + py) * Wo + px] = bv;
                    argmax[((size_t)c * Ho + py) * Wo + px] = best;
                }
    }

    void backward(const T* x, const T* d_pooled)
    {
        std::vector<T> g((size_t)C * H * Wd, T(0));
        for (int c = 0; c < C; ++c)
            for (int q = 0; q < Ho * Wo; ++q) {
                const int pos = argmax[(size_t)c * Ho * Wo + q];
                const size_t i = (size_t)c * H * Wd + pos;
                T v = d_pooled[(size_t)c * Ho * Wo + q] * activate_derivative(act[c], h[i]);
                if (v > clip) v = clip;
                if (v < -clip) v = -clip;
                g[i] = v;
            }
        dW.assign(W.size(), T(0));
        abs_dW.assign(W.size(), T(0));
        db.assign(C, T(0));
        for (int c = 0; c < C; ++c)
            for (int ci = 0; ci < cin; ++ci)
                for (int ky = 0; ky < K; ++ky)
                    for (int kx = 0; kx < K; ++kx) {
                        T s = 0, a = 0;
                        for (int y = 0; y < H; ++y)
                            for (int xx = 0; xx < Wd; ++xx) {
                                const T t = g[((size_t)c * H + y) * Wd + xx] * in(x, ci, y + ky - r, xx + kx - r);
                                s += t;
                                a += std::fabs(t);
                            }
                        const size_t k = (((size_t)c * cin + ci) * K + ky) * K + kx;
                        dW[k] = s;
                        abs_dW[k] = a;
                    }
        abs_db.assign(C, T(0));
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H * Wd; ++i) {
                db[c] += g[(size_t)c * H * Wd + i];
                abs_db[c] += std::fabs(g[(size_t)c * H * Wd + i]);
            }
    }
};

#endif
