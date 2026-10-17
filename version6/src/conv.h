#ifndef CONV_H
#define CONV_H

#include <cstdint>
#include <stdexcept>
#include <vector>
#include "net.h"

// One convolutional block: Conv2D (stride 1, "same" zero padding, full channel mixing)
// -> per-channel activation -> MaxPool(P, stride P) -> flatten in CHW order.
//
// Shapes: input C_in x H x W (CHW), weights C x C_in x K x K, one bias per output
// channel, pre-activation z and activation h are C x H x W, pooled output is
// C x (H/P) x (W/P).
//
// This is the production version: loop order and memory layout tuned for cache, still
// single-threaded with a fixed summation order. tests/conv_reference.h holds the plain
// textbook loops it is checked against (tests/test_conv_reference.cpp).
//
// Summation order (fixed, platform independent):
//   z[c,y,x] = b[c] + sum over (ci, ky, kx) in lexicographic order of
//              W[c,ci,ky,kx] * xpad[ci, y+ky, x+kx]
//   dW[c,ci,ky,kx] = sum over pooled cells (py, px) in row-major order of
//              g[c, argmax(py,px)] * xpad[ci, y+ky, x+kx]
// where xpad is the input with (K-1)/2 zeros on every side. Only arg-max positions carry
// a gradient (MaxPool backward); every other dL/dz is exactly 0 and contributes nothing.
template <class T>
class Conv
{
public:
    Conv(int cin_, int h_, int w_, int cout_, int k_, int pool_, const std::vector<Act>& acts)
        : cin(cin_), H(h_), Wd(w_), C(cout_), K(k_), P(pool_), r((k_ - 1) / 2),
          Hp(h_ + k_ - 1), Wp(w_ + k_ - 1), Ho(h_ / pool_), Wo(w_ / pool_),
          W((size_t)cout_ * cin_ * k_ * k_, T(0)), b(cout_, T(0)), act(acts)
    {
        if (K % 2 == 0) throw std::runtime_error("Conv: kernel size must be odd");
        if (H % P || Wd % P) throw std::runtime_error("Conv: pool size must divide the image size");
        if ((int)act.size() != C) throw std::runtime_error("Conv: activation mask size does not match channels");
        xpad.assign((size_t)cin * Hp * Wp, T(0));          // borders stay zero for good
        z.assign((size_t)C * H * Wd, T(0));
        h.assign((size_t)C * H * Wd, T(0));
        pooled.assign((size_t)C * Ho * Wo, T(0));
        argmax.assign((size_t)C * Ho * Wo, 0);
        dW.assign(W.size(), T(0));
        db.assign(C, T(0));
    }

    int fan_in() const { return cin * K * K; }
    int out_size() const { return C * Ho * Wo; }

    void forward(const T* x)
    {
        // Copy the image into the interior of the padded buffer.
        for (int ci = 0; ci < cin; ++ci)
            for (int y = 0; y < H; ++y) {
                const T* src = x + ((size_t)ci * H + y) * Wd;
                T* dst = &xpad[((size_t)ci * Hp + y + r) * Wp + r];
                for (int xx = 0; xx < Wd; ++xx) dst[xx] = src[xx];
            }

        for (int c = 0; c < C; ++c) {
            T* zc = &z[(size_t)c * H * Wd];
            for (int i = 0; i < H * Wd; ++i) zc[i] = b[c];
            for (int ci = 0; ci < cin; ++ci) {
                const T* xc = &xpad[(size_t)ci * Hp * Wp];
                for (int ky = 0; ky < K; ++ky)
                    for (int kx = 0; kx < K; ++kx) {
                        const T w = W[(((size_t)c * cin + ci) * K + ky) * K + kx];
                        for (int y = 0; y < H; ++y) {
                            const T* xr = xc + (size_t)(y + ky) * Wp + kx;
                            T* zr = zc + (size_t)y * Wd;
                            for (int xx = 0; xx < Wd; ++xx) zr[xx] += w * xr[xx];
                        }
                    }
            }
            const Act a = act[c];
            T* hc = &h[(size_t)c * H * Wd];
            for (int i = 0; i < H * Wd; ++i) hc[i] = activate(a, zc[i]);

            // MaxPool over h; ties go to the first element in row-major order (strict >).
            T* pc = &pooled[(size_t)c * Ho * Wo];
            int32_t* ac = &argmax[(size_t)c * Ho * Wo];
            for (int py = 0; py < Ho; ++py)
                for (int px = 0; px < Wo; ++px) {
                    int best = (py * P) * Wd + px * P;
                    T bv = hc[best];
                    for (int dy = 0; dy < P; ++dy)
                        for (int dx = 0; dx < P; ++dx) {
                            const int pos = (py * P + dy) * Wd + px * P + dx;
                            if (hc[pos] > bv) { bv = hc[pos]; best = pos; }
                        }
                    pc[py * Wo + px] = bv;
                    ac[py * Wo + px] = best;
                }
        }
    }

    // d_pooled: dL/d(pooled output), C x Ho x Wo. Accumulates dW and db over every
    // position first, then applies one update per sample (the kernel is shared).
    void backward(const T* d_pooled, T lr)
    {
        std::fill(dW.begin(), dW.end(), T(0));
        std::fill(db.begin(), db.end(), T(0));
        const int KK = K * K;
        for (int c = 0; c < C; ++c) {
            const Act a = act[c];
            const T* hc = &h[(size_t)c * H * Wd];
            const int32_t* ac = &argmax[(size_t)c * Ho * Wo];
            const T* gc = d_pooled + (size_t)c * Ho * Wo;
            T* dWc = &dW[(size_t)c * cin * KK];
            for (int q = 0; q < Ho * Wo; ++q) {
                const int pos = ac[q];
                T g = gc[q] * activate_derivative(a, hc[pos]);
                if (gstats) gstats->add((double)g, (double)clip);
                if (g >  clip) g =  clip;
                if (g < -clip) g = -clip;
                // A zero gradient (dead ReLU, saturated unit) would add +-0 to every
                // accumulator, which leaves every value bit-for-bit unchanged; skip it.
                if (g == T(0)) continue;
                db[c] += g;
                const int y = pos / Wd, xx = pos % Wd;
                for (int ci = 0; ci < cin; ++ci) {
                    const T* xc = &xpad[(size_t)ci * Hp * Wp];
                    T* dWk = dWc + (size_t)ci * KK;
                    for (int ky = 0; ky < K; ++ky) {
                        const T* xr = xc + (size_t)(y + ky) * Wp + xx;
                        T* dr = dWk + ky * K;
                        for (int kx = 0; kx < K; ++kx) dr[kx] += g * xr[kx];
                    }
                }
            }
            if (gstats) {
                // Non-arg-max positions have dL/dz = 0 exactly; count them so the mean is
                // over every pre-activation element, as in the dense layers.
                gstats->n += (uint64_t)(H * Wd - Ho * Wo);
            }
        }
        for (size_t i = 0; i < W.size(); ++i) W[i] -= lr * dW[i];
        for (int c = 0; c < C; ++c) b[c] -= lr * db[c];
    }

    int cin, H, Wd, C, K, P, r, Hp, Wp, Ho, Wo;
    std::vector<T> W, b;
    std::vector<Act> act;
    std::vector<T> xpad, z, h, pooled, dW, db;
    std::vector<int32_t> argmax;

    T clip = T(GRAD_CLIP);
    GradStats* gstats = nullptr;
};

#endif
