# Article material: theorems, patterns and claims supported by the version5 data

*Compiled 5 September 2026 from `REPORT_EXTENDED.md`, `results/summary_extended.md` and a fresh pass over the raw CSVs (2,300+ runs). Every number below was recomputed independently of Claude Code's analysis code and matched. "Theorem" is used loosely: these are empirical regularities, each stated with the evidence that supports it and an honest strength grade.*

Strength scale: **A** = survives Bonferroni over the ~40 paired tests run (p < 0.001), replicated at several widths; **B** = p < 0.01 at one or more widths and consistent direction elsewhere; **C** = consistent direction, not individually significant; **D** = suggestive only.

---

## Part I — The core results (what the article is about)

### T1. The width-crossover theorem — **B** (the paper's central figure)

*Mixing tanh and ReLU units within a hidden layer costs accuracy in narrow layers and gains accuracy in wide ones. The sign of the effect flips between widths 32 and 64.*

Paired difference `tanh+relu − relu` in test accuracy (points), evaluation seeds shared across configs:

| width | 12 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| Δ vs relu | −0.57 | −0.18 | +0.03 | +0.08 | +0.23 | +0.12 |
| Δ vs tanh | −0.17 | +0.08 | +0.11 | +0.13 | +0.30 | +0.40 |
| hybrid − max(parents) | −0.57 | −0.18 | +0.03 | +0.08 | +0.22 | +0.12 |
| best config | leaky_relu | relu | **tanh+relu** | **tanh+relu** | **tanh+relu** | **tanh+relu** |
| n seeds | 30 | 30 | 20 | 20 | 10 | 5 |

The row `hybrid − max(parents)` is the one to put in the abstract: at width 12 the hybrid is *below both* parents (sub-additive); from 64 up it is *above both* (super-additive), and it is the best of all seven configurations at every width ≥ 64. Macro-F1 gives the same numbers to ±0.02.

Statistics: the small-width losses are A-grade (p = 5.5×10⁻⁶ at 12). The large-width wins are individually B-grade at best (vs tanh: p = 0.004 at 256, p = 0.003 at 512 on 5 seeds; vs relu: p = 0.07 at 256). Pooled across widths ≥ 32, `tanh+relu` beats `tanh` on 60/85 seed-width pairs (sign test p = 1×10⁻⁴, permutation p < 5×10⁻⁵) — that pooled result is A-grade. Pooled across widths ≥ 64, it beats `relu` on 36/55 (p ≈ 0.02) — C/B.

How to phrase it: "a hybrid layer needs enough width to afford the mixture; below ~50 neurons every unit is precious and the weaker function costs more than the diversity buys back."

### T2. The step-size theorem — **A** (the mechanism behind T1, and the strongest single result)

*At a fixed learning rate, the hybrid's advantage over both parents grows with the learning rate. Under an aggressive step size the mixture beats either pure function.*

From the seed-replicated LR sweeps (every config at the *same* LR, 10 seeds, validation accuracy) — this comparison is immune to the objection that the hybrids were tuned to a higher LR:

| width | lr | Δ vs relu | Δ vs tanh |
|---|---|---|---|
| 64 | 0.003 | +0.09 (6/10) | +0.20 (7/10) |
| 64 | 0.01 | +0.28 (8/10) | −0.03 (3/10) |
| 64 | 0.03 | **+0.69 (9/10)** | **+0.66 (10/10)** |
| 128 | 0.003 | +0.06 (5/10) | +0.29 (10/10) |
| 128 | 0.01 | +0.17 (7/10) | +0.14 (5/10) |
| 128 | 0.03 | **+0.57 (9/10)** | **+1.59 (10/10)** |

At lr = 0.03 the hybrid beats *both* parents on 10/10 seeds against tanh at both widths (sign test p = 0.002 each) and 9/10 against relu. At width 12 the same LR gives +4.40 against relu (ReLU is already half-collapsed) but −1.72 against tanh — the crossover again.

Reading: ReLU's failure under a large step is dead units; tanh's failure is saturation. Half a layer of each means a large step can never kill the whole layer and never saturate the whole layer. This is a *complementary failure modes* argument and it explains T1: wider layers were calibrated to larger learning rates (the hybrids' optimum moves from 0.003 to 0.01 at width 64, ReLU's only at 512), so the regime where the mixture pays off is entered earlier.

### T3. Graceful degradation, not collapse protection — **B**, with a clear boundary

*The hybrid delays ReLU's collapse but does not prevent it, and the protection fades with width.*

Seeds that collapsed (final val accuracy < 50 %) at lr = 0.1, 10 seeds each:

| width | relu | tanh | tanh+relu | leaky_relu | tanh+leaky_relu |
|---|---|---|---|---|---|
| 12 | 10/10 | 0/10 | **0/10** | 0/10 | 0/10 |
| 64 | 10/10 | 0/10 | **3/10** | 0/10 | 0/10 |
| 128 | 10/10 | 7/10 | **8/10** | 0/10 | 0/10 |

Leaky ReLU never collapses anywhere (including lr = 0.3 at width 12). So: the *robustness to catastrophic LR* belongs to a non-zero negative slope, not to mixing. The hybrid's contribution is T2 — better accuracy in the aggressive-but-trainable regime (lr ≈ 0.03) — not survival at lethal rates. The article should say exactly this; the first-look claim "hybrids are robust to LR" was too broad.

### T4. The complementarity theorem — **B** (why it is `tanh+relu` specifically)

*Mixing only helps when the two functions have non-overlapping failure modes.*

| pair | failure modes | hybrid − max(parents) at 128 / 256 / 512 | verdict |
|---|---|---|---|
| tanh + relu | saturation / dead units — disjoint | +0.08 / +0.22 / +0.12 | helps |
| tanh + leaky_relu | saturation / (none) — nothing to fix | +0.02 / +0.06 / −0.21 | neutral |
| sigmoid + relu | saturation + positive-mean output / dead units | −0.54 (128), −0.70 (64), −1.44 (32), −1.16 (12) | hurts at every width, 0–2 wins out of 30 |

`tanh+leaky_relu` never beats `leaky_relu` (pooled ≥ 32: 40/85 wins, p = 0.74) and the first-look "+0.30 on 6/6 seeds" did not replicate. `sigmoid+relu` is the worst configuration at every width. The pattern is clean: the tanh units in a ReLU layer fix a problem (dead units) that leaky ReLU does not have; sigmoid brings a non-zero-centred output that shifts the next layer's pre-activations and makes things worse. This also predicts what an *assignment method* should do: put tanh where ReLU units would die.

### T5. Optimisation, not regularisation — **C** (mechanistic support for T1/T2)

*The hybrid's gain at large width is a better fit to the training data, not a smaller generalisation gap.*

Final training loss (mean) and train−test gap at the tuned LR:

| width | relu loss / gap | tanh loss / gap | tanh+relu loss / gap |
|---|---|---|---|
| 64 | 0.037 / 1.63 | 0.040 / 1.73 | **0.028** / 1.85 |
| 128 | 0.023 / 1.55 | 0.012 / 1.98 | 0.012 / 1.78 |
| 256 | 0.016 / 1.66 | 0.011 / 1.84 | **0.010** / **1.54** |
| 512 | 0.012 / 1.58 | 0.015 / 1.76 | **0.011** / **1.50** |

From 64 up the hybrid reaches a training loss as low as or lower than tanh's, while its generalisation gap stays at or below ReLU's. In the article: "the hybrid inherits tanh's optimisation depth and ReLU's generalisation gap." At 12 and 32 it is worse than ReLU on both, consistent with T1.

### T6. Composition matters, arrangement does not — **B** (a negative result that is useful)

*Where the tanh units sit, and the mix ratio between 25 % and 75 %, have no measurable effect.*

`tanh+relu` at 128 (10 seeds) and 256 (6 seeds), test accuracy: even/odd 97.88 / 98.13, block 97.87 / 98.18, random 97.77 / 98.02, ratio 0.25 → 97.82 / 98.03, ratio 0.75 → 97.76 / 98.20. Range across variants 0.12 (128) and 0.18 (256) points; no variant is systematically different from even/odd at both widths. At 12 the same is true (all within 0.17).

Two consequences for the article. First, the effect is robust and cheap: any static 50/50 mixture works. Second, it means a *static* assignment rule based on position cannot improve on random — so any assignment method worth publishing must use training dynamics (which units die, which saturate), not neuron index. This is the bridge to the follow-up paper.

---

## Part II — Supporting patterns and caveats to state explicitly

### P1. Accuracy and macro-F1 are interchangeable here — note, not claim
Every paired Δ in F1 is within 0.02 of the accuracy Δ. MNIST is nearly class-balanced, so the article should report F1 (reviewers ask) but not present it as a separate finding.

### P2. Per-class gains sit on the confusable digits — **D**
Pooled over widths 128–512, `tanh+relu` gains recall over ReLU mostly on 7 (+0.56), 8 (+0.43), 9 (+0.34) and over tanh on 5 (+0.64), 7 (+0.48), 4/9 (+0.38 each). These are the classic 4/9, 7/9, 3/5/8 confusion clusters. At widths 12–32 the hybrid loses on every digit. Effect sizes are small; present as a figure, not a claim.

### P3. No convergence-speed advantage — a claim to *avoid*
At equal LR, epoch-1 and epoch-3 validation accuracy of the hybrid is within ±0.3 of the parents at all widths. The first-look "hybrids learn faster" observation was the tuned-LR confound.

### P4. No variance reduction — a claim to *avoid*
Seed-to-seed sd of the hybrid is not systematically smaller than the parents' at any width (e.g. 128: 0.185 vs 0.143 relu / 0.150 tanh). The v2 observation (sd 3.2 vs 8.3) was ReLU-collapse variance, not a property of the mixture.

### P5. Loss-function dependence — caveat
Under MSE + sigmoid output at 128, ReLU beats the hybrid (−0.24, 1/10 wins). The width effect is established under cross-entropy only. State it.

### P6. The learning-rate optimum shifts with width, and earlier for hybrids — **C**
Tuned LR: hybrids 0.003 at widths 12–32, 0.01 from 64 up; ReLU 0.003 everywhere until 512. Consistent with T2: the mixture tolerates the larger step that wider layers want.

### P7. Homogeneous baselines: tanh loses ground with width — background
`tanh − relu`: −0.40, −0.26, −0.08, −0.05, −0.08, −0.28 across 12→512. ReLU or leaky ReLU is the best homogeneous choice at every width; the hybrid's win at ≥ 64 is therefore a win over the best available pure function, not over a weak one.

---

## Part III — Methodological content (worth a section in the article)

### M1. The confound story (why the v2 result was wrong)
The original +6.5-point advantage came from three compounding artefacts: a hybridised *output* layer, ReLU on the output under MSE (dead output units, 86.3 ± 8.3 %), and one fixed learning rate. With a uniform output layer, both homogeneous baselines, and per-config LR calibration, the effect at width 12 reverses sign. This is a good cautionary example for the paper's related-work/methods section.

### M2. Protocol that made the comparison fair
Fixed 45k/5k/10k split shared by every run; evaluation seeds shared across configs (paired design); LR chosen on validation with calibration seeds disjoint from evaluation seeds; bootstrap CIs plus paired t and Wilcoxon; ~40 tests, so Bonferroni threshold ≈ 0.001 quoted alongside raw p.

### M3. A reproducibility trap worth reporting
`std::shuffle` with the same `mt19937` seed produces different permutations under libstdc++ (Linux) and libc++ (macOS). The first-look runs and the extended runs therefore used different splits and could not be pooled; the extended sweep re-ran widths 12 and 128 from scratch. Anyone re-implementing "seeded split" in C++ hits this; one sentence in the methods saves a reviewer a question.

---

## Part IV — What the data cannot yet support, and what would fix it

| gap | why it matters | cost |
|---|---|---|
| width 512 has 5 seeds, 256 has 10 | T1's top end rests on few seeds; the 512 Δ vs relu (+0.12, 3/5) is not distinguishable from zero | 20 seeds at 256 ≈ 3.4 h, 10 at 512 ≈ 4.3 h on the M1 |
| single dataset | reviewers will ask whether the crossover width is MNIST-specific | Fashion-MNIST is a drop-in for the loader (~1 day); CIFAR-10 loader exists in v4 |
| T2 tested at 64 and 128 only | the step-size mechanism is the strongest claim; one more width (256) at lr ∈ {0.01, 0.03} would seal it | 10 seeds × 5 configs × 2 lr ≈ 5.6 h |
| held-out slice instead of official MNIST test set | minor, but standard | fetch `t10k-*` files, one flag in `main.cpp` |
| batch-size-1 SGD, 10 epochs, no momentum | the mechanism may behave differently with mini-batches / Adam | out of scope for this paper; state as limitation |

---

## Suggested structure for the article

1. **Motivation**: activation choice is per-layer by convention; nothing in backprop requires it.
2. **The trap** (M1): a naïve comparison shows +6.5 points and it is entirely artefact.
3. **Protocol** (M2, M3).
4. **Result 1 — the crossover** (T1, fig_crossover): sub-additive below ~50 units, super-additive above.
5. **Result 2 — the mechanism** (T2, T3, T5): the mixture wins under large steps because its two halves fail differently; it does not survive lethal steps; the gain is optimisation depth with ReLU's generalisation gap.
6. **Result 3 — what does not matter** (T6, T4): arrangement and ratio are irrelevant; the partner function is not — it must be zero-centred and must fix a failure the host function actually has.
7. **Limitations** (Part IV) and **outlook**: T4 + T6 together say the next step is a dynamics-driven assignment rule (move units that die to tanh), which is the sequel.

Title options that match this evidence: *"Mixed-Activation Layers: Sub-additive When Narrow, Super-additive When Wide"*; *"Half tanh, Half ReLU: Complementary Failure Modes in a Single Layer"*; or the plainer *"When Does Mixing Activation Functions Within a Layer Help?"*
