# Heterogeneous Activation Allocation in Multilayer Perceptrons: Accuracy, Learning-Rate Robustness, and the Effect of the Activation Allocation Ratio

**Report / manuscript draft — version 5 (AWS results)**
*Generated 15 September 2026 from the raw result CSVs in `results_aws/`. All numbers were recomputed from the CSVs; nothing in `manuscript_draft/` was read or modified.*

> **Draft conventions.** Unless stated otherwise, "accuracy" and "macro-F1" are **test scores at the validation-selected checkpoint** (`test_acc_best`, `test_f1_best`). Final-epoch scores are given in Appendix B. `±` is the standard deviation across 30 training seeds, not a confidence interval. Δ values are **paired by seed** (same seed → same data order and initial weights for the non-random layouts) and are reported in percentage points (pp) with 95% percentile bootstrap intervals (10,000 resamples of seed pairs). `*` marks a bootstrap interval that excludes zero. "Wins" counts seeds on which the hybrid scores higher. Text in *[brackets]* marks places where literature or author input is still needed.

---

## Abstract

Activation functions are almost always assigned homogeneously: every neuron in a hidden layer uses the same nonlinearity. We study a simple alternative in which each hidden layer is split between a bounded activation (tanh or sigmoid) and a rectifier (ReLU or leaky-ReLU), with neurons assigned by a fixed mask. Using a two-hidden-layer MLP on MNIST and CIFAR-10, six widths (12–512 neurons), 30 evaluation seeds per cell (2,520 main runs), per-configuration learning-rate calibration on disjoint seeds, dense 30-seed learning-rate sweeps (6,300 runs), an allocation-ratio/layout ablation and an MSE replication (13,620 runs in total), we find three consistent patterns. **(i) Accuracy:** the tanh+ReLU hybrid is the best configuration on average across widths on CIFAR-10 (mean rank 2.0 of 7) and second on MNIST (2.17, behind ReLU at 1.83); it is the top-ranked model in 5 of the 12 dataset–width cells, beats homogeneous tanh at 11 of 12 cells, and exceeds the mean of its two parents at 9 of 12 cells. Against the strongest homogeneous baseline, ReLU, its gains are small (up to +0.18 pp on MNIST and +0.36 pp on CIFAR-10) and appear only once the network is at least 64 neurons wide; at 12–32 neurons the hybrid is behind ReLU. **(ii) Robustness:** in learning-rate stress tests the hybrid is markedly harder to break. On MNIST it suffers **2.8–3.5× fewer near-chance training collapses than ReLU** and keeps accuracy within 2 pp of its peak at a **3× higher learning rate** (0.03 vs 0.01) at widths 32–128; at LR = 0.03 on MNIST-128 all 30 ReLU runs collapse to ≈10% while all 30 hybrid runs finish at 96.8%. On CIFAR-10 the same 2.9× reduction appears at LR = 0.03, but the advantage over the whole grid is smaller (1.2–1.3×). **(iii) Allocation ratio:** at a fixed learning rate, accuracy is a concave function of the rectifier share ρ, with the maximum at ρ = 50–75% rather than at either homogeneous extreme; for tanh+leaky-ReLU the ρ = 75% network beats both pure tanh (+0.23 pp) and pure leaky-ReLU (+0.36 pp) on 29/30 and 28/30 seeds. The hybrid advantage does not replicate under an MSE objective. Together, the results position heterogeneous activation allocation as a cheap, parameter-free way to obtain ReLU-level accuracy with substantially wider learning-rate tolerance in moderately wide MLPs.

---

## 1. Introduction

*[Literature context to be added: activation-function design (Glorot & Bengio 2010; Glorot, Bordes & Bengio 2011; He et al. 2015), learned/adaptive activations, mixtures of activations within a neuron, and prior heterogeneous-activation networks. Novelty claims must be checked against this literature; the phrase "activation ratio" has earlier, different uses and should not be claimed as new.]*

Homogeneous hidden layers force a single trade-off between the properties of an activation function. Rectifiers (ReLU, leaky-ReLU) give non-saturating gradients and sparse representations but are unbounded and, in plain ReLU, can permanently silence units whose preactivations become negative. Bounded, zero-centred functions such as tanh keep activations in [−1, 1] but saturate. A layer that contains both kinds of neurons can, in principle, combine the fast optimisation of rectifiers with the bounded, always-differentiable signal path of tanh.

This draft asks three questions:

1. **Accuracy.** Does assigning half of each hidden layer to tanh (or sigmoid) and half to a rectifier change test accuracy and macro-F1 relative to each homogeneous parent, and how does the answer depend on width and dataset?
2. **Robustness.** Does heterogeneity widen the range of learning rates over which training remains stable, i.e. reduce late-training collapse?
3. **Allocation ratio.** How do accuracy and stability change when the rectifier share ρ is 25%, 50% or 75% instead of 0% or 100%?

We define the **activation allocation ratio** ρ as the fraction of neurons in each hidden layer that are assigned to the second (rectifier) activation, ρ_l = N_act2,l / N_hidden,l. Each neuron carries exactly one activation; this is not a per-neuron blend ρ·ReLU(x) + (1−ρ)·tanh(x).

---

## 2. Experimental setup

### 2.1 Architecture and training protocol

| Item | Setting |
|---|---|
| Architecture | Fully connected MLP, D → H → H → 10 |
| Input dimension D | 784 (MNIST), 3072 (CIFAR-10); inputs scaled to [0, 1] |
| Hidden widths H | 12, 32, 64, 128, 256, 512 |
| Parameters | P(H, D) = H² + (D + 12)H + 10 (identical for all activation variants at a given width) |
| Output / loss (main) | Linear logits + softmax, cross-entropy |
| Output / loss (replication) | Sigmoid outputs, MSE (widths 64 and 128) |
| Optimiser | Plain SGD, batch size 1, per-epoch shuffling |
| Budget | 40 epochs; checkpoint with maximum validation accuracy is evaluated (all 40 epochs are computed) |
| Initialisation | U(−1/√fan_in, +1/√fan_in), zero biases (identical for every activation) |
| Derivative clipping | Element-wise clipping of dL/dz at ±10 |
| Leaky-ReLU slope | 0.1 |
| Data split | Fixed 45k / 5k / 10k train / validation / test. MNIST test = held-out slice of the 60k training file; CIFAR-10 test = official 10k test batch |
| Hybrid masks (main) | ρ = 0.5, `interleave` layout, same ρ in both hidden layers |
| Seeds | Calibration 101–106 (6 seeds); evaluation 1–30 (30 seeds) |

### 2.2 Configurations

| Label | Type | Hidden-layer composition | Parents |
|---|---|---|---|
| relu | homogeneous | 100% ReLU | — |
| leaky_relu | homogeneous | 100% leaky-ReLU | — |
| tanh | homogeneous | 100% tanh | — |
| sigmoid | homogeneous | 100% sigmoid | — |
| tanh+relu | hybrid | 50% tanh, 50% ReLU | tanh, relu |
| tanh+leaky_relu | hybrid | 50% tanh, 50% leaky-ReLU | tanh, leaky_relu |
| sigmoid+relu | hybrid | 50% sigmoid, 50% ReLU | sigmoid, relu |

### 2.3 Test suite

| Test | Files | Dataset(s) | Widths | Grid | Seeds | Runs | Purpose |
|---|---|---|---|---|---|---:|---|
| LR calibration | `calib_*.csv`, `c10_calib_*.csv` | MNIST, CIFAR-10 | all 6 | 7 configs × 8 LRs (1e-4 … 0.3) | 101–106 | 4,032 | Select LR per config/width; also used as a stress test |
| Main evaluation | `main_*.csv`, `c10_main_*.csv` | MNIST, CIFAR-10 | all 6 | 7 configs at calibrated LR | 1–30 | 2,520 | Accuracy, macro-F1, paired comparisons |
| Dense LR sweep | `lrsweep_64/128.csv` | MNIST | 64, 128 | 7 configs × 11 LRs | 1–30 | 4,620 | Seed-replicated robustness |
| Dense LR sweep | `c10_lrsweep_128.csv` | CIFAR-10 | 128 | 7 configs × 8 LRs | 1–30 | 1,680 | Robustness replication on a harder task |
| Ratio & layout | `layout_128.csv` | MNIST | 128 | 2 pairs × {ρ = 0.25, 0.5, 0.75 interleave; ρ = 0.5 block, random}, LR 0.01 | 1–30 | 300 | Allocation ratio and neuron placement |
| MSE calibration | `calib_mse_64/128.csv` | MNIST | 64, 128 | relu, tanh, tanh+relu × 8 LRs | 101–106 | 288 | LR selection under MSE |
| MSE evaluation | `mse_64/128.csv` | MNIST | 64, 128 | relu, tanh, tanh+relu | 1–30 | 180 | Loss-function replication |
| **Total** | 32 CSV files | | | | | **13,620** | all rows `status = ok` |

### 2.4 Metrics and statistical analysis

| Quantity | Definition |
|---|---|
| Checkpoint accuracy / macro-F1 | Test score of the epoch with the highest validation accuracy (primary endpoint) |
| Final-epoch accuracy | Test score after epoch 40 (Appendix B) |
| Paired Δ | mean over seeds of (hybrid − reference); positive favours the hybrid |
| 95% CI | percentile bootstrap over seed pairs, 10,000 resamples |
| p | two-sided paired t-test; Holm correction over the 72 hybrid-vs-parent checkpoint-accuracy tests (3 hybrids × 2 parents × 6 widths × 2 datasets) |
| Parent mean | per-seed average of the two parents' accuracies; hybrid − parent mean > 0 indicates a gain beyond simple interpolation |
| Near-chance collapse | final validation accuracy < 15% (10 classes → chance = 10%) |
| Other failure thresholds | final val < 50% (MNIST), < 25% (CIFAR-10), < 90% (MNIST, "useful accuracy lost") |
| Peak − final drop | best validation accuracy during training minus final validation accuracy |
| LR tolerance | largest LR, moving upward from a configuration's own best LR through contiguous grid points, at which the mean final validation accuracy stays within 2 pp of its peak (or at which no seed collapses) |
| Generalisation gap | clean training accuracy (fixed 10k training subset) at the selected epoch − checkpoint test accuracy |

---

## 3. Results I — Why the hybrid is better

### 3.1 Accuracy across widths

**Table 1a. MNIST, checkpoint test accuracy (%), mean ± SD over 30 seeds (bold = best in column).**

{{T:WIDE mnist accuracy}}

**Table 1b. CIFAR-10, checkpoint test accuracy (%), mean ± SD over 30 seeds.**

{{T:WIDE cifar10 accuracy}}

**Table 2a. MNIST, checkpoint macro-F1 (%).**

{{T:WIDE mnist macro-F1}}

**Table 2b. CIFAR-10, checkpoint macro-F1 (%).**

{{T:WIDE cifar10 macro-F1}}

**Table 3a. MNIST, rank of each configuration by mean accuracy (1 = best of 7).**

{{T:RANK mnist}}

**Table 3b. CIFAR-10, rank by mean accuracy.**

{{T:RANK cifar10}}

Observations:

- **tanh+ReLU is the most consistently strong configuration.** It has the best mean rank on CIFAR-10 (2.0) and the second-best on MNIST (2.17, vs ReLU 1.83), is ranked first at MNIST W = 128 and 512 and at CIFAR-10 W = 64, 128 and 256, and is never ranked below 4th. The highest MNIST accuracy in the whole study is tanh+ReLU at W = 512 (98.289%).
- **Width is the main moderator.** At W = 12–32 the rectifier homogeneous networks (ReLU, leaky-ReLU) lead on both datasets; from W = 64 onward the tanh-based hybrids move to the top.
- **Macro-F1 follows accuracy** with the same ordering in almost every cell, so the gains are not produced by a shift in class balance.
- **sigmoid+ReLU is not a useful hybrid** in this setting: it ranks 5th–7th everywhere and trails ReLU at all 12 cells, although it improves on homogeneous sigmoid on CIFAR-10.

### 3.2 Paired comparisons with the parents

**Table 4. Scoreboard of paired checkpoint-accuracy comparisons across the six widths.**

{{T:SCOREBOARD}}

**Table 5. Hybrid − best homogeneous configuration at each width (pp; `*` = CI excludes 0; wins in parentheses; the best homogeneous model is named).**

{{T:BESTHOMO}}

**Table 6. Hybrid − per-seed mean of its two parents (pp). Positive values mean the hybrid does better than simple interpolation between its parents.**

{{T:SYNERGY}}

![Figure 1](figures/fig1_paired_delta_vs_width.png)

*Figure 1. Paired Δ test accuracy (checkpoint) for the tanh hybrids against each parent, with 95% bootstrap bands.*

Key numbers (full table with F1 and Holm-adjusted p in Appendix C):

| Dataset | W | tanh+ReLU − ReLU | wins | Holm p | tanh+ReLU − tanh | wins |
|---|---:|---|---:|---:|---|---:|
| MNIST | 64 | +0.004 [−0.056, +0.066] | 15/30 | 1.000 | +0.192 [+0.125, +0.254] | 27/30 |
| MNIST | 128 | **+0.178 [+0.142, +0.213]** | 28/30 | <0.001 | +0.211 [+0.172, +0.251] | 29/30 |
| MNIST | 512 | +0.003 [−0.036, +0.041] | 15/30 | 1.000 | +0.265 [+0.221, +0.305] | 29/30 |
| CIFAR-10 | 64 | **+0.363 [+0.154, +0.579]** | 22/30 | 0.047 | +1.112 [+0.917, +1.309] | 29/30 |
| CIFAR-10 | 128 | +0.178 [−0.079, +0.424] | 17/30 | 1.000 | +1.291 [+1.077, +1.515] | 29/30 |
| CIFAR-10 | 256 | +0.153 [−0.048, +0.354] | 16/30 | 1.000 | +1.738 [+1.494, +1.978] | 30/30 |

Interpretation:

1. **Mixing is better than interpolation.** For both tanh hybrids and at every width ≥ 64 on both datasets, the hybrid is significantly above the mean of its parents (Table 6; 8/8 MNIST cells and 8/8 CIFAR-10 cells, plus CIFAR-10 W = 32 for tanh+ReLU). A mixed layer is therefore not simply "half as good as tanh and half as good as ReLU"; the two neuron populations complement each other.
2. **The hybrid recovers everything tanh loses and occasionally more.** Against tanh the gain is significant at 11/12 cells for tanh+ReLU and grows with width on CIFAR-10 (+0.57 pp at W = 12 → +1.74 pp at W = 256).
3. **Against ReLU the hybrid is non-inferior from W = 64 upward.** tanh+ReLU is never significantly worse than ReLU at W ≥ 64 on either dataset, and is significantly better at MNIST-128 (28/30 wins; survives Holm correction) and CIFAR-10-64 (22/30 wins; Holm p = 0.047). The size of the advantage is modest in absolute terms but corresponds to an **8.3% relative error reduction** at MNIST-128.
4. **Small networks prefer homogeneous rectifiers.** At W = 12–32, replacing half of the rectifier units costs 0.2–0.6 pp. With very few neurons, each bounded unit that saturates is a larger fraction of the representational budget.

### 3.3 Calibration stress tests and learning-rate robustness (the "≈3×" result)

The calibration grids were designed to pick learning rates, but because they span three orders of magnitude (1e-4 to 0.3) they double as a stress test. The dense LR sweeps repeat this with the 30 evaluation seeds.

**Table 7. Training failures across all tested learning rates, by failure definition (counts of runs).**

{{T:THR}}

**Table 8. Learning-rate tolerance in the calibration grids: largest LR (above each configuration's own best) that keeps mean final validation accuracy within 2 pp of its peak, and largest LR with no near-chance collapse.**

{{T:LRTOL calibration}}

**Table 9. Robustness summary, 30-seed LR sweeps.**

*MNIST, W = 64*

{{T:SWEEPSUM MNIST W=64}}

*MNIST, W = 128*

{{T:SWEEPSUM MNIST W=128}}

*CIFAR-10, W = 128*

{{T:SWEEPSUM CIFAR-10 W=128}}

![Figure 2](figures/fig2_lr_sweep_final_val.png)

*Figure 2. Mean final validation accuracy against learning rate (30 seeds per point).*

![Figure 3](figures/fig3_collapse_counts.png)

*Figure 3. Failed seeds per learning rate for ReLU and tanh+ReLU.*

**Table 10. The most extreme single condition: MNIST, W = 128, LR = 0.03 (30 seeds each).**

| Measure | ReLU | tanh+ReLU |
|---|---:|---:|
| Mean final validation accuracy | 10.61% | 96.76% |
| Mean best validation accuracy during training | 96.37% | 97.23% |
| Runs finishing below 15% | 30/30 | 0/30 |
| Paired Δ final val (hybrid − ReLU) | — | +86.2 pp, 30/30 wins |

The same pattern at W = 64: ReLU loses 12/30 runs at LR = 0.03, tanh+ReLU loses none (paired Δ final val +37.96 pp [+24.50, +51.62]).

**What "≈3× more robust" means precisely.** The data support the claim in two concrete forms:

| Form of the claim | Evidence | Factor |
|---|---|---|
| **3× higher tolerable learning rate** | MNIST calibration, W = 32, 64, 128: max LR within 2 pp of peak = 0.03 for tanh+ReLU vs 0.01 for ReLU (Table 8) | **3×** |
| **≈3× fewer near-chance collapses** | MNIST sweeps: 72 → 25 runs (W = 64), 91 → 33 runs (W = 128); MNIST calibration over all widths: 90 → 26 runs (Table 7) | **2.8×, 2.8×, 3.5×** |
| **≈3× fewer collapses at the critical LR on CIFAR-10** | CIFAR-10 W = 128, LR = 0.03: 26/30 → 9/30 (Table 7) | **2.9×** |

The factor depends on the failure definition and the dataset: with looser thresholds (final val < 50% or < 90%) the MNIST factor is 1.3–2.6×, and over the whole CIFAR-10 grid it is 1.2–1.3×. In the 30-seed MNIST sweeps leaky-ReLU and sigmoid never collapse to chance, but they pay for it differently: leaky-ReLU survives high learning rates by stalling at ≈85% accuracy (it loses useful accuracy in 120–150 of 330 runs, more than either ReLU or the hybrid), while sigmoid is the least accurate homogeneous model at calibrated LRs. **tanh+ReLU keeps ReLU-level (or better) calibrated accuracy while removing most of ReLU's catastrophic collapses on MNIST; the configurations that never collapse either stall at reduced accuracy at high learning rates (leaky-ReLU) or are less accurate at their optimum (sigmoid)** (Table 11).

**Table 11. Accuracy–robustness trade-off (calibrated checkpoint accuracy vs. behaviour across the LR sweep).**

{{T:PARETO2}}

### 3.4 Why the hybrid helps: evidence from the learning-rate axis

**Table 12. Same-LR paired comparison of final validation accuracy (hybrid − rectifier parent).**

{{T:SAMELR2}}

Table 12 reveals the mechanism behind the accuracy results of §3.2. **On MNIST, at small learning rates (≤ 0.005) the hybrid is slightly worse than ReLU** (−0.02 to −0.25 pp), as expected from replacing half of the rectifier units with saturating ones. **From LR = 0.01 upward the sign flips** (+0.96 pp at MNIST-64, 30/30 wins) because ReLU starts to degrade while the hybrid does not. Its best learning rate is therefore higher (0.01 vs 0.003–0.005 for ReLU on MNIST), and SGD at a larger step size reaches a slightly better solution within the 40-epoch budget. In other words, **the hybrid's accuracy advantage is a by-product of its robustness**: it can be trained at step sizes that destabilise homogeneous ReLU. On CIFAR-10 the picture is less clean: the hybrid is ahead at LR = 0.003 (+0.48 pp, 24/30) and LR = 0.03 (+6.98 pp, 26/30), but at LR = 0.01 it degrades more than ReLU (−9.41 pp, 0/30 wins), so on the harder task the stable region of the hybrid is not uniformly wider.

Mechanistic hypotheses consistent with these observations *(not directly measured; see §7)*:

- **Bounded signal path.** tanh units cap their output at |1|, so half of each layer cannot contribute unbounded activations to the next layer's preactivations. This limits the growth of preactivations after a large update, which is the usual trigger for ReLU units being pushed permanently negative ("dying ReLU") and for the network output collapsing to a constant class (≈10% accuracy).
- **Gradient path that never switches off.** tanh has a non-zero derivative everywhere, so even if many ReLU units in a layer die, error signals still reach earlier layers through the tanh half and the network can recover. ReLU runs in Table 10 reach 96.4% before collapsing, consistent with late-training unit death rather than failure to learn.
- **Complementary features.** The hybrid exceeding its parent mean (Table 6) at every width ≥ 64 suggests the two populations encode complementary information (sparse, piecewise-linear features from the rectifiers; smooth, sign-symmetric features from tanh), and that enough neurons of each type are needed for the complementarity to pay off, explaining the width threshold.

### 3.5 Seed-level stability, generalisation and per-class effects

**Table 13. Seed-to-seed variability and worst-case seeds (checkpoint accuracy, %).**

{{T:SEEDSTAB}}

**Table 14. Generalisation gap on CIFAR-10 (clean train accuracy at the selected epoch − test accuracy, pp).**

{{T:GAP cifar10}}

**Table 15. Generalisation gap on MNIST (pp).**

{{T:GAP mnist}}

**Table 16. Per-class F1, tanh+ReLU vs ReLU — MNIST, W = 128 (the width with the largest MNIST gain).**

{{T:CLASSMNIST}}

**Table 17. Per-class F1, tanh+ReLU vs ReLU — CIFAR-10, W = 64 and 256.**

{{T:CLASSC10}}

Observations:

- **More consistent runs on CIFAR-10.** tanh+ReLU has a lower seed SD than ReLU at all six CIFAR-10 widths (ratio 0.75–0.92), and its worst seed is better than ReLU's worst seed at W = 64, 128 and 256 (e.g. 50.91% vs 49.66% at W = 64). On MNIST the SDs are indistinguishable.
- **Less overfitting on CIFAR-10.** From W = 64 upward the tanh hybrids have a smaller train–test gap than ReLU (e.g. 7.23 vs 8.73 pp at W = 64; 12.23 vs 14.32 pp at W = 512), consistent with the bounded half of the layer acting as a mild regulariser.
- **The MNIST gain is broad, not class-specific.** At W = 128, tanh+ReLU improves F1 for all ten digits, with the interval above zero for nine of them; the largest gains are on digits 2, 4, 6 and 8 (+0.24 to +0.29 pp).
- **On CIFAR-10 the significant per-class gains are on confusable animal classes** — frog (W = 64, 128, 512), cat and dog (W = 256) — while vehicle classes are essentially unchanged; significant losses are isolated (bird at W = 128, horse at W = 512). *[Class-index order assumed to be the standard CIFAR-10 order; verify against the loader.]*

