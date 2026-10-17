
---

## 4. Results II — Activation allocation ratio: 25%, 50% and 75% rectifier share

All ratio runs use MNIST, W = 128, LR = 0.01, `interleave` layout, and evaluation seeds 1–30. Because the dense LR sweep contains homogeneous tanh, ReLU and leaky-ReLU at exactly the same learning rate and seeds, the two homogeneous networks can be added as the ρ = 0% and ρ = 100% end points, giving a **five-point, same-LR ratio curve**. ρ always denotes the share of **rectifier** neurons (ReLU or leaky-ReLU); the remainder are tanh.

**Table 18. Allocation-ratio curve (MNIST, W = 128, LR = 0.01, 30 seeds per row).**

{{T:RATIO table (MNIST W=128, all at LR=0.01)}}

**Table 19. Paired accuracy differences between ratios (pp).**

{{T:RATIO paired}}

**Table 20. Paired macro-F1 differences between ratios (pp).**

{{T:RATIO paired F1}}

**Table 21. Each ratio against the homogeneous baselines at their own calibrated learning rates (main grid: ReLU and leaky-ReLU at 0.003, tanh at 0.01). Entries: Δ acc [95% CI] (wins).**

{{T:RATIO vs calibrated baselines}}

![Figure 4](figures/fig4_ratio_curve.png)

*Figure 4. Accuracy as a function of the rectifier share ρ at a fixed learning rate.*

**Summary of the ratio comparison.**

| Question | tanh+ReLU | tanh+leaky-ReLU |
|---|---|---|
| Best ratio (mean accuracy) | ρ = 75% (98.052%), statistically tied with ρ = 50% (98.038%) | ρ = 75% (98.052%) |
| 25% vs 0% (adding a quarter rectifiers to tanh) | +0.131 pp, 25/30 wins | +0.138 pp, 28/30 wins |
| 50% vs 25% | +0.081 pp, 19/30 wins (CI excludes 0) | +0.021 pp, n.s. |
| 75% vs 50% | +0.014 pp, n.s. | **+0.066 pp, 20/30 wins (CI excludes 0)** |
| 100% vs 75% (removing the last tanh units) | −0.043 pp, 11/30 wins (p = 0.106) | **−0.359 pp, 2/30 wins** |
| Beats both homogeneous extremes at the same LR? | ρ = 50% and 75% nominally yes (vs ReLU: +0.029, +0.043 pp; CIs include 0) | **Yes at all three ratios** (vs leaky-ReLU: +0.27 to +0.36 pp, 27–28/30 wins) |
| Beats calibrated homogeneous baselines? | Yes at all ratios (+0.10 to +0.19 pp vs ReLU@0.003) | Yes at all ratios (+0.12 to +0.20 pp vs leaky-ReLU@0.003) |
| Lowest seed SD | ρ = 50% (0.073) | ρ = 50% (0.081) |
| Smallest peak − final drop | ρ = 75% (0.041 pp) vs ReLU 0.082 | ρ = 50% (0.049 pp) vs leaky-ReLU 0.394 |
| Train − test gap | falls from 2.17 (tanh) to 1.95 (ρ = 75%) | falls from 2.17 (tanh) to 1.95 (ρ = 75%) |

Patterns:

1. **The curve is concave with an interior optimum.** Moving from pure tanh toward rectifiers improves accuracy steeply at first (+0.13 pp for the first 25%), then flattens between 50% and 75%, and then either flattens (ReLU) or drops sharply (leaky-ReLU) at 100%. The best networks are mixed ones, not homogeneous ones.
2. **A small tanh minority is enough to stabilise the rectifiers.** The 75% network keeps only 32 tanh neurons per layer, yet it has half of ReLU's peak-to-final drop and a seven-fold smaller drop than leaky-ReLU at this learning rate. Pure leaky-ReLU at LR = 0.01 is already past its stable region (final test accuracy 97.49% vs checkpoint 97.69%), and inserting 25% tanh neurons removes the problem.
3. **Rectifier-majority hybrids are the recommended default.** ρ = 75% is the best or tied-best ratio for both pairs; ρ = 25% is significantly worse than ρ = 75% for both pairs (−0.095 and −0.087 pp, 23/30 wins each) and is the only ratio that falls below pure ReLU at the same LR (−0.051 pp, 8/30).
4. **Caveat:** ratios 0.25 and 0.75 inherit the learning rate calibrated for ρ = 0.5; each ratio has not been tuned separately, and all ratio results come from one dataset and width.

### 4.1 Neuron placement (layout) at ρ = 50%

**Table 22. Layout comparison (MNIST, W = 128, LR = 0.01, ρ = 0.5).**

{{T:LAYOUT}}

Placement does not matter: block, random and interleave arrangements are within ±0.04 pp of one another and no interval clearly excludes zero (the random-layout tanh+leaky-ReLU interval touches zero, p = 0.075). This is expected for a fully connected layer, where permuting neurons (with the matching weight permutation) leaves the function unchanged; what matters is **how many** neurons of each type a layer has, i.e. ρ, not **where** they are. (Random masks also consume the RNG stream used for initialisation, so seeds are not perfectly matched in the random-layout rows.)

---

## 5. Results III — Additional checks

### 5.1 Replication under MSE loss

**Table 23. MSE runs (MNIST, sigmoid output layer, 30 seeds).**

{{T:MSE}}

**Table 24. MSE calibration grids (mean final validation accuracy over 6 calibration seeds; last column = collapsed runs).**

*W = 64*

{{T:MSECAL W=64}}

*W = 128*

{{T:MSECAL W=128}}

Under MSE the hybrid still beats tanh by a wide margin (+0.17 and +0.19 pp, 25/30 and 29/30 wins) and still roughly halves ReLU's calibration collapses at W = 64 (5 vs 11), but it **does not beat ReLU** (−0.20 pp at W = 64, −0.03 pp n.s. at W = 128). Cross-entropy is the better objective for the hybrid (+0.15 pp and +0.07 pp CE − MSE), whereas ReLU does equally well or slightly better with MSE (−0.06 pp n.s. at W = 64, −0.14 pp at W = 128). The accuracy advantage over ReLU is therefore specific to the cross-entropy setting. *(Implementation note: the MSE gradient is not divided by the number of outputs, so MSE updates are ten times larger than the logged objective implies; this should be fixed or documented before publication.)*

### 5.2 Sensitivity to the calibration criterion

Learning rates were selected by **final** validation accuracy, while the reported endpoint is the **best** checkpoint. Re-selecting by mean best validation accuracy changes 17 of 84 main-grid choices (Table E-3). Where 30-seed evaluation data exist at the re-selected LR, the comparisons can be recomputed:

**Table 25. Hybrid − parent under the original and aligned LR-selection criteria, plus an optimistic dense-sweep selection.**

{{T:ALIGN}}

**Table 26. Best checkpoint test accuracy reached anywhere on the dense LR sweep (descriptive upper envelope; selected on the test metric).**

{{T:ENVELOPE best test acc over sweep (descriptive, test-selected)}}

The robustness and tanh comparisons are unaffected by the criterion. The MNIST-128 accuracy gain over ReLU is the sensitive result: aligned selection moves ReLU to LR = 0.01 and the gain shrinks from +0.178 to +0.029 pp (n.s.); selecting each model's best LR on the dense 11-point sweep gives +0.046 pp [+0.018, +0.075] (22/30 wins, p = 0.004). With a dense sweep, however, ReLU at W = 64 is 0.107 pp ahead of the hybrid. **The accuracy advantage over a carefully tuned ReLU should therefore be presented as "matches or slightly exceeds", while the robustness advantage is independent of the tuning protocol.**

### 5.3 Convergence speed

**Table 27. Validation accuracy after epoch 1 and epochs needed to come within a tolerance of each run's own best validation accuracy.**

{{T:CONV2}}

On MNIST at W = 32–128 the hybrids start faster than ReLU (epoch-1 validation accuracy up to +1.5 pp at W = 64–128) and reach within 0.5 pp of their best in a similar number of epochs (7.6–11.6 vs 8.7–10.2 for ReLU). On CIFAR-10 all configurations are still improving late in training (best epochs ≈ 30–38 of 40), so convergence speed cannot be ranked reliably with this budget.

---

## 6. Discussion — cross-cutting patterns

| # | Pattern | Supporting tables |
|---|---|---|
| P1 | **Width threshold.** Tanh hybrids trail homogeneous rectifiers at W ≤ 32 and match or beat them from W = 64 on both datasets. | 1, 3, 5, Fig. 1 |
| P2 | **Complementarity.** At W ≥ 64 both tanh hybrids exceed the mean of their parents in 16/16 dataset–width cells. | 6 |
| P3 | **Tanh is always improved by adding rectifiers**, and the gain grows with width on CIFAR-10 (+0.6 → +1.7 pp). | 4, 5, 19 |
| P4 | **ReLU fails catastrophically at high LR; the hybrid mostly does not.** 2.8–3.5× fewer near-chance collapses on MNIST, 3× higher tolerable LR at W = 32–128. | 7–10, Fig. 2–3 |
| P5 | **Accuracy gains follow from robustness.** Below LR ≈ 0.005 the hybrid is slightly worse than ReLU; its advantage appears at the higher learning rates it tolerates. | 12 |
| P6 | **Robustness transfers only partly to CIFAR-10.** Large reduction at LR = 0.03 (2.9×), but at LR = 0.01 the hybrid degrades more than ReLU, and the whole-grid factor is 1.2–1.3×. | 7, 9, 12 |
| P7 | **Lower variance and smaller generalisation gap on CIFAR-10.** | 13, 14 |
| P8 | **Interior optimum of the allocation ratio** at ρ = 50–75% rectifiers; homogeneous extremes are never the best. | 18–21, Fig. 4 |
| P9 | **Neuron placement is irrelevant**; the count of each neuron type is what matters. | 22 |
| P10 | **Bounded partner matters.** tanh works as the bounded partner; sigmoid (non-zero-centred, derivative ≤ 1/4) does not. | 1, 4, 6 |
| P11 | **Objective matters.** The advantage over ReLU is specific to cross-entropy; under MSE the hybrid only beats tanh. | 23 |

*[Discussion to be expanded: relation to adaptive and mixed activation literature; practical recommendation — use a rectifier-majority tanh hybrid (ρ ≈ 0.75) when the learning rate cannot be tuned carefully or when training at larger step sizes is desirable.]*

---

## 7. Limitations and threats to validity

| Limitation | Consequence for the claims | Suggested remedy |
|---|---|---|
| LR selected by final validation accuracy while the best checkpoint is reported | MNIST-128 gain over ReLU shrinks to +0.03 pp under aligned selection (§5.2) | Re-calibrate on `best_val_acc`, evaluate on fresh seeds |
| Ratios 0.25 / 0.75 not separately calibrated; ratio study only at MNIST-128, one LR | Ratio optimum is conditional on LR = 0.01 | Ratio × LR grid on both datasets |
| Robustness measured only as LR tolerance | No evidence about input noise, distribution shift, adversarial or pruning robustness | Report "learning-rate robustness / training stability", not general robustness |
| No activation, gradient-norm or dead-unit logging | Mechanisms in §3.4 are hypotheses | Log inactive-unit fraction, gradient norms, clipping frequency |
| Collapse factor depends on threshold and dataset | "≈3×" holds for near-chance collapse on MNIST and at LR = 0.03 on CIFAR-10, not universally | State the definition alongside every factor |
| Initialisation U(±1/√fan_in) (variance 1/(3·fan_in)), identical for all activations; derivative clipping at ±10 | Baselines may not be individually optimal; results may change under He/Glorot initialisation | Factorial initialisation × activation study |
| MNIST test set is a held-out slice of the training file | MNIST accuracies are not comparable to official-test-set numbers | Replicate on official test set |
| 30 seeds on a single split; multiple comparisons | CIs reflect training stochasticity only; only the 72 main tests are Holm-corrected | Additional splits, pre-registered primary contrast |
| 40-epoch budget; CIFAR-10 still improving | CIFAR-10 accuracies right-censored; MLP far from state of the art | Longer schedules; CNN reference |
| MSE gradient scaling mismatch (factor 10) | MSE LR values are not directly comparable to CE | Fix or document |
| Timing logged under ~190 concurrent workers | Seconds are not clean cost measurements | Isolated timing |
| CIFAR-10 class order assumed standard for per-class tables | Class names in Tables 16–17 / F-1 need verification | Check loader |

---

## 8. Conclusion

Splitting each hidden layer of an MLP between tanh and rectifier neurons costs no parameters and, once the layer has at least ≈64 neurons, gives accuracy that matches or slightly exceeds the best homogeneous network while making training far more tolerant of large learning rates: on MNIST the tanh+ReLU hybrid tolerates a 3× larger step size and suffers roughly 3× fewer catastrophic collapses than ReLU, and on CIFAR-10 it cuts collapses at the critical learning rate by the same factor. The allocation ratio behaves like a genuine design parameter with an interior optimum: rectifier-majority layers (ρ ≈ 50–75%) outperform both pure tanh and pure rectifier layers at the same learning rate, whereas the spatial arrangement of the neurons is irrelevant. The benefits depend on width, on tanh (not sigmoid) as the bounded partner, and on the cross-entropy objective, and the robustness mechanism remains to be verified with direct measurements of unit activity and gradients.

---

# Appendices

## Appendix A — Detailed main-grid tables (checkpoint endpoint)

Columns: calibrated LR; checkpoint accuracy and macro-F1 (mean ± SD); worst seed; final-epoch accuracy; mean selected epoch; train−test gap; mean peak−final validation drop; mean logged run time (seconds, under heavy concurrent load).

### A.1 MNIST

{{DETAILS mnist}}

### A.2 CIFAR-10

{{DETAILS cifar10}}

## Appendix B — Final-epoch endpoint

**Table B-1. MNIST, final-epoch test accuracy (%), mean ± SD.**

{{T:FINALWIDE mnist}}

**Table B-2. CIFAR-10, final-epoch test accuracy (%).**

{{T:FINALWIDE cifar10}}

**Table B-3. MNIST, paired Δ final-epoch accuracy (pp) [95% CI].**

{{T:FINALPAIRED mnist}}

**Table B-4. CIFAR-10, paired Δ final-epoch accuracy (pp) [95% CI].**

{{T:FINALPAIRED cifar10}}

## Appendix C — All paired checkpoint comparisons with F1 and Holm correction

**Table C-1. MNIST.**

{{T:PAIRED mnist}}

**Table C-2. CIFAR-10.**

{{T:PAIRED cifar10}}

## Appendix D — Full 30-seed learning-rate sweeps

{{SWEEPS}}

## Appendix E — Calibration grids (6 calibration seeds)

Cells show mean final validation accuracy (%); "(k✗)" = k of 6 seeds finished below 15%.

{{CALGRIDS}}

**Table E-1. MNIST, LR tolerance and collapse counts derived from the calibration grids.**

{{T:CAL mnist largest LR (≥ own best LR) keeping mean final val within 2 pts of peak}}

{{T:CAL mnist largest LR (≥ own best LR) with no calibration seed below 15%}}

{{T:CAL mnist near-chance collapsed calibration runs (of 48 per cell)}}

**Table E-2. CIFAR-10, LR tolerance and collapse counts derived from the calibration grids.**

{{T:CAL cifar10 largest LR (≥ own best LR) keeping mean final val within 2 pts of peak}}

{{T:CAL cifar10 largest LR (≥ own best LR) with no calibration seed below 15%}}

{{T:CAL cifar10 near-chance collapsed calibration runs (of 48 per cell)}}

**Table E-3. Calibration choices that change when selecting by best instead of final validation accuracy.**

{{T:ALIGNCHANGES}}

## Appendix F — Per-class F1, tanh+ReLU vs ReLU (all computed widths)

{{T:CLASS}}

## Appendix G — Reproducibility

All tables and Figures 1–4 were produced by the scripts in `report_draft/analysis_scripts/` (Python 3, pandas, numpy, matplotlib) directly from the CSV files in `results_aws/`. Run order: `build.py` → `a_main.py` → `b_rob.py` → `c_more.py` → `d_ratio.py` → `e_misc.py` → `f_class.py` → `g_align.py` → `h_tables.py` → `i_thr.py` → `fig.py` → `assemble.py`. Bootstrap resampling uses a fixed generator seed (12345). Earlier figures in `results_aws/figures/` (e.g. `fig_crossover.png`, `fig_lrsweep_128.png`) are based on final-epoch scores and can be used as supplementary material.
