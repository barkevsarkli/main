# Do hybrid activation functions improve accuracy and F1? — version5 findings

*Status: first-look sweep (≈1 h of compute), 4 September 2026. Numbers below come from `results/summary.md`, produced by `analysis/analyze.py` from the raw CSVs in `results/`.*

## The short answer

Under the corrected protocol, the earlier 6.5-point advantage of the `tanh/relu` hybrid over `relu/relu` (86.3% → 92.9%, 232 seeds, v2) **disappears**. It was a symptom of the homogeneous ReLU network being unstable under the v2 setup (MSE loss, ReLU also applied on the output layer, one fixed learning rate), not of the hybrid being a better function approximator.

What survives is a narrower but real claim, and one that is worth building the article around:

1. **Hybrid layers are markedly more robust to the learning rate.** Wherever the homogeneous ReLU network collapses (lr = 0.03 under MSE: 76.0%; lr = 0.1 under CE: 17.8%), the `tanh+relu` hybrid still trains normally (92.6% and 79.3%). This is the mechanism behind the original v2 result.
2. **At small width (12 neurons) the hybrid never beats the better parent.** With a tuned learning rate, `tanh+relu` is 0.46 points *below* `relu` (95% CI −0.67 to −0.24, 4/20 seeds win, paired t p = 0.0006) and statistically indistinguishable from `tanh` (−0.20, CI −0.43 to +0.04). Accuracy and macro-F1 agree in every comparison to within 0.02 points.
3. **At larger width (128 neurons) the picture reverses and the hybrid becomes competitive or better.** `tanh+leaky_relu` is the best configuration tested (97.66 ± 0.08) and beats `leaky_relu` on 6/6 seeds (+0.30, CI +0.19 to +0.43, p = 0.007). `tanh+relu` ties `tanh` (Δ = 0.00) and is ahead of `relu` on 5/6 seeds (+0.18, CI −0.04 to +0.38, not significant). Only 6 seeds, so this is a lead to confirm rather than a result to publish.

So the honest headline is: **hybrid activation is a stability/robustness device, and possibly a small accuracy gain at higher width — not a small-network accuracy booster.**

## What was wrong with the original comparison

| Issue in v2/v3 | Effect on the old result | Fix in version5 |
|---|---|---|
| Output layer was also hybridised (classes 0,2,4,… used `tanh`, classes 1,3,5,… used `relu`) | Confounds "hybrid hidden layers" with "hybrid output layer" | Output layer uniform: linear + softmax (CE) or sigmoid (MSE) |
| `relu/relu` had ReLU on the output layer with MSE loss | Dead output units → the 86.3% ± 8.3 mean with many collapsed seeds | Same fix as above |
| No `tanh/tanh` baseline | Could not tell "hybrid helps" from "tanh helps" | All four homogeneous baselines run |
| One learning rate (0.01) for everything | Favoured whichever function tolerates 0.01 | Per-config LR chosen on validation with separate calibration seeds |
| `leaky_relu` slope 0.1 (even) / 0.01 (odd) in forward, 0.1 in backward | `leaky/leaky` was neither homogeneous nor correctly differentiated | One constant slope (0.1) everywhere |
| Test set was the last 20% of the training file, no validation set | Selection and reporting on the same data | Fixed 45k/5k/10k train/val/test split, identical for every run |
| `rand()` seeding, no per-epoch shuffle | Platform-dependent, fixed sample order | `std::mt19937` for init + shuffle; seed means the same thing everywhere |
| Accuracy only | No F1 | Macro-F1 and confusion matrix on val (per epoch) and test |

## Protocol

MLP 784 → H → H → 10, batch-size-1 SGD, 10 epochs, Kaiming-uniform init, per-neuron activation masks in both hidden layers. Hybrids use the v2 even/odd layout (`interleave`, ratio 0.5) unless stated. The learning rate for each configuration was picked from a grid on validation accuracy using seeds 101–103; evaluation seeds are 1–20 (12 hidden) and 1–6 (128 hidden). Every configuration sees the same seeds and the same split, so all comparisons are paired. CIs are 95% bootstrap intervals of the paired mean difference (10,000 resamples); p-values are paired t-tests, with Wilcoxon signed-rank as a check.

## Results

### 12 hidden neurons, cross-entropy, 20 seeds

| config | lr | test acc | test macro-F1 |
|---|---|---|---|
| relu | 0.003 | 93.87 ± 0.36 | 93.81 ± 0.37 |
| tanh | 0.003 | 93.62 ± 0.36 | 93.56 ± 0.36 |
| leaky_relu | 0.003 | **93.97 ± 0.35** | **93.92 ± 0.35** |
| sigmoid | 0.01 | 93.20 ± 0.31 | 93.13 ± 0.32 |
| tanh+relu | 0.001 | 93.42 ± 0.44 | 93.35 ± 0.44 |
| tanh+leaky_relu | 0.003 | 93.57 ± 0.39 | 93.51 ± 0.39 |
| sigmoid+relu | 0.003 | 92.28 ± 0.36 | 92.20 ± 0.38 |

Paired differences (hybrid − parent, accuracy points): `tanh+relu` − `relu` = −0.46 [−0.67, −0.24]; `tanh+relu` − `tanh` = −0.20 [−0.43, +0.04]; `tanh+leaky_relu` − `leaky_relu` = −0.40 [−0.60, −0.20]; `tanh+leaky_relu` − `tanh` = −0.05 [−0.24, +0.17]; `sigmoid+relu` loses to both parents by 0.9–1.6 points. Every hybrid sits between its parents or slightly below the weaker one. F1 tracks accuracy exactly.

![](results/figures/fig1_acc_12.png)
![](results/figures/fig2_paired_acc_12.png)

### Layout and ratio (tanh+relu, 12 hidden, 20 seeds)

Block, random and interleaved layouts at ratio 0.5, and interleaved at ratios 0.25 and 0.75, all land within 93.42–93.59% — none reaches `relu` (93.87). Where the tanh neurons sit does not matter at this width; the ratio barely matters either. (The 0.25/0.75 variants reused the lr tuned for 0.5, so a small tuning penalty is possible.)

### Learning-rate sensitivity — the real effect

Validation accuracy from the calibration sweeps (mean of 3 seeds):

| 12 hidden, CE | lr 0.001 | 0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|---|
| relu | 93.28 | 93.51 | 93.29 | 88.72 | **17.78** |
| tanh | 93.23 | 93.33 | 92.41 | 90.79 | 77.79 |
| leaky_relu | 93.07 | 93.48 | 93.41 | 92.80 | 82.75 |
| tanh+relu | 93.14 | 92.93 | 92.01 | 89.69 | **79.31** |
| tanh+leaky_relu | 93.14 | 93.16 | 92.64 | 91.75 | 79.72 |

| 12 hidden, MSE + sigmoid output | lr 0.003 | 0.01 | 0.03 |
|---|---|---|---|
| relu | 92.61 | 92.21 | **75.96** |
| tanh | 91.64 | 92.05 | 90.89 |
| tanh+relu | 92.50 | 92.73 | **92.57** |

Under MSE — the v2 setting — `tanh+relu` is the *only* configuration whose accuracy is flat across the whole grid. Under CE at lr = 0.1, ReLU dies (17.8%, chance is 10%) while the hybrid is still at 79%. The interleaved tanh neurons keep every layer's gradient alive even when half the ReLU units are dead. This is the reproducible, mechanistically explainable version of the v2 observation ("hybrid: mean 92.9, sd 3.2; relu: mean 86.3, sd 8.3").

![](results/figures/fig3_lr_12_ce.png)
![](results/figures/fig3_lr_12_mse.png)

### Replication under the original loss (MSE, sigmoid output, 12 hidden, 20 seeds)

`relu` 93.13 ± 0.38, `tanh` 92.49 ± 0.50, `tanh+relu` 92.66 ± 0.51. Hybrid − relu = −0.47 [−0.72, −0.20]; hybrid − tanh = +0.17 [−0.13, +0.48], 13/20 wins, n.s. Same conclusion as with CE: once the output layer is uniform and the lr is tuned, the hybrid does not beat the better parent.

### 128 hidden neurons, cross-entropy, 6 seeds

| config | lr | test acc | test macro-F1 |
|---|---|---|---|
| relu | 0.003 | 97.38 ± 0.20 | 97.36 ± 0.20 |
| tanh | 0.01 | 97.56 ± 0.10 | 97.54 ± 0.10 |
| leaky_relu | 0.003 | 97.36 ± 0.13 | 97.33 ± 0.13 |
| tanh+relu | 0.01 | 97.56 ± 0.20 | 97.54 ± 0.20 |
| tanh+leaky_relu | 0.01 | **97.66 ± 0.08** | **97.64 ± 0.08** |

Paired: `tanh+leaky_relu` − `leaky_relu` = +0.30 [+0.19, +0.43], 6/6 wins, p = 0.007; `tanh+leaky_relu` − `tanh` = +0.10 [−0.01, +0.20], 4/6; `tanh+relu` − `relu` = +0.18 [−0.04, +0.38], 5/6; `tanh+relu` − `tanh` = 0.00. The hybrids also reach higher validation accuracy in the first epochs (95.1% vs 93.5% after epoch 1), though that is partly because their tuned lr is higher.

![](results/figures/fig6_acc_128.png)
![](results/figures/fig7_paired_acc_128.png)
![](results/figures/fig8_curves_128.png)

## Interpretation

At 12 neurons per layer every unit is precious, and a `tanh` unit is simply a slightly weaker feature detector than a ReLU unit on MNIST; mixing them costs a fraction of a point. At 128 neurons there is spare capacity, ReLU's dead-unit problem starts to bite (note `relu` and `leaky_relu` are the weakest configs there), and having half the units carry a bounded, always-differentiable non-linearity is worth a few tenths of a point. The learning-rate results say the same thing from another angle: the hybrid's advantage is in the failure modes of ReLU, not in its best case.

This is a more interesting story for an article than "hybrid = +6.5 points", because it comes with a mechanism, it survives proper controls, and it points directly at the phase-2 question: **which neurons should be `tanh`?** If the benefit is about keeping gradient flow alive where ReLU units would die, the assignment rule should follow neuron-level dead-ness / pre-activation statistics rather than the neuron's index.

## What is needed before this is publishable

- More seeds at 128 hidden (6 → 20–30) and a finer LR grid there (only {0.003, 0.01} × 1 seed was affordable in this pass).
- A width sweep (12, 32, 64, 128, 256) to show the crossover point directly, which would make figure 1 of the article.
- The official MNIST 10k test set instead of a held-out slice of the training file (the file is not in the repo and could not be downloaded from this environment).
- A learning-rate-robustness figure with more seeds and the 128-hidden case, since that is the strongest claim.
- Optionally a second dataset (Fashion-MNIST is drop-in for the loader; CIFAR-10 already has a loader in v4).

## Files

- `main.cpp`, `net.cpp`, `net.h`, `data.cpp`, `data.h` — corrected experiment code (`./main --help` style flags documented at the top of `main.cpp`)
- `sweep/` — job generation (`make_jobs.py`), LR selection (`select_lr.py`), resumable chunked runner (`run_chunk.sh`, `pending.py`), and the cloud pipeline used for 128 hidden (`run_128_cloud.sh`)
- `results/*.csv` — one row per run: config, per-epoch loss / train acc / val acc / val F1, final test accuracy, macro-F1 and the 10×10 confusion matrix
- `results/best_lr.json` — the learning rate selected for each configuration
- `analysis/analyze.py` → `results/summary.md` + `results/figures/`
