# Hybrid Activations in Multilayer Perceptrons: Learning-Rate Stability, Predictive Accuracy, and Activation Allocation

English manuscript draft · 15 September 2026

Analysis of the version5 AWS results. All experimental numbers below were recomputed from the 32 original CSV files. This draft reports completed experiments; mechanism explanations are identified as hypotheses. The accompanying Supplementary Tables provide the complete main grid and calibration grids.

## Abstract

We examine whether assigning different activation functions to neurons within the same hidden layer improves predictive accuracy and training stability. The experiments use two-hidden-layer multilayer perceptrons on MNIST and CIFAR-10, six hidden widths, seven activation configurations, six calibration seeds, and thirty evaluation seeds per condition. The strongest benefit is conditional resistance to late-training deterioration. On MNIST at width 128 and learning rate 0.03, all 30 ReLU runs finish below 15% validation accuracy, whereas none of the tanh–ReLU runs does; mean final validation accuracy is 10.611% and 96.761%, respectively. However, this benefit does not imply consistently higher optimally tuned accuracy. Aligning learning-rate selection with the validation-selected checkpoint reduces the width-128 MNIST accuracy difference from +0.178 to +0.029 percentage points, with a paired 95% bootstrap interval spanning zero. The frequently suggested threefold robustness advantage describes a particular upper learning-rate comparison on a coarse grid, rather than a general robustness multiplier. At fixed learning rate 0.01 and width 128, increasing the rectifier allocation from 25% to 75% improves mean MNIST checkpoint accuracy in both tanh-based hybrids. Increasing it from 50% to 75% gives only +0.014 points for tanh–ReLU, compared with +0.066 points for tanh–leaky-ReLU. CIFAR-10 and MSE experiments reveal important limits to transfer. These results support a study of activation-dependent optimization stability and allocation trade-offs, rather than an unconditional claim of hybrid superiority.

## 1. Experimental questions and the observed hybrid benefit

The central question is whether heterogeneous activation assignment changes the learning-rate range over which useful performance can be maintained. Accuracy after model selection, accuracy at the end of training, and resistance to a large learning rate answer different questions. We therefore report them separately. ReLU is the principal homogeneous reference, while tanh, sigmoid, and leaky-ReLU provide additional controls. A hybrid must be evaluated against both of its homogeneous parents to establish what mixing contributes.

**Table 1. Questions addressed by the completed tests**

| Test | Available design | What it establishes |
| --- | --- | --- |
| Main accuracy comparison | 2 datasets × 6 widths × 7 configurations × 30 seeds | Performance at the originally calibrated LR |
| LR calibration | 8 LRs × 6 seeds per configuration and width | Validation-based hyperparameter selection |
| Evaluation LR sweep | MNIST H=64/128; CIFAR-10 H=128; 30 seeds per LR | Sensitivity and endpoint deterioration on the tested grid |
| Allocation ratio | MNIST H=128; 25%, 50%, 75% rectifier; LR=0.01 | Ratio effect at one common LR |
| Mask layout | Interleave, block, random at 50% | Sensitivity to activation assignment and randomization |
| Loss/output ablation | MNIST H=64/128; CE/softmax and MSE/sigmoid | Transfer across the implemented loss/output formulations |

### 1.1 A substantial but conditional stability advantage

The clearest result appears at MNIST width 128 and LR=0.03. ReLU achieves high validation accuracy during training and subsequently loses it. Tanh–ReLU retains high final accuracy. The behavior is also visible at width 64, although fewer ReLU seeds fail. “Calibration” in these files refers to learning-rate tuning. Predictive-probability calibration, such as reliability diagrams, expected calibration error, Brier score, or negative log-likelihood, was not measured in a form that supports such a comparison.

**Table 2. MNIST stability at LR=0.03**

| H | Activation | Final validation % ± SD | Peak validation % | Peak-to-final drop, pp | Final <15% |
| --- | --- | --- | --- | --- | --- |
| 64 | ReLU | 58.431 ± 40.047 | 95.891 | 37.460 | 12/30 |
| 64 | Leaky-ReLU | 84.005 ± 2.889 | 96.629 | 12.625 | 0/30 |
| 64 | tanh | 95.907 ± 0.303 | 96.398 | 0.491 | 0/30 |
| 64 | tanh + ReLU | 96.393 ± 0.302 | 96.861 | 0.468 | 0/30 |
| 64 | tanh + leaky-ReLU | 95.689 ± 0.506 | 96.539 | 0.849 | 0/30 |
| 128 | ReLU | 10.611 ± 0.911 | 96.371 | 85.760 | 30/30 |
| 128 | Leaky-ReLU | 85.228 ± 2.276 | 96.365 | 11.137 | 0/30 |
| 128 | tanh | 95.111 ± 0.580 | 95.862 | 0.751 | 0/30 |
| 128 | tanh + ReLU | 96.761 ± 0.258 | 97.230 | 0.469 | 0/30 |
| 128 | tanh + leaky-ReLU | 84.633 ± 2.122 | 96.357 | 11.724 | 0/30 |

Source: [lrsweep_64.csv](/Users/khaos/Desktop/article/version5/results_aws/lrsweep_64.csv) and [lrsweep_128.csv](/Users/khaos/Desktop/article/version5/results_aws/lrsweep_128.csv). Peak is the mean of each run’s maximum validation accuracy. Drops are computed within each run before averaging. The 15% cutoff is an exploratory near-chance diagnostic for a ten-class task.

At H=128, the mean peak-to-final decline is 85.760 points for ReLU and only 0.469 points for tanh–ReLU. Leaky-ReLU also avoids near-chance failure, but its final validation accuracy is 85.228%, substantially below tanh–ReLU. Thus avoiding catastrophic failure and preserving high accuracy are distinct properties. The hybrid with leaky-ReLU does not reproduce the tanh–ReLU high-performance result at this width.

**Table 3. Timing and prediction concentration at MNIST LR=0.03**

| H | Activation | Seeds ever below 15% | First such epoch: min / median / max | Mean largest predicted-class share, % |
| --- | --- | --- | --- | --- |
| 64 | ReLU | 12 | 19 / 27 / 40 | 48.9 |
| 64 | tanh + ReLU | 0 | Not observed | 11.5 |
| 128 | ReLU | 30 | 17 / 21.5 / 28 | 100.0 |
| 128 | tanh + ReLU | 0 | Not observed | 11.4 |

The first crossing is observed at epoch resolution. Prediction concentration uses final test confusion matrices and is descriptive; it does not identify a causal failure mechanism.

At width 128, ReLU first crosses below 15% between epochs 17 and 28, with a median of 21.5 among the affected runs. This temporal pattern directly supports late-training failure. Zero observed failures among 30 hybrid seeds still allows a one-sided 95% binomial upper bound of approximately 9.5% for the failure probability under independent seed sampling.

### 1.2 What could explain the benefit?

The model assigns one activation function to each hidden neuron. A tanh–ReLU layer can therefore contain bounded tanh outputs alongside non-saturating positive ReLU outputs. This combination could alter the propagation of large activations and gradients. The experiment establishes the behavior, but does not measure the internal quantities needed to establish why it occurs.

**Table 4. Mechanistic interpretation and evidence limits**

| Candidate explanation | Why it is plausible | What the current results establish | Missing direct measurement |
| --- | --- | --- | --- |
| Bounded tanh outputs limit some forward signals | tanh outputs lie between −1 and 1 | Tanh–ReLU preserves high MNIST accuracy at selected high LRs | Activation distributions and layerwise magnitudes |
| Positive rectifier paths retain useful gradients | ReLU derivative is 1 for positive inputs | Increasing rectifier allocation can improve accuracy at LR=0.01 | Gradient norms and feature-level contribution |
| Negative-side gradient affects recovery | Leaky-ReLU has negative-side slope 0.1 | Leaky variants avoid chance-level MNIST collapse but may lose accuracy | Inactive-unit counts and recovery trajectories |
| Initialization and clipping interact with activation | All variants share the implemented initialization and derivative clipping | The observed effect is conditional on this implementation | Factorial initialization × clipping × activation controls |
| Limited width constrains useful mixed features | At small widths, each activation receives few neurons | Tanh–ReLU underperforms ReLU at H=12 and H=32 on both datasets | Matched representation or feature-utilization analysis |

Prior work motivates examining saturation, activation statistics, and initialization, but does not prove a causal explanation for these particular hybrids. See [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html), [Glorot, Bordes, and Bengio (2011)](https://proceedings.mlr.press/v15/glorot11a.html), and [He et al. (2015)](https://arxiv.org/abs/1502.01852). These are background references, not a completed novelty review.

The source clips each preactivation derivative to ±10. A weight update still has the form ΔW = −η·clip(∂L/∂z)·h, so a large preceding activation can produce a large update despite clipping. This observation motivates measuring internal dynamics; it is not evidence that gradient explosion caused the recorded failures.

## 2. Materials, protocol, and statistical analysis

**Table 5. Implemented experimental protocol**

| Item | Specification |
| --- | --- |
| Architecture | D → H → H → 10; D=784 for MNIST and 3072 for CIFAR-10 |
| Widths | H ∈ {12, 32, 64, 128, 256, 512} per hidden layer |
| Main hidden activations | Four homogeneous activations and three 50:50 hybrids |
| Training | Single-example SGD, fixed LR, 40 completed epochs; shuffled training order each epoch |
| Model selection | First epoch attaining maximum validation accuracy; selected weights evaluated after training |
| Calibration seeds | 101–106; LR selected using mean final validation accuracy in the original pipeline |
| Evaluation seeds | 1–30; matched seed identifiers within each comparison |
| Initialization | Weights U(−1/√fan_in, +1/√fan_in), zero biases |
| Clipping and slope | Preactivation derivatives clipped to ±10; leaky-ReLU slope 0.1 |
| CE output | Linear logits followed by softmax |
| MSE output | Sigmoid outputs; implemented gradient scale differs from logged mean loss |
| Inputs | Pixel scaling to [0,1] in the adjacent dataset loaders |
| Training diagnostic | Clean accuracy on a fixed 10,000-image subset of the training split |

Source-level details were checked in [../main.cpp](/Users/khaos/Desktop/article/version5/main.cpp), [../net.cpp](/Users/khaos/Desktop/article/version5/net.cpp), and [../sweep/select_lr.py](/Users/khaos/Desktop/article/version5/sweep/select_lr.py). The AWS executable was not rebuilt and matched to these sources. “Checkpoint selection” describes the computation more accurately than “early stopping,” since all 40 epochs were executed.

**Table 6. Dataset provenance**

| Dataset | Train | Validation | Test | Test provenance |
| --- | --- | --- | --- | --- |
| MNIST | 45,000 | 5,000 | 10,000 | Held out from the 60,000-image training file |
| CIFAR-10 | 45,000 | 5,000 | 10,000 | Official CIFAR-10 test batch |

The split seed is 12345. MNIST scores in this draft must not be described as official MNIST test-set results.

**Table 7. Run inventory after removing repeated CSV headers**

| Stage | Recorded rows | Files | Datasets | Widths |
| --- | --- | --- | --- | --- |
| main | 2520 | 12 | CIFAR-10, MNIST | 12, 32, 64, 128, 256, 512 |
| calib | 4032 | 12 | CIFAR-10, MNIST | 12, 32, 64, 128, 256, 512 |
| lrsweep | 6300 | 3 | CIFAR-10, MNIST | 64, 128 |
| layout | 300 | 1 | MNIST | 128 |
| mse | 180 | 2 | MNIST | 64, 128 |
| calib_mse | 288 | 2 | MNIST | 64, 128 |

Total: 13,620 data rows in 32 CSV files. Thirty-one repeated headers were excluded. There are 690 repeated job identities across stages; they were not pooled as independent replicates. Main-grid sample size remains 30 per cell, not 60 when a sweep repeats a job.

For all 13,620 records, both final and checkpoint confusion matrices reproduce the corresponding accuracy and macro-F1 to CSV precision. Validation histories have 40 entries and reproduce the saved maximum and selected epoch. No within-file duplicate job identities were found, and every recorded status is `ok`. These checks establish consistency of the exported records; the status field alone is not a complete numerical-health check.

Unless explicitly labeled final, reported test accuracy and macro-F1 use `test_acc_best` and `test_f1_best`. Means and sample standard deviations describe seed variation. Differences use matched seeds, with 10,000 bootstrap resamples of the 30 complete pairs and percentile 95% confidence intervals. A fixed bootstrap RNG seed of 20260915 makes the current analysis reproducible; interval endpoints can differ slightly from earlier audit files. Paired t-test p-values are supplied with a Holm correction across 120 main-grid checkpoint-accuracy comparisons. A separate six-comparison correction is provided for ratio accuracy contrasts; it does not cover the entire adaptive experiment search.

Accuracy differences are percentage points (pp). Relative error reduction is 100·(A_hybrid − A_baseline)/(100 − A_baseline), where A is accuracy in percent. All inferential results are conditional on one fixed split and the observed tuning procedure. The 30 training seeds are not 30 independent test datasets.

## 3. Predictive accuracy and calibration sensitivity

### 3.1 Complete width trends

The following matrices use the original learning-rate selections but consistently report validation-selected checkpoint accuracy. Complete accuracy, macro-F1, endpoint, timing, and training-gap tables for all 84 main conditions appear in the Supplementary Tables.

**Table 8. MNIST checkpoint test accuracy, mean ± seed SD (%)**

| Activation | H=12 | H=32 | H=64 | H=128 | H=256 | H=512 |
| --- | --- | --- | --- | --- | --- | --- |
| ReLU | 94.297 ± 0.303 | 96.707 ± 0.190 | 97.579 ± 0.125 | 97.860 ± 0.075 | 98.214 ± 0.074 | 98.285 ± 0.064 |
| Leaky-ReLU | 94.274 ± 0.226 | 96.683 ± 0.189 | 97.560 ± 0.107 | 97.849 ± 0.082 | 98.176 ± 0.063 | 98.248 ± 0.091 |
| tanh | 93.952 ± 0.260 | 96.375 ± 0.160 | 97.390 ± 0.122 | 97.827 ± 0.094 | 98.018 ± 0.073 | 98.024 ± 0.068 |
| sigmoid | 93.770 ± 0.331 | 96.269 ± 0.138 | 97.123 ± 0.083 | 97.658 ± 0.095 | 97.966 ± 0.069 | 98.042 ± 0.071 |
| tanh + ReLU | 93.990 ± 0.250 | 96.521 ± 0.185 | 97.583 ± 0.126 | 98.038 ± 0.073 | 98.214 ± 0.075 | 98.289 ± 0.087 |
| tanh + leaky-ReLU | 94.021 ± 0.316 | 96.513 ± 0.159 | 97.607 ± 0.116 | 97.986 ± 0.081 | 98.194 ± 0.071 | 98.274 ± 0.076 |
| sigmoid + ReLU | 93.772 ± 0.389 | 95.989 ± 0.174 | 96.709 ± 0.159 | 97.550 ± 0.141 | 98.037 ± 0.083 | 98.116 ± 0.070 |

Each cell contains 30 seeds. Main hybrids use a 50% allocation to the second activation.

**Table 9. CIFAR-10 checkpoint test accuracy, mean ± seed SD (%)**

| Activation | H=12 | H=32 | H=64 | H=128 | H=256 | H=512 |
| --- | --- | --- | --- | --- | --- | --- |
| ReLU | 43.461 ± 0.745 | 49.186 ± 0.588 | 51.239 ± 0.504 | 53.043 ± 0.527 | 54.074 ± 0.506 | 54.875 ± 0.469 |
| Leaky-ReLU | 43.618 ± 0.880 | 49.254 ± 0.490 | 51.270 ± 0.491 | 52.953 ± 0.512 | 53.852 ± 0.482 | 54.607 ± 0.424 |
| tanh | 42.343 ± 0.562 | 47.894 ± 0.476 | 50.490 ± 0.389 | 51.930 ± 0.517 | 52.489 ± 0.496 | 53.190 ± 0.487 |
| sigmoid | 41.360 ± 0.381 | 47.206 ± 0.545 | 50.013 ± 0.357 | 51.661 ± 0.463 | 52.777 ± 0.534 | 53.137 ± 0.530 |
| tanh + ReLU | 42.911 ± 0.647 | 48.801 ± 0.452 | 51.603 ± 0.376 | 53.221 ± 0.439 | 54.227 ± 0.459 | 54.711 ± 0.430 |
| tanh + leaky-ReLU | 42.987 ± 0.756 | 48.759 ± 0.448 | 51.288 ± 0.439 | 53.086 ± 0.521 | 53.983 ± 0.405 | 54.474 ± 0.372 |
| sigmoid + ReLU | 42.340 ± 0.861 | 47.774 ± 0.612 | 50.274 ± 0.427 | 51.902 ± 0.453 | 53.481 ± 0.500 | 54.357 ± 0.513 |

Each cell contains 30 seeds. Main hybrids use a 50% allocation to the second activation.

**Table 10. Paired tanh–ReLU minus ReLU accuracy at original LRs**

| Dataset | H | Δ, pp | 95% paired CI | Wins | Raw p | Holm p (120) |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 12 | -0.307 | [-0.449, -0.160] | 8/30 | 3.60e-04 | 0.020 |
| MNIST | 32 | -0.186 | [-0.281, -0.087] | 7/30 | 8.20e-04 | 0.042 |
| MNIST | 64 | +0.004 | [-0.058, +0.067] | 15/30 | 0.910 | 1.000 |
| MNIST | 128 | +0.178 | [+0.143, +0.213] | 28/30 | 1.44e-10 | 1.29e-08 |
| MNIST | 256 | -0.001 | [-0.035, +0.034] | 14/30 | 0.971 | 1.000 |
| MNIST | 512 | +0.003 | [-0.036, +0.042] | 15/30 | 0.869 | 1.000 |
| CIFAR-10 | 12 | -0.550 | [-0.868, -0.241] | 11/30 | 0.002 | 0.085 |
| CIFAR-10 | 32 | -0.385 | [-0.616, -0.145] | 10/30 | 0.004 | 0.167 |
| CIFAR-10 | 64 | +0.363 | [+0.156, +0.577] | 22/30 | 0.002 | 0.095 |
| CIFAR-10 | 128 | +0.178 | [-0.082, +0.425] | 17/30 | 0.186 | 1.000 |
| CIFAR-10 | 256 | +0.153 | [-0.049, +0.366] | 16/30 | 0.156 | 1.000 |
| CIFAR-10 | 512 | -0.164 | [-0.404, +0.059] | 13/30 | 0.186 | 1.000 |

These pointwise CIs are unadjusted; the Holm column adjusts the paired t-test family. A positive mean or nominal CI does not by itself establish a robust global advantage.

Three patterns are consistent across the main grid. First, tanh–ReLU performs worse than ReLU at the two smallest widths on both datasets. Second, the effect does not grow monotonically with width: MNIST differences nearly vanish at widths 256 and 512, while CIFAR-10 changes sign again at width 512. Third, sigmoid–ReLU underperforms ReLU at every main-grid width on both datasets. Heterogeneity alone is therefore insufficient to explain improved performance.

The nominal highest MNIST mean is 98.289% for tanh–ReLU at H=512, compared with 98.285% for ReLU. CIFAR-10 instead has its highest mean with ReLU at H=512, 54.875%, compared with 54.711% for tanh–ReLU. These are descriptive ranks under the original calibration rule, not uniquely established winners. On CIFAR-10 H=64, the positive tanh–ReLU difference has a nominal paired interval above zero, but its Holm-adjusted p-value is approximately 0.095.

### 3.2 The reporting endpoint changes the apparent result

The earlier `summary_extended.md` describes checkpoint selection but tabulates final-epoch test fields. This changes some conclusions, particularly on CIFAR-10. The source and CSV files provide both endpoints, so this draft can distinguish them without new training runs.

**Table 11. Final-epoch versus checkpoint interpretation**

| Dataset | H | ReLU final % | Hybrid final % | Final Δ, pp | Checkpoint Δ, pp |
| --- | --- | --- | --- | --- | --- |
| MNIST | 32 | 96.477 | 96.490 | +0.014 | -0.186 |
| MNIST | 128 | 97.880 | 98.053 | +0.173 | +0.178 |
| CIFAR-10 | 64 | 50.627 | 51.393 | +0.766 | +0.363 |
| CIFAR-10 | 512 | 54.333 | 54.361 | +0.029 | -0.164 |

A validation-selected checkpoint can have lower test accuracy than the final epoch. Selection maximizes validation accuracy, not test accuracy; choosing whichever test endpoint looks better would introduce further selection bias.

### 3.3 Aligning learning-rate calibration with checkpoint selection

Original calibration maximizes mean final validation accuracy, even though deployment of the selected checkpoint uses the maximum validation accuracy along the trajectory. Recomputing LR selection using mean `best_val_acc` changes 17 of the 84 main selections. Existing evaluation sweeps cover four of these changed conditions; thirteen need additional evaluation runs before a fully recalibrated main grid can be reported. The MSE width-64 hybrid selection also changes and lacks matching evaluation runs.

**Table 12. MNIST H=128 calibration sensitivity**

| Comparison | ReLU LR | Hybrid LR | ReLU checkpoint % | Hybrid checkpoint % | Δ, pp | 95% CI | Relative error reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original tuning | 0.003 | 0.01 | 97.860 | 98.038 | +0.178 | [+0.143, +0.213] | 8.30% |
| Checkpoint-aligned tuning | 0.01 | 0.01 | 98.009 | 98.038 | +0.029 | [-0.017, +0.080] | 1.47% |

After this alignment, the observed difference shrinks by 83.5%. The remaining +0.029 pp corresponds to approximately 2.9 additional correct predictions per 10,000 images per seed on average. The paired p-value is 0.266, with 17/30 positive seed differences. This retrospective comparison does not establish a reliable accuracy advantage; it also does not prove equivalence. The new rate was chosen using calibration validation records, not evaluation test accuracy, but the comparison still reuses the existing evaluation seeds.

**Table 13. All changed main calibrations with existing evaluation coverage**

| Dataset | H | Activation | LR change | Old accuracy % | Aligned accuracy % | Δ, pp | 95% paired CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 128 | sigmoid + ReLU | 0.0003 → 0.001 | 51.902 | 51.985 | +0.083 | [-0.137, +0.296] |
| CIFAR-10 | 128 | tanh | 0.0003 → 0.001 | 51.930 | 51.788 | -0.142 | [-0.384, +0.103] |
| MNIST | 64 | sigmoid | 0.01 → 0.03 | 97.123 | 97.232 | +0.109 | [+0.054, +0.165] |
| MNIST | 128 | ReLU | 0.003 → 0.01 | 97.860 | 98.009 | +0.148 | [+0.108, +0.188] |

These four sensitivity checks are exploratory and are not a replacement for the missing thirteen conditions. Full changed selections are listed in the supplement.

## 4. Quantifying learning-rate robustness

### 4.1 Why a universal “threefold” statement is too broad

A robustness multiplier requires a defined criterion, comparator, grid, dataset, and width. As one transparent reconstruction of the suggested 3× claim, we can compare the highest tested LR whose mean final validation accuracy remains at least 95% on the original MNIST calibration grid. At widths 64 and 128, this is 0.01 for ReLU and 0.03 for tanh–ReLU. The ratio is exactly 3 on that grid. The 95% criterion is introduced here for descriptive interpretation and was not preregistered.

**Table 14. Upper tested LR retaining mean final validation accuracy ≥95%**

| H | Stage | Seeds per LR | ReLU upper LR | Tanh–ReLU upper LR | Upper-LR ratio |
| --- | --- | --- | --- | --- | --- |
| 64 | calib | 6 | 0.01 | 0.03 | 3.00× |
| 64 | lrsweep | 30 | 0.02 | 0.03 | 1.50× |
| 128 | calib | 6 | 0.01 | 0.03 | 3.00× |
| 128 | lrsweep | 30 | 0.02 | 0.03 | 1.50× |

The evaluation grid adds 0.005, 0.02, and 0.05. The observed ratio falls to 1.5× when the intermediate 0.02 point is available. This is a ratio of upper tested LRs, not a ratio of probabilities, accuracy, or interval widths.

**Table 15. Alternative robustness definitions on evaluation LR grids**

| Dataset | H | Criterion | ReLU interval (points) | Hybrid interval (points) | Upper-LR ratio | Log10 span: ReLU / hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 64 | Within 1 pp of own best | 0.0003–0.01 (5) | 0.0003–0.01 (5) | 1.00× | 1.523 / 1.523 |
| MNIST | 64 | No final validation <15% | 0.0001–0.02 (7) | 0.0001–0.05 (9) | 2.50× | 2.301 / 2.699 |
| MNIST | 128 | Within 1 pp of own best | 0.0003–0.01 (5) | 0.0003–0.02 (6) | 2.00× | 1.523 / 1.824 |
| MNIST | 128 | No final validation <15% | 0.0001–0.02 (7) | 0.0001–0.05 (9) | 2.50× | 2.301 / 2.699 |
| CIFAR-10 | 128 | Within 1 pp of own best | 0.0003–0.001 (2) | 0.0003–0.001 (2) | 1.00× | 0.523 / 0.523 |
| CIFAR-10 | 128 | No final validation <15% | 0.0001–0.01 (5) | 0.0001–0.003 (4) | 0.30× | 2.000 / 1.477 |

Intervals are the widest contiguous passing sequences on the tested grid, selected by log span. No behavior between sampled rates is assumed. Each configuration’s “own best” is its highest mean final validation score within that grid.

For MNIST H=128, the near-optimal upper LR increases from 0.01 to 0.02, a 2× upper-bound ratio. The full log-span ratio is only 1.8239/1.5229 ≈ 1.20. For the zero-observed-failure criterion, the upper LR increases from 0.02 to 0.05, or 2.5×, and the log-span ratio is about 1.17. At H=64, the near-optimal interval is unchanged. These measures support a conditional extension of stability, but cannot be compressed into one general threefold number. Survival at a large LR can coexist with an unacceptable loss of accuracy.

### 4.2 Dataset transfer and a non-monotonic failure pattern

CIFAR-10 does not reproduce a general high-performance stability advantage. At H=128 and LR=0.01, tanh–ReLU has lower mean final validation accuracy than ReLU. At LR=0.03, it improves on a largely failed ReLU model, but remains far below the accuracy achieved at small LRs. Leaky-ReLU is a particularly important control on this dataset.

**Table 16. CIFAR-10 H=128 LR sensitivity**

| LR | Activation | Final validation % | Peak validation % | Drop, pp | Final <15% |
| --- | --- | --- | --- | --- | --- |
| 0.0003 | ReLU | 51.887 | 52.753 | 0.867 | 0/30 |
| 0.0003 | tanh + ReLU | 52.006 | 52.744 | 0.738 | 0/30 |
| 0.0003 | Leaky-ReLU | 51.709 | 52.541 | 0.831 | 0/30 |
| 0.003 | ReLU | 48.351 | 49.786 | 1.435 | 0/30 |
| 0.003 | tanh + ReLU | 48.833 | 49.981 | 1.148 | 0/30 |
| 0.003 | Leaky-ReLU | 49.223 | 50.867 | 1.644 | 0/30 |
| 0.01 | ReLU | 37.516 | 40.855 | 3.339 | 0/30 |
| 0.01 | tanh + ReLU | 28.110 | 37.652 | 9.542 | 1/30 |
| 0.01 | Leaky-ReLU | 44.602 | 47.206 | 2.604 | 0/30 |
| 0.03 | ReLU | 10.903 | 15.503 | 4.600 | 26/30 |
| 0.03 | tanh + ReLU | 17.879 | 26.346 | 8.467 | 9/30 |
| 0.03 | Leaky-ReLU | 36.411 | 41.501 | 5.090 | 0/30 |

Source: [c10_lrsweep_128.csv](/Users/khaos/Desktop/article/version5/results_aws/c10_lrsweep_128.csv). All seven activations and eight LRs appear in the supplement.

The CIFAR-10 tanh–ReLU condition has one final near-chance failure at LR=0.01 and nine at LR=0.03. Thus the lower failure rate than ReLU at 0.03 does not imply a uniformly expanded failure-free interval: under the contiguous zero-failure definition, its upper bound is 0.003 versus 0.01 for ReLU. In addition, the old 50% cutoff labels 561 of 1,260 calibrated CIFAR-10 main runs as “collapsed,” although none ends below 15%. The draft consequently separates task accuracy, chance-level failure, and peak-to-final deterioration.

## 5. Comparing 25%, 50%, and 75% hybrid allocations

### 5.1 Definition and controlled scope

We define the activation allocation ratio ρ as the fraction of neurons assigned to the second activation function within each hidden layer. For tanh–ReLU, ρ is the ReLU share; for tanh–leaky-ReLU, it is the leaky-ReLU share. Each neuron uses a single assigned function. The model does not compute a weighted sum of both functions at every neuron. The same allocation is applied to both hidden layers.

**Table 17. Neuron counts for each tested ratio at H=128**

| Second-activation share ρ | tanh share | tanh neurons per layer | Rectifier neurons per layer | Layers |
| --- | --- | --- | --- | --- |
| 25% | 75% | 96 | 32 | Both hidden layers |
| 50% | 50% | 64 | 64 | Both hidden layers |
| 75% | 25% | 32 | 96 | Both hidden layers |

All ratio comparisons use MNIST, H=128, interleave masks, cross-entropy, LR=0.01, and 30 paired evaluation seeds. Parameter count is 118,282 for every condition. The ratios were not independently tuned.

**Table 18. Allocation ratio performance**

| Hybrid | ρ | Checkpoint accuracy % ± SD | Checkpoint macro-F1 % ± SD | Peak validation % | Final test % | Selected epoch mean |
| --- | --- | --- | --- | --- | --- | --- |
| tanh + ReLU | 25% | 97.957 ± 0.099 | 97.937 ± 0.101 | 98.282 | 97.976 | 19.8 |
| tanh + ReLU | 50% | 98.038 ± 0.073 | 98.020 ± 0.074 | 98.328 | 98.053 | 24.0 |
| tanh + ReLU | 75% | 98.052 ± 0.079 | 98.036 ± 0.080 | 98.400 | 98.067 | 23.6 |
| tanh + leaky-ReLU | 25% | 97.965 ± 0.096 | 97.944 ± 0.098 | 98.341 | 97.968 | 21.6 |
| tanh + leaky-ReLU | 50% | 97.986 ± 0.081 | 97.967 ± 0.080 | 98.333 | 98.001 | 21.2 |
| tanh + leaky-ReLU | 75% | 98.052 ± 0.084 | 98.035 ± 0.085 | 98.405 | 98.067 | 25.0 |

Source: [layout_128.csv](/Users/khaos/Desktop/article/version5/results_aws/layout_128.csv). The 50% interleave conditions repeat the main-grid jobs and are not independent additional evidence.

**Table 19. Paired allocation contrasts**

| Hybrid | Contrast | Δ accuracy, pp | 95% CI | Wins/ties/losses | Raw p | Holm p (6) |
| --- | --- | --- | --- | --- | --- | --- |
| tanh + ReLU | 50% − 25% | +0.081 | [+0.036, +0.126] | 19/2/9 | 0.002 | 0.007 |
| tanh + ReLU | 75% − 25% | +0.095 | [+0.050, +0.138] | 23/1/6 | 3.64e-04 | 0.002 |
| tanh + ReLU | 75% − 50% | +0.014 | [-0.016, +0.047] | 10/4/16 | 0.406 | 0.811 |
| tanh + leaky-ReLU | 50% − 25% | +0.021 | [-0.029, +0.068] | 17/0/13 | 0.411 | 0.811 |
| tanh + leaky-ReLU | 75% − 25% | +0.087 | [+0.051, +0.123] | 23/2/5 | 7.96e-05 | 4.78e-04 |
| tanh + leaky-ReLU | 75% − 50% | +0.066 | [+0.025, +0.108] | 20/0/10 | 0.005 | 0.015 |

The separate Holm family includes these six ratio accuracy contrasts only. It does not adjust CIs or cover layout, LR, loss, or prior adaptive exploration.

For tanh–ReLU, increasing the ReLU share from 25% to 50% raises mean checkpoint accuracy by 0.081 pp. Increasing it further to 75% adds only 0.014 pp, with an interval crossing zero. The latter comparison has 10 wins, 4 ties, and 16 losses across the 30 seeds, with a median difference of −0.010 pp despite its positive mean. This indicates that the small average advantage is not a consistent seed-by-seed improvement.

For tanh–leaky-ReLU, the 25% to 50% change is small and unresolved, while the 50% to 75% increase is +0.066 pp. That contrast remains below p=0.05 after the local six-comparison Holm correction, but remains an exploratory result at one fixed LR. Both 75% hybrids have the same mean checkpoint accuracy, 98.052%, to the available precision. The results suggest a plateau between 50% and 75% for tanh–ReLU and a larger benefit of the 75% allocation for tanh–leaky-ReLU within this narrow design. They do not establish a universal optimum.

### 5.2 Comparing ratio variants with homogeneous endpoints

Pure tanh is available at LR=0.01, and the MNIST H=128 LR sweep supplies a pure ReLU control at that same rate. This gives a useful common-LR context for the tanh–ReLU allocation curve. Pure ReLU’s original main-grid LR=0.003 would inflate the apparent benefits of all ratio variants, so the two baselines are distinguished.

**Table 20. Tanh–ReLU allocations versus pure ReLU at the common LR=0.01**

| ReLU share | Hybrid accuracy % | Δ versus ReLU, pp | 95% paired CI |
| --- | --- | --- | --- |
| 25% | 97.957 | -0.051 | [-0.099, -0.004] |
| 50% | 98.038 | +0.029 | [-0.017, +0.080] |
| 75% | 98.052 | +0.043 | [-0.005, +0.093] |

Pure ReLU accuracy is 98.009%. Pure tanh at LR=0.01 reaches 97.827%. These additional endpoint comparisons are exploratory and are not included in the six-ratio correction.

The ratio experiment measures accuracy at LR=0.01 only. It contains no high-LR sweep for 25% or 75%, so it cannot determine whether a 75% rectifier model retains the 50% model’s stability at LR=0.03. A promising next hypothesis is that a minority of tanh neurons could preserve stability while a larger rectifier allocation preserves accuracy. That hypothesis requires new measurements.

### 5.3 Layout sensitivity at a fixed 50% allocation

**Table 21. Mask layout results at ρ=50%, LR=0.01**

| Hybrid | Layout | Checkpoint accuracy % ± SD | Δ versus interleave, pp | 95% CI |
| --- | --- | --- | --- | --- |
| tanh + ReLU | interleave | 98.038 ± 0.073 | +0.000 | [+0.000, +0.000] |
| tanh + ReLU | block | 98.020 ± 0.088 | -0.018 | [-0.056, +0.021] |
| tanh + ReLU | random | 98.034 ± 0.108 | -0.004 | [-0.055, +0.046] |
| tanh + leaky-ReLU | interleave | 97.986 ± 0.081 | +0.000 | [+0.000, +0.000] |
| tanh + leaky-ReLU | block | 98.022 ± 0.086 | +0.036 | [-0.013, +0.085] |
| tanh + leaky-ReLU | random | 98.017 ± 0.071 | +0.031 | [-0.000, +0.066] |

Same seed identifiers are paired. Random mask generation consumes the RNG stream later used for initialization and shuffling, so nuisance randomness is not matched for those comparisons.

No intrinsic advantage of neuron position is established. In a fully connected layer, permuting hidden neurons together with the corresponding incoming and outgoing weights preserves the represented function. Different fixed masks can still interact with a particular initialization. The present random-layout runs also change the subsequent RNG sequence, which prevents isolating mask placement alone.

## 6. Further patterns: loss, classes, capacity, and endpoint behavior

### 6.1 Loss/output transfer

**Table 22. MNIST MSE/sigmoid-output checkpoint results**

| H | Activation | LR | Accuracy % ± SD | Macro-F1 % ± SD | Δ versus ReLU, pp | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | ReLU | 0.01 | 97.637 ± 0.133 | 97.612 ± 0.134 | +0.000 | [+0.000, +0.000] |
| 64 | tanh | 0.01 | 97.266 ± 0.132 | 97.232 ± 0.133 | -0.371 | [-0.436, -0.311] |
| 64 | tanh + ReLU | 0.01 | 97.435 ± 0.119 | 97.407 ± 0.120 | -0.202 | [-0.265, -0.138] |
| 128 | ReLU | 0.01 | 97.998 ± 0.080 | 97.981 ± 0.081 | +0.000 | [+0.000, +0.000] |
| 128 | tanh | 0.01 | 97.777 ± 0.070 | 97.752 ± 0.072 | -0.221 | [-0.263, -0.177] |
| 128 | tanh + ReLU | 0.03 | 97.964 ± 0.088 | 97.943 ± 0.089 | -0.034 | [-0.075, +0.008] |

These are the implemented MSE experiments, not a claim of a fully matched normalized-MSE benchmark.

The hybrid does not outperform ReLU in the MSE results: its difference is −0.202 pp at H=64 and −0.034 pp at H=128. The H=64 hybrid would receive a different LR under checkpoint-aligned calibration, but that evaluation is unavailable. The implementation records the mean squared error over ten outputs while using a gradient corresponding to their sum, a tenfold scale difference. In addition, the loss ablation changes the output activation. Consequently, differences between CE and MSE cannot be attributed to the scalar loss function alone, and identical numerical learning rates are not equivalent update scales.

### 6.2 Per-class patterns

Class metrics were recomputed from each run’s checkpoint confusion matrix before taking seed means. This preserves per-run macro-F1 semantics. The following contrasts show where the tanh–ReLU accuracy changes occur. MNIST uses the checkpoint-aligned ReLU LR=0.01 control to avoid repeating the known calibration artifact; CIFAR-10 uses the original H=128 main LRs. These class-wise comparisons are descriptive and not multiplicity-adjusted.

**Table 23. Class recall at H=128: tanh–ReLU versus ReLU**

| Dataset | Class | Test support | ReLU recall % | Hybrid recall % | Δ recall, pp |
| --- | --- | --- | --- | --- | --- |
| MNIST | 0 | 985 | 99.465 | 99.442 | -0.024 |
| MNIST | 1 | 1149 | 98.938 | 98.918 | -0.020 |
| MNIST | 2 | 1007 | 97.418 | 97.153 | -0.265 |
| MNIST | 3 | 997 | 96.830 | 96.693 | -0.137 |
| MNIST | 4 | 985 | 97.787 | 97.990 | +0.203 |
| MNIST | 5 | 866 | 97.710 | 97.791 | +0.081 |
| MNIST | 6 | 977 | 98.656 | 98.826 | +0.171 |
| MNIST | 7 | 1082 | 98.549 | 98.561 | +0.012 |
| MNIST | 8 | 996 | 97.701 | 97.905 | +0.204 |
| MNIST | 9 | 956 | 96.789 | 96.883 | +0.094 |
| CIFAR-10 | airplane | 1000 | 60.440 | 59.713 | -0.727 |
| CIFAR-10 | automobile | 1000 | 63.380 | 64.207 | +0.827 |
| CIFAR-10 | bird | 1000 | 39.230 | 37.413 | -1.817 |
| CIFAR-10 | cat | 1000 | 33.210 | 32.600 | -0.610 |
| CIFAR-10 | deer | 1000 | 44.553 | 45.233 | +0.680 |
| CIFAR-10 | dog | 1000 | 42.273 | 44.080 | +1.807 |
| CIFAR-10 | frog | 1000 | 60.750 | 62.527 | +1.777 |
| CIFAR-10 | horse | 1000 | 59.963 | 61.273 | +1.310 |
| CIFAR-10 | ship | 1000 | 68.120 | 67.760 | -0.360 |
| CIFAR-10 | truck | 1000 | 58.513 | 57.403 | -1.110 |

Per-class recall concerns class-conditional accuracy; macro-F1 additionally depends on predicted-class composition. Confusion matrices cannot recover whether different models err on the same individual images.

The small overall MNIST difference conceals opposing class effects: the aligned hybrid improves recall for digits 4 and 8 by about 0.20 pp each, while losing 0.265 pp on digit 2 and 0.137 pp on digit 3. On CIFAR-10, the same-width hybrid gains approximately 1.81 pp for dogs and 1.78 pp for frogs, while losing 1.82 pp for birds and 1.11 pp for trucks. Its small aggregate gain therefore reflects offsetting class changes, not a uniform improvement. Because these classes were examined after the overall results, these observations are hypotheses for confirmation rather than class-specific superiority claims.

### 6.3 Capacity, generalization gap, and logged cost

**Table 24. ReLU width scaling and matched-checkpoint training diagnostics**

| Dataset | H | Parameters | Test % | Clean train % | Train–test gap, pp | Logged seconds | Selected epoch | Epoch 40 selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | 12 | 9,706 | 94.297 | 96.247 | 1.950 | 18.7 | 29.1 | 1/30 |
| MNIST | 32 | 26,506 | 96.707 | 99.385 | 2.679 | 49.9 | 26.9 | 1/30 |
| MNIST | 64 | 55,050 | 97.579 | 99.998 | 2.419 | 103.4 | 26.1 | 0/30 |
| MNIST | 128 | 118,282 | 97.860 | 99.993 | 2.132 | 213.7 | 25.1 | 1/30 |
| MNIST | 256 | 269,322 | 98.214 | 100.000 | 1.786 | 482.5 | 26.9 | 1/30 |
| MNIST | 512 | 669,706 | 98.285 | 99.999 | 1.714 | 1456.7 | 23.8 | 1/30 |
| CIFAR-10 | 12 | 37,162 | 43.461 | 45.070 | 1.609 | 74.8 | 36.3 | 8/30 |
| CIFAR-10 | 32 | 99,722 | 49.186 | 53.678 | 4.492 | 191.1 | 36.5 | 8/30 |
| CIFAR-10 | 64 | 201,482 | 51.239 | 59.972 | 8.733 | 371.6 | 32.9 | 2/30 |
| CIFAR-10 | 128 | 411,146 | 53.043 | 62.778 | 9.735 | 751.2 | 36.7 | 4/30 |
| CIFAR-10 | 256 | 855,050 | 54.074 | 65.991 | 11.918 | 2240.3 | 36.6 | 5/30 |
| CIFAR-10 | 512 | 1,841,162 | 54.875 | 69.194 | 14.318 | 8400.7 | 36.9 | 7/30 |

Parameter count is P(H,D)=H²+(D+12)H+10. Training accuracy comes from the fixed training subset at the selected checkpoint. Recorded seconds include loading, training, and evaluation under concurrent AWS workload; they are not isolated throughput or inference latency.

Increasing width strongly improves absolute accuracy, but its marginal return decreases. The CIFAR-10 ReLU training–test gap grows from 1.609 pp at H=12 to 14.318 pp at H=512; extra capacity is fitting the training subset faster than it improves test performance. MNIST approaches a much smaller residual test error, so additional width provides relatively small absolute gains. These patterns concern the present training budget and do not identify an optimal deployment size.

Across the CIFAR-10 main grid, 239 of 1,260 runs select epoch 40. This places a nontrivial fraction of maxima at the observation boundary. It supports extending the training horizon in a follow-up, while not guaranteeing that longer training improves generalization. Conversely, an earlier selected epoch does not itself demonstrate faster convergence or saved compute because all runs execute 40 epochs.

**Table 25. Selected accuracy–resource comparisons**

| Dataset | Configuration | Accuracy % | Parameters | Logged seconds |
| --- | --- | --- | --- | --- |
| MNIST | ReLU, H=256 | 98.214 | 269,322 | 482.5 |
| MNIST | tanh + ReLU, H=512 | 98.289 | 669,706 | 1474.4 |
| CIFAR-10 | ReLU, H=256 | 54.074 | 855,050 | 2240.3 |
| CIFAR-10 | ReLU, H=512 | 54.875 | 1,841,162 | 8400.7 |

On MNIST, width-256 ReLU is only 0.074 pp below the nominal width-512 hybrid mean while using 59.8% fewer parameters. On CIFAR-10, width-256 ReLU is 0.802 pp below width-512 ReLU with 53.6% fewer parameters. These are useful capacity trade-offs. Logged elapsed-time reductions of approximately 67.3% and 73.3%, respectively, are workload-specific and should not be advertised as controlled speedups.

## 7. Discussion

The data favor a conditional interpretation of hybrid activations. Tanh–ReLU can be markedly more stable at learning rates that induce late failure in ReLU, while offering similar tuned accuracy. The two properties need not move together. Sigmoid and leaky-ReLU can also resist near-chance failure at high LRs, yet this may be accompanied by lower peak or retained performance. The meaningful comparison therefore combines failure frequency with absolute accuracy, peak-to-final decline, and calibration sensitivity.

The apparent width-128 MNIST accuracy advantage illustrates why tuning and reporting must target the same endpoint. A pipeline optimized for final validation accuracy does not necessarily provide the strongest baseline for a deployed checkpoint selected earlier in training. Correcting this mismatch removes most of the focal advantage without removing the high-LR stability observation.

The allocation experiment adds a separate contribution. Increasing the rectifier share from 25% to 75% improves mean accuracy in both tanh-based hybrids at the tested LR. The marginal effect is not the same for the two rectifiers, and a larger positive mean does not guarantee more seed-wise wins. Ratio selection and stability must consequently be studied jointly before claiming that a particular mixture is best.

Cross-dataset and loss results define the boundary of the claim. The CIFAR-10 sweep does not show uniformly better stability for tanh–ReLU, and the implemented MSE experiments do not replicate a hybrid advantage over ReLU. Together with the uniformly weaker sigmoid–ReLU main results, these observations argue against attributing gains simply to mixing activation functions.

**Table 26. Evidence-supported patterns and appropriate claims**

| Pattern | Evidence | Suitable manuscript interpretation |
| --- | --- | --- |
| Late-training stability | MNIST H=128, LR=.03: 30/30 ReLU versus 0/30 hybrid failures | Tanh–ReLU prevents observed late failure in this condition |
| Calibration dependence | Focal accuracy Δ shrinks from +.178 to +.029 pp | Tuned accuracy advantage is not established after alignment |
| Width dependence | Small-width deficits, intermediate gains, wide-model near ties or reversals | Hybrid benefit is not monotonic in width |
| Activation dependence | Sigmoid–ReLU trails ReLU at all 12 main dataset/width cells | Mixing alone is insufficient |
| Allocation response | 75% exceeds 25% for both tanh-based pairs at one LR | Rectifier allocation changes accuracy within the tested design |
| Mean versus seed consistency | Tanh–ReLU 75% vs 50%: +.014 pp, 10 wins and 16 losses | A small mean gain need not be a typical-seed gain |
| Stability versus retention | Leaky variants can stay above chance while losing >10 pp | Report both collapse and retained performance |
| Dataset/loss limits | CIFAR-10 reversals and MSE deficits | Do not generalize the MNIST stability result to all settings |
| Capacity trade-off | Substantial parameter increases for modest high-width gains | Report accuracy with resource measures, without a universal optimum |

## 8. Limitations and requirements for confirmation

**Table 27. Limitations and focused confirmation experiments**

| Limitation | Effect on interpretation | Required confirmation |
| --- | --- | --- |
| One fixed split | Seed CIs omit dataset-sampling and split uncertainty | Replicate on independent splits and official MNIST test data |
| Endpoint-misaligned calibration | 17 main LR choices change; 13 lack evaluation coverage | Evaluate missing choices with disjoint confirmation seeds |
| Adaptive searches and multiple outcomes | Nominal effects may reflect selection | Predeclare primary contrasts and practical effect margins |
| Only H=128/MNIST for ratio study | No evidence of an optimal ratio elsewhere | Factorial ratio × LR × width × dataset design |
| Missing activation/gradient diagnostics | Mechanism remains hypothetical | Log norms, inactive neurons, clipping, weight scales, and class outputs |
| Shared RNG stream for random masks | Layout cannot be isolated from subsequent randomness | Independent RNG streams and permutation-matched controls |
| Implemented small initialization scale | Results may depend on initialization choice | Compare matched and activation-appropriate initialization |
| MSE scale and output change | Loss effects are confounded | Verify gradients and tune equivalent update budgets |
| 40-epoch ceiling | Some best epochs remain right-censored | Extend learning curves with independently tuned schedules |
| Concurrent elapsed timings | Cannot infer hardware-independent speed or energy savings | Measure isolated training, latency, memory, and energy |
| No individual predictions/probabilities | No paired disagreement test or probability calibration analysis | Store labels, logits/probabilities, and per-example predictions |

## 9. Conclusion

Hybrid activation assignment changes optimization behavior in a way that can be useful, particularly for retaining MNIST performance at learning rates that cause ReLU to fail late in training. The existing records support that conditional stability claim much more strongly than a general accuracy-superiority claim. A threefold figure is defensible only when explicitly attached to the coarse-grid upper-LR criterion; denser sweeps and alternative definitions produce different ratios. The 25%, 50%, and 75% allocation study shows measurable ratio dependence, with diminishing returns for tanh–ReLU between 50% and 75% and a larger 75% benefit for tanh–leaky-ReLU. Joint ratio and LR tuning, aligned checkpoint calibration, and direct internal measurements are the next steps needed to turn these observations into a broader explanatory result.

## Data and analysis availability

The analysis leaves the original experiment files unchanged. Companion outputs contain the complete grouped metric summaries, paired main contrasts, calibration changes, LR stability summaries, ratio comparisons, and per-class summaries. Raw-file SHA-256 hashes and reconciliation results are recorded in `verification.json`. The source scripts permit regeneration of all tables. The independent recomputation reproduces all 240 accuracy and macro-F1 mean differences from the prior main-grid audit. No new training runs were performed.

Primary local documentation: [summary_extended.md](/Users/khaos/Desktop/article/version5/results_aws/summary_extended.md), [rapor_ve_makale_gorusme_notlari.md](/Users/khaos/Desktop/article/version5/results_aws/rapor_ve_makale_gorusme_notlari.md), and [research_audit/evaluation.md](/Users/khaos/Desktop/article/version5/results_aws/research_audit/evaluation.md). The notes copied under `research_audit` are byte-identical to the root notes. The previous summary is treated as context; original CSV fields and source semantics govern the current numerical presentation.

Companion: [Supplementary Tables](/Users/khaos/Desktop/article/version5/results_aws/manuscript_draft/supplementary_tables.md).
