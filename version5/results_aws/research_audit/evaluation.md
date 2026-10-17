Experimental evaluation of hybrid-activation MLPs

The supplied prompt contains placeholders. This evaluation therefore uses the 32 result CSVs in the shared results_aws workspace and the adjacent experimental source. The inferred objective is to assess whether heterogeneous hidden activations improve accuracy, macro-F1, optimization stability, or computational trade-offs relative to homogeneous parents. ReLU is the primary baseline; each hybrid is also compared with both parents. No raw result or existing report was modified.

The strongest defensible finding is a learning-rate-dependent stability benefit on MNIST, with limited evidence of a generally superior hybrid classifier. The nominal highest main-grid checkpoint accuracy is tanh+ReLU at width 512 on MNIST (98.289% ± 0.087% seed SD), effectively tied with ReLU (98.285% ± 0.064%). CIFAR-10 instead favors ReLU at width 512 descriptively (54.875% ± 0.469%; macro-F1 54.577%). Its difference from tanh+ReLU is not resolved by the paired interval. Rankings are descriptive and conditional on the original calibration rule, one data split, and a 40-epoch budget.

A critical endpoint mismatch changes the central MNIST width-128 result. The existing summary calls its scores early-stopping results but reads final-epoch test_acc/test_f1. The runner also stores test_acc_best/test_f1_best from the maximum-validation-accuracy checkpoint. Moreover, calibration selects learning rates by final val_acc. For ReLU width 128, calibrating on best_val_acc selects 0.01 rather than 0.003. Comparing checkpoint scores at those aligned rates reduces the tanh+ReLU advantage from +0.178 percentage points (95% paired bootstrap CI +0.141 to +0.214) to +0.029 (−0.018 to +0.082; paired t p=0.266; 17/30 wins). Roughly 83.5% of the apparent difference disappears. The remaining difference is about 2.9 additional correct classifications per 10,000 images per seed, on average. Failure to reject zero does not establish equivalence.

For resource-constrained selection, width-256 ReLU is a reasonable provisional compromise on both datasets. On MNIST it reaches 98.214%, only 0.074 percentage points below the nominal width-512 hybrid winner, using 269,322 versus 669,706 parameters (59.8% fewer) and 482.5 versus 1,474.4 logged seconds (67.3% less). On CIFAR-10 width-256 ReLU reaches 54.074%, 0.802 points below width-512 ReLU, with 855,050 versus 1,841,162 parameters (53.6% fewer) and 2,240.3 versus 8,400.7 seconds (73.3% less). A unique efficiency optimum requires a deployment objective and direct latency or energy measurements.

Protocol and integrity

The network is D→H→H→10, D=784 or 3072, with H∈{12,32,64,128,256,512}. Training uses single-example SGD, 40 completed epochs, zero initial biases, a shared fixed split, shuffled epochs, and elementwise clipping of preactivation derivatives at ±10. Leaky-ReLU slope is 0.1. CE uses linear logits and softmax; the MSE runs use sigmoid outputs. Main hybrids assign 50% of each hidden layer to each activation. The implementation selects a checkpoint after training all epochs; it does not stop computation early.

The audit found 13,620 data rows, all status=ok, after removing 31 repeated CSV header lines. There were no duplicated job keys within individual files and no missing checkpoint scores. The 2,520 main runs comprise 2 datasets × 6 widths × 7 configurations × 30 seeds. Calibration uses seeds 101–106; evaluation uses 1–30. There are 690 repeated job identities across stages, so pooling stages as independent replication would inflate n. Confusion matrices for every row contain 10,000 predictions and reproduce checkpoint accuracy and macro-F1 within CSV rounding. All validation histories have 40 entries and reproduce the stored maximum and selected epoch. Class-support vectors are consistent within each dataset; this verifies composition, not image identity or absence of duplicate images.

MNIST is split 45,000/5,000/10,000 from the 60,000-image training file; its reported test accuracy must not be described as accuracy on the official MNIST test set. CIFAR-10 uses 45,000 training and 5,000 validation examples from its training set and the official 10,000-image test batch. Source-level observations below refer to the adjacent source; the AWS executable was not rebuilt and matched to that source during this audit.

Quantitative interpretation

Unless explicitly marked final, accuracy and F1 below use the validation-selected checkpoint. SD describes variation across seeds, not uncertainty across new dataset samples. For paired differences d_s=A_variant,s−A_baseline,s, the estimated effect is mean(d_s); confidence limits are percentiles of 10,000 resamples of complete seed pairs. Relative accuracy change is 100×(A_variant−A_baseline)/A_baseline; relative error reduction is 100×(A_variant−A_baseline)/(100−A_baseline), with accuracies expressed in percent. These are different quantities. The original width-128 MNIST gain represents 0.182% relative accuracy improvement or 8.30% error reduction; after aligned calibration those become approximately 0.030% and 1.47%.

At widths 12 and 32 on MNIST, tanh+ReLU underperforms ReLU by 0.307 and 0.186 points at selected checkpoints. At 64, 256, and 512 the differences are +0.004, −0.001, and +0.003 points. Thus the results do not support a monotonic width-dependent emergence of hybrid superiority. On CIFAR-10 its differences versus ReLU are −0.550, −0.385, +0.363, +0.178, +0.153, and −0.164 points in ascending width order. The width-64 effect is nominally positive (95% CI +0.155 to +0.577), but its Holm-adjusted p is 0.095 across the 120 audited main-grid checkpoint-accuracy contrasts. At widths 128 and 256 its pointwise intervals include zero. Beating tanh alone is insufficient evidence that mixing is better than the strongest homogeneous parent.

Sigmoid+ReLU underperforms ReLU at every main-grid width on both datasets. Its MNIST deficit reaches 0.870 points at width 64 (approximately 35.9% relative error increase). Nevertheless, it can improve on homogeneous sigmoid, especially on CIFAR-10. Macro-F1 broadly follows accuracy, but the two criteria are not interchangeable: CIFAR-10 width-64 tanh+leaky-ReLU has slightly higher accuracy than leaky-ReLU (51.288 vs 51.270) and lower macro-F1 (50.847 vs 50.904). The companion class table provides seed means and SDs for precision, recall, and F1 for every main, MSE, and layout condition. A confusion matrix does not provide predictive-probability calibration, AUROC, or example-level cross-model error agreement.

Anomalies and mechanisms

At MNIST width 128 and LR=0.03, all 30 ReLU runs finish below 15% validation accuracy, with mean 10.611%, despite mean best validation accuracy of 96.371%. Tanh+ReLU at the same width and LR finishes at 96.761%, with no runs below 15%. Width-64 ReLU has 12/30 such failures at LR=0.03, versus 0/30 for tanh+ReLU. This is a reproducible late-training instability and a strong conditional robustness result. It does not establish superior optimal accuracy, and an all-zero observed failure count still permits an approximately 9.5% one-sided 95% binomial upper bound under independent-seed sampling.

ReLU’s zero negative-side derivative can prevent recovery of inactive units, whereas leaky-ReLU retains a negative-side gradient and tanh bounds forward activations. These mechanisms plausibly contribute to stability differences. The logs contain no activation histograms, gradient norms, clipping frequencies, or dead-unit counts, so neither dead ReLUs nor gradient explosion is established as the cause. Clipping dL/dz does not bound weight updates when the preceding activation is large: ΔW=−η·clip(dL/dz)·h. The source’s non-finite check tests only the first output for NaN; status=ok is not a comprehensive numerical-health certificate.

The report’s final-validation threshold of 50% labels 561/1,260 calibrated CIFAR-10 main runs as collapsed, although none finishes below 15%. This confuses limited accuracy with optimization failure. The 15% level used here is only a transparent near-chance diagnostic, not a universally validated threshold. Report chance-level failure, substantial deterioration from the run’s peak, and task-relative performance degradation separately. High-LR survival can also be misleading: MNIST width-128 leaky-ReLU at LR=0.03 finishes around 85.228% validation accuracy, while its best validation mean is 96.365%. Remaining above 50% does not mean retaining useful performance.

The implemented initializer samples U(−1/√fan_in,+1/√fan_in), giving Var(W)=1/(3 fan_in). Standard He variance for ReLU is 2/fan_in, six times larger. This is the Linear-default-style scale, not ReLU-gain He initialization. Identical weight distributions across activations control one implementation variable but do not establish equally well-tuned baselines. Sigmoid derivatives are at most 1/4, tanh derivatives approach zero in saturation, and input scaling to [0,1] is not centering. Their interaction with the small initialization scale and finite budget is a plausible explanation for optimization differences. These hypotheses are consistent with the activation and initialization analyses of [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) and [He et al. (2015)](https://arxiv.org/abs/1502.01852); those papers do not validate the present hybrid-specific causal claims or constitute matched performance benchmarks.

Layout and loss ablations

At width 128, increasing the rectifier fraction from 0.5 to 0.75 raises tanh+ReLU checkpoint accuracy by 0.014 points (95% CI −0.017 to +0.047), and tanh+leaky-ReLU by 0.066 points (+0.025 to +0.109). Both 0.75 variants average 98.052%; this is a descriptive tie. Ratios were not independently calibrated, and the latter interval is unadjusted for layout searches. Interleave, block, and random arrangements do not establish an intrinsic positional advantage. In a dense MLP, neuron permutations with corresponding weight permutations preserve the represented function. Random masks additionally consume the same RNG stream used later for weights and shuffles, so same seed does not match these nuisance variables across random-layout comparisons.

MSE does not replicate a hybrid advantage over ReLU. At width 64, tanh+ReLU scores 97.435% versus ReLU 97.637% (difference −0.202 points; 95% CI −0.267 to −0.138). At width 128 it scores 97.964% versus 97.998% (−0.034; −0.077 to +0.009). The width-64 checkpoint-based calibration would select a different hybrid LR, so that deficit also requires endpoint-aligned confirmation. The existing extended summary omits the available width-64 MSE results. MSE comparisons change both objective and output activation. Furthermore, the source logs (1/10)Σ_j error_j² but backpropagates 2·error_j rather than (2/10)·error_j. This is a tenfold gradient-scale mismatch to the logged objective, which must be resolved before interpreting equal numerical learning rates across losses.

Efficiency, convergence, and validity limits

The parameter count including biases is P(H,D)=H²+(D+12)H+10. At fixed width all activation variants have equal trainable parameter counts, but nonlinear-function cost can differ. Logged seconds include data loading, training, per-epoch validation and clean-training evaluation, and final evaluations. Logs show up to 190 workers on 192 cores. These are run elapsed times under that workload, not inference latency or isolated training throughput. CIFAR-10 width-512 sigmoid+ReLU appears 38.8% faster than ReLU, but this cannot be attributed to activation mechanics without controlled timing. Repeated width-128 jobs across main and LR stages reproduce accuracy exactly and timing within a few percent; that supports local repeatability, not cross-width hardware comparability.

Checkpoint training accuracy is reconstructed from the clean-training history at the selected epoch, using a fixed 10,000-image training subset. It must not be combined with final-epoch training accuracy or online training accuracy to infer overfitting. ReLU’s CIFAR-10 checkpoint train–test gap grows from 1.609 points at width 12 to 14.318 at width 512; MNIST width-512 ReLU’s gap is 1.714 points. This supports investigating generalization limits on CIFAR-10, although the training subset is not an independent validation sample. Best epochs average roughly 32–38 on CIFAR-10, and 239/1,260 main runs select epoch 40. The right-censored budget warrants longer learning curves. Selecting an earlier maximum alone does not establish faster convergence or saved computation.

Thirty training seeds quantify stochastic training variation conditional on a single split. They are not 30 independent test sets or 300,000 independent test examples. The bootstrap intervals do not include split uncertainty, calibration-seed uncertainty, or adaptive configuration selection. With 10,000 test images, an individual-model binomial SE is roughly 0.13 percentage points near 98.3% accuracy and 0.50 points near 54.9%; paired model comparisons require per-example joint predictions to exploit shared errors. Confusion matrices cannot recover those disagreements. Multiple widths, activations, LRs, layouts, endpoints, and losses create substantial multiplicity; the Holm correction supplied covers 120 main checkpoint-accuracy comparisons only. F1, layout, MSE, and LR analyses remain exploratory. Disjoint calibration seeds are a strength, but six calibration seeds can yield unstable LR choices, and repeated use of the same validation examples does not remove selection optimism. Seventeen of 84 main calibration choices change when switching from final to maximum validation accuracy (18 of 90 including MSE). Unavailable evaluation runs at newly selected rates require further experiments.

Prioritized next experiments

1. Align calibration and reporting with the deployed endpoint. Select LR by mean best_val_acc if the best checkpoint is deployed; report test_acc_best/test_f1_best, selected epoch, and checkpoint-matched training metrics. Keep final-epoch scores as a separately named endpoint. Re-evaluate changed selections on untouched seeds; available width-64/128 sweeps permit retrospective checks but not independent confirmation.
2. Validate implementation semantics. Add finite-difference checks of CE and normalized-MSE gradients, scan every activation/weight/gradient for non-finite values, document derivative clipping, and compare predictions and updates against a reference implementation. Record executable, compiler, source, dataset, split, and initial-weight hashes. Decouple RNG streams for initialization, masks, and example order.
3. Test the strongest robustness claim directly. At widths 64 and 128, densely sample LRs around 0.01–0.03 using independent seeds and identical initial weights/order. Measure dead-unit fraction, activation distributions, gradient norms, clipping frequency, prediction-class entropy, peak-to-final deterioration, and time of failure. Include ReLU, leaky-ReLU, and tanh+ReLU.
4. Run a factorial initialization × activation × LR study. Include the present scale, activation-appropriate He/Glorot initialization, and centered inputs. Give all variants equal tuning budgets. Separately vary clipping and output/loss formulation. Only then attribute improvements to activation heterogeneity.
5. Replicate across data splits and the official MNIST test set. Predeclare the main paired comparison, multiplicity family, and a practical margin such as 0.1 percentage points on MNIST if relevant to the application. Store per-example labels, predictions, and probabilities for paired disagreement tests, hierarchical uncertainty, class-level diagnostics, NLL, Brier score, and calibration analysis.
6. Resolve capacity and cost trade-offs. Extend CIFAR-10 training beyond 40 epochs with independently tuned schedules and weight decay; test a CNN reference if the goal is image-classification performance. Measure isolated training time, time to a fixed validation target, inference latency distributions, memory, and energy at widths 128/256/512. Retune ratios 0.25/0.5/0.75 and use permutation-matched controls for layout.

Complete main-grid comparison

Each table row below is one configuration at its original calibrated LR and n=30. Acc/F1 are checkpoint percentages with sample SD. Δ is checkpoint accuracy minus ReLU at the same dataset and width; rel. error is percent error reduction versus that baseline. Time is the logged total run mean. The companion baseline_metric_deltas.csv compares all 15 summarized metric means against ReLU for the main and MSE stages, including absolute and relative changes (relative changes of gaps or near-zero quantities require caution). Full final/selected validation, train, test, epoch, timing, and loss summaries for all stages are in all_metrics.csv, including means, SDs, medians, minima and maxima. Paired intervals, wins, raw p-values and the primary Holm corrections are in paired_comparisons.csv. LR cell counts and final/best metrics are in lr_diagnostics.csv; derived per-class precision/recall/F1 are in class_metrics.csv. These companion tables are analytical outputs, not additional independent observations.


MNIST, width 12; 9,706 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.001 | 94.297 ± 0.303 | 94.201 ± 0.312 | +0.000 | +0.00 | 94.106 | 29.1 | 18.7 |
| leaky_relu | 0.001 | 94.274 ± 0.226 | 94.176 ± 0.231 | -0.023 | -0.40 | 94.151 | 30.4 | 19.4 |
| tanh | 0.001 | 93.952 ± 0.260 | 93.846 ± 0.269 | -0.345 | -6.05 | 93.819 | 32.1 | 20.0 |
| sigmoid | 0.003 | 93.770 ± 0.331 | 93.658 ± 0.336 | -0.527 | -9.23 | 93.737 | 35.1 | 19.7 |
| tanh+relu | 0.001 | 93.990 ± 0.250 | 93.883 ± 0.259 | -0.307 | -5.38 | 93.836 | 30.9 | 19.3 |
| tanh+leaky_relu | 0.001 | 94.021 ± 0.316 | 93.917 ± 0.322 | -0.276 | -4.83 | 93.925 | 29.1 | 19.6 |
| sigmoid+relu | 0.003 | 93.772 ± 0.389 | 93.663 ± 0.403 | -0.525 | -9.20 | 93.538 | 33.3 | 19.0 |

MNIST, width 32; 26,506 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.003 | 96.707 ± 0.190 | 96.667 ± 0.193 | +0.000 | +0.00 | 96.477 | 26.9 | 49.9 |
| leaky_relu | 0.003 | 96.683 ± 0.189 | 96.641 ± 0.191 | -0.024 | -0.73 | 96.519 | 26.2 | 53.3 |
| tanh | 0.001 | 96.375 ± 0.160 | 96.322 ± 0.163 | -0.332 | -10.08 | 96.335 | 28.3 | 53.6 |
| sigmoid | 0.003 | 96.269 ± 0.138 | 96.216 ± 0.141 | -0.438 | -13.30 | 96.273 | 34.3 | 52.7 |
| tanh+relu | 0.003 | 96.521 ± 0.185 | 96.475 ± 0.189 | -0.186 | -5.65 | 96.490 | 25.1 | 51.2 |
| tanh+leaky_relu | 0.003 | 96.513 ± 0.159 | 96.465 ± 0.160 | -0.194 | -5.89 | 96.486 | 24.9 | 53.0 |
| sigmoid+relu | 0.01 | 95.989 ± 0.174 | 95.933 ± 0.179 | -0.718 | -21.80 | 95.822 | 31.2 | 50.6 |

MNIST, width 64; 55,050 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.003 | 97.579 ± 0.125 | 97.555 ± 0.129 | +0.000 | +0.00 | 97.608 | 26.1 | 103.4 |
| leaky_relu | 0.003 | 97.560 ± 0.107 | 97.536 ± 0.107 | -0.019 | -0.77 | 97.571 | 29.4 | 108.7 |
| tanh | 0.01 | 97.390 ± 0.122 | 97.359 ± 0.125 | -0.189 | -7.79 | 97.399 | 23.0 | 106.9 |
| sigmoid | 0.01 | 97.123 ± 0.083 | 97.089 ± 0.083 | -0.456 | -18.82 | 97.135 | 29.3 | 107.2 |
| tanh+relu | 0.01 | 97.583 ± 0.126 | 97.555 ± 0.130 | +0.004 | +0.15 | 97.605 | 28.3 | 105.3 |
| tanh+leaky_relu | 0.01 | 97.607 ± 0.116 | 97.581 ± 0.118 | +0.028 | +1.14 | 97.640 | 29.2 | 106.6 |
| sigmoid+relu | 0.003 | 96.709 ± 0.159 | 96.669 ± 0.162 | -0.870 | -35.92 | 96.532 | 24.2 | 101.9 |

MNIST, width 128; 118,282 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.003 | 97.860 ± 0.075 | 97.842 ± 0.075 | +0.000 | +0.00 | 97.880 | 25.1 | 213.7 |
| leaky_relu | 0.003 | 97.849 ± 0.082 | 97.831 ± 0.085 | -0.011 | -0.51 | 97.852 | 25.8 | 230.5 |
| tanh | 0.01 | 97.827 ± 0.094 | 97.804 ± 0.096 | -0.034 | -1.57 | 97.831 | 22.9 | 228.2 |
| sigmoid | 0.03 | 97.658 ± 0.095 | 97.633 ± 0.095 | -0.202 | -9.46 | 97.684 | 23.4 | 223.1 |
| tanh+relu | 0.01 | 98.038 ± 0.073 | 98.020 ± 0.074 | +0.178 | +8.30 | 98.053 | 24.0 | 219.9 |
| tanh+leaky_relu | 0.01 | 97.986 ± 0.081 | 97.967 ± 0.080 | +0.126 | +5.87 | 98.001 | 21.2 | 226.4 |
| sigmoid+relu | 0.003 | 97.550 ± 0.141 | 97.526 ± 0.143 | -0.310 | -14.49 | 97.574 | 26.4 | 213.4 |

MNIST, width 256; 269,322 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.01 | 98.214 ± 0.074 | 98.199 ± 0.074 | +0.000 | +0.00 | 98.228 | 26.9 | 482.5 |
| leaky_relu | 0.01 | 98.176 ± 0.063 | 98.159 ± 0.063 | -0.038 | -2.13 | 98.201 | 29.9 | 508.8 |
| tanh | 0.01 | 98.018 ± 0.073 | 97.998 ± 0.075 | -0.197 | -11.01 | 98.043 | 20.6 | 528.6 |
| sigmoid | 0.1 | 97.966 ± 0.069 | 97.944 ± 0.072 | -0.249 | -13.93 | 97.998 | 24.8 | 498.1 |
| tanh+relu | 0.01 | 98.214 ± 0.075 | 98.199 ± 0.075 | -0.001 | -0.04 | 98.225 | 20.9 | 504.9 |
| tanh+leaky_relu | 0.01 | 98.194 ± 0.071 | 98.178 ± 0.073 | -0.021 | -1.16 | 98.202 | 24.6 | 518.3 |
| sigmoid+relu | 0.01 | 98.037 ± 0.083 | 98.020 ± 0.085 | -0.177 | -9.91 | 98.054 | 33.7 | 478.9 |

MNIST, width 512; 669,706 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.01 | 98.285 ± 0.064 | 98.272 ± 0.065 | +0.000 | +0.00 | 98.300 | 23.8 | 1456.7 |
| leaky_relu | 0.01 | 98.248 ± 0.091 | 98.233 ± 0.092 | -0.037 | -2.18 | 98.276 | 26.1 | 1504.8 |
| tanh | 0.01 | 98.024 ± 0.068 | 98.006 ± 0.070 | -0.261 | -15.24 | 98.049 | 20.0 | 1551.4 |
| sigmoid | 0.1 | 98.042 ± 0.071 | 98.022 ± 0.072 | -0.243 | -14.19 | 98.066 | 26.1 | 1469.7 |
| tanh+relu | 0.01 | 98.289 ± 0.087 | 98.276 ± 0.087 | +0.003 | +0.19 | 98.318 | 24.0 | 1474.4 |
| tanh+leaky_relu | 0.01 | 98.274 ± 0.076 | 98.262 ± 0.077 | -0.011 | -0.64 | 98.289 | 23.2 | 1485.0 |
| sigmoid+relu | 0.03 | 98.116 ± 0.070 | 98.097 ± 0.071 | -0.169 | -9.86 | 98.143 | 25.8 | 1277.2 |

CIFAR10, width 12; 37,162 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.0003 | 43.461 ± 0.745 | 42.753 ± 0.895 | +0.000 | +0.00 | 43.318 | 36.3 | 74.8 |
| leaky_relu | 0.0003 | 43.618 ± 0.880 | 43.017 ± 0.838 | +0.157 | +0.28 | 43.200 | 37.1 | 76.1 |
| tanh | 0.0003 | 42.343 ± 0.562 | 41.618 ± 0.592 | -1.118 | -1.98 | 41.953 | 34.9 | 76.4 |
| sigmoid | 0.003 | 41.360 ± 0.381 | 40.637 ± 0.475 | -2.101 | -3.72 | 40.714 | 34.6 | 76.3 |
| tanh+relu | 0.0003 | 42.911 ± 0.647 | 42.105 ± 0.797 | -0.550 | -0.97 | 42.640 | 36.9 | 76.1 |
| tanh+leaky_relu | 0.0003 | 42.987 ± 0.756 | 42.270 ± 0.907 | -0.474 | -0.84 | 42.651 | 33.8 | 76.3 |
| sigmoid+relu | 0.001 | 42.340 ± 0.861 | 41.537 ± 0.967 | -1.121 | -1.98 | 41.712 | 35.3 | 74.3 |

CIFAR10, width 32; 99,722 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.0003 | 49.186 ± 0.588 | 48.615 ± 0.662 | +0.000 | +0.00 | 48.891 | 36.5 | 191.1 |
| leaky_relu | 0.0003 | 49.254 ± 0.490 | 48.642 ± 0.597 | +0.068 | +0.13 | 48.632 | 36.7 | 192.9 |
| tanh | 0.0003 | 47.894 ± 0.476 | 47.389 ± 0.511 | -1.292 | -2.54 | 47.482 | 35.3 | 193.2 |
| sigmoid | 0.003 | 47.206 ± 0.545 | 46.629 ± 0.593 | -1.980 | -3.90 | 46.354 | 35.3 | 193.0 |
| tanh+relu | 0.0003 | 48.801 ± 0.452 | 48.301 ± 0.487 | -0.385 | -0.76 | 48.274 | 35.4 | 192.5 |
| tanh+leaky_relu | 0.0003 | 48.759 ± 0.448 | 48.152 ± 0.512 | -0.427 | -0.84 | 48.092 | 36.7 | 193.0 |
| sigmoid+relu | 0.001 | 47.774 ± 0.612 | 47.171 ± 0.783 | -1.411 | -2.78 | 47.175 | 37.3 | 187.6 |

CIFAR10, width 64; 201,482 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.001 | 51.239 ± 0.504 | 50.872 ± 0.591 | +0.000 | +0.00 | 50.627 | 32.9 | 371.6 |
| leaky_relu | 0.001 | 51.270 ± 0.491 | 50.904 ± 0.541 | +0.031 | +0.06 | 50.638 | 34.2 | 378.2 |
| tanh | 0.0003 | 50.490 ± 0.389 | 50.075 ± 0.536 | -0.749 | -1.54 | 50.142 | 37.1 | 380.3 |
| sigmoid | 0.003 | 50.013 ± 0.357 | 49.399 ± 0.543 | -1.226 | -2.52 | 49.731 | 37.1 | 377.4 |
| tanh+relu | 0.0003 | 51.603 ± 0.376 | 51.227 ± 0.499 | +0.363 | +0.75 | 51.393 | 37.1 | 376.0 |
| tanh+leaky_relu | 0.0003 | 51.288 ± 0.439 | 50.847 ± 0.427 | +0.048 | +0.10 | 51.057 | 36.0 | 380.0 |
| sigmoid+relu | 0.001 | 50.274 ± 0.427 | 49.798 ± 0.450 | -0.965 | -1.98 | 49.922 | 36.5 | 368.8 |

CIFAR10, width 128; 411,146 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.0003 | 53.043 ± 0.527 | 52.674 ± 0.692 | +0.000 | +0.00 | 52.521 | 36.7 | 751.2 |
| leaky_relu | 0.0003 | 52.953 ± 0.512 | 52.534 ± 0.622 | -0.091 | -0.19 | 52.403 | 37.1 | 770.4 |
| tanh | 0.0003 | 51.930 ± 0.517 | 51.499 ± 0.594 | -1.114 | -2.37 | 51.315 | 36.7 | 772.0 |
| sigmoid | 0.003 | 51.661 ± 0.463 | 51.196 ± 0.609 | -1.382 | -2.94 | 51.183 | 36.0 | 761.9 |
| tanh+relu | 0.0003 | 53.221 ± 0.439 | 52.803 ± 0.492 | +0.178 | +0.38 | 52.749 | 35.9 | 761.3 |
| tanh+leaky_relu | 0.0003 | 53.086 ± 0.521 | 52.610 ± 0.569 | +0.042 | +0.09 | 52.494 | 36.8 | 771.6 |
| sigmoid+relu | 0.0003 | 51.902 ± 0.453 | 51.481 ± 0.607 | -1.142 | -2.43 | 51.329 | 37.4 | 744.0 |

CIFAR10, width 256; 855,050 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.0003 | 54.074 ± 0.506 | 53.680 ± 0.547 | +0.000 | +0.00 | 53.694 | 36.6 | 2240.3 |
| leaky_relu | 0.0003 | 53.852 ± 0.482 | 53.513 ± 0.608 | -0.222 | -0.48 | 53.341 | 37.8 | 2233.8 |
| tanh | 0.001 | 52.489 ± 0.496 | 52.053 ± 0.627 | -1.585 | -3.45 | 51.908 | 32.1 | 2263.7 |
| sigmoid | 0.003 | 52.777 ± 0.534 | 52.335 ± 0.755 | -1.296 | -2.82 | 52.128 | 36.9 | 2223.3 |
| tanh+relu | 0.0003 | 54.227 ± 0.459 | 53.862 ± 0.515 | +0.153 | +0.33 | 53.884 | 37.5 | 2233.3 |
| tanh+leaky_relu | 0.0003 | 53.983 ± 0.405 | 53.489 ± 0.593 | -0.090 | -0.20 | 53.511 | 37.5 | 2230.7 |
| sigmoid+relu | 0.0003 | 53.481 ± 0.500 | 53.063 ± 0.666 | -0.592 | -1.29 | 53.164 | 37.4 | 1777.4 |

CIFAR10, width 512; 1,841,162 parameters

| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relu | 0.0003 | 54.875 ± 0.469 | 54.577 ± 0.630 | +0.000 | +0.00 | 54.333 | 36.9 | 8400.7 |
| leaky_relu | 0.0003 | 54.607 ± 0.424 | 54.188 ± 0.595 | -0.269 | -0.60 | 53.995 | 36.8 | 8387.9 |
| tanh | 0.001 | 53.190 ± 0.487 | 52.896 ± 0.623 | -1.685 | -3.73 | 52.187 | 33.0 | 8401.8 |
| sigmoid | 0.003 | 53.137 ± 0.530 | 52.829 ± 0.682 | -1.739 | -3.85 | 52.889 | 36.9 | 8339.1 |
| tanh+relu | 0.0003 | 54.711 ± 0.430 | 54.347 ± 0.553 | -0.164 | -0.36 | 54.361 | 36.7 | 8300.5 |
| tanh+leaky_relu | 0.0003 | 54.474 ± 0.372 | 54.213 ± 0.480 | -0.401 | -0.89 | 53.899 | 38.1 | 8351.5 |
| sigmoid+relu | 0.0003 | 54.357 ± 0.513 | 53.996 ± 0.659 | -0.518 | -1.15 | 53.940 | 35.9 | 5143.8 |

Layout and MSE checkpoint comparisons

| Stage | Width | Config | Layout | Rectifier fraction | LR | n | Acc ± SD (%) | F1 (%) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| layout | 128 | tanh+leaky_relu | block | 0.5 | 0.01 | 30 | 98.022 ± 0.086 | 98.003 |
| layout | 128 | tanh+leaky_relu | interleave | 0.25 | 0.01 | 30 | 97.965 ± 0.096 | 97.944 |
| layout | 128 | tanh+leaky_relu | interleave | 0.5 | 0.01 | 30 | 97.986 ± 0.081 | 97.967 |
| layout | 128 | tanh+leaky_relu | interleave | 0.75 | 0.01 | 30 | 98.052 ± 0.084 | 98.035 |
| layout | 128 | tanh+leaky_relu | random | 0.5 | 0.01 | 30 | 98.017 ± 0.071 | 98.000 |
| layout | 128 | tanh+relu | block | 0.5 | 0.01 | 30 | 98.020 ± 0.088 | 98.001 |
| layout | 128 | tanh+relu | interleave | 0.25 | 0.01 | 30 | 97.957 ± 0.099 | 97.937 |
| layout | 128 | tanh+relu | interleave | 0.5 | 0.01 | 30 | 98.038 ± 0.073 | 98.020 |
| layout | 128 | tanh+relu | interleave | 0.75 | 0.01 | 30 | 98.052 ± 0.079 | 98.036 |
| layout | 128 | tanh+relu | random | 0.5 | 0.01 | 30 | 98.034 ± 0.108 | 98.016 |
| mse | 64 | relu | interleave | 0 | 0.01 | 30 | 97.637 ± 0.133 | 97.612 |
| mse | 64 | tanh | interleave | 0 | 0.01 | 30 | 97.266 ± 0.132 | 97.232 |
| mse | 64 | tanh+relu | interleave | 0.5 | 0.01 | 30 | 97.435 ± 0.119 | 97.407 |
| mse | 128 | relu | interleave | 0 | 0.01 | 30 | 97.998 ± 0.080 | 97.981 |
| mse | 128 | tanh | interleave | 0 | 0.01 | 30 | 97.777 ± 0.070 | 97.752 |
| mse | 128 | tanh+relu | interleave | 0.5 | 0.03 | 30 | 97.964 ± 0.088 | 97.943 |