# version5 results (extended sweep): hybrid activations across width, learning rate and layout

Protocol: MLP D→H→H→10 (D = 784 on MNIST, 3072 on CIFAR-10), batch-size-1 SGD, 40 epochs with early stopping on validation accuracy, fixed 45k/5k/10k train/val/test split shared by all runs, per-epoch shuffling, He/Kaiming-uniform init. Hidden layers carry per-neuron activation masks; the output layer is uniform (linear+softmax with cross-entropy, or sigmoid with MSE). Learning rate chosen per configuration on validation accuracy using calibration seeds (101–106) that are disjoint from the evaluation seeds (1–30). All comparisons are paired by seed. Bootstrap CIs use 10,000 resamples and a collapsed run means final validation accuracy below 50%.

## 0. Files used

| stage | width | file | superseded |
|---|---|---|---|
| main | 12 | `main_12.csv` | — |
| main | 32 | `main_32.csv` | — |
| main | 64 | `main_64.csv` | — |
| main | 128 | `main_128.csv` | — |
| main | 256 | `main_256.csv` | — |
| main | 512 | `main_512.csv` | — |
| calib | 12 | `calib_12.csv` | — |
| calib | 32 | `calib_32.csv` | — |
| calib | 64 | `calib_64.csv` | — |
| calib | 128 | `calib_128.csv` | — |
| calib | 256 | `calib_256.csv` | — |
| calib | 512 | `calib_512.csv` | — |
| lrsweep | 64 | `lrsweep_64.csv` | — |
| lrsweep | 128 | `lrsweep_128.csv` | — |
| layout | 128 | `layout_128.csv` | — |
| main (CIFAR-10) | 12 | `c10_main_12.csv` | — |
| main (CIFAR-10) | 32 | `c10_main_32.csv` | — |
| main (CIFAR-10) | 64 | `c10_main_64.csv` | — |
| main (CIFAR-10) | 128 | `c10_main_128.csv` | — |
| main (CIFAR-10) | 256 | `c10_main_256.csv` | — |
| main (CIFAR-10) | 512 | `c10_main_512.csv` | — |
| calib (CIFAR-10) | 12 | `c10_calib_12.csv` | — |
| calib (CIFAR-10) | 32 | `c10_calib_32.csv` | — |
| calib (CIFAR-10) | 64 | `c10_calib_64.csv` | — |
| calib (CIFAR-10) | 128 | `c10_calib_128.csv` | — |
| calib (CIFAR-10) | 256 | `c10_calib_256.csv` | — |
| calib (CIFAR-10) | 512 | `c10_calib_512.csv` | — |

## 1. Run inventory (evaluation runs, config × width → n)

Every width is reported at the same seed count, n = 30 (seeds 1–30), so the width axis compares like with like.

| config | W=12 | W=32 | W=64 | W=128 | W=256 | W=512 |
|---|---|---|---|---|---|---|
| relu | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh | 30 | 30 | 30 | 30 | 30 | 30 |
| leaky_relu | 30 | 30 | 30 | 30 | 30 | 30 |
| sigmoid | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh+relu | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh+leaky_relu | 30 | 30 | 30 | 30 | 30 | 30 |
| sigmoid+relu | 30 | 30 | 30 | 30 | 30 | 30 |

| other stage | file | ok runs |
|---|---|---|
| calib | `calib_12.csv` | 336 |
| calib | `calib_32.csv` | 336 |
| calib | `calib_64.csv` | 336 |
| calib | `calib_128.csv` | 336 |
| calib | `calib_256.csv` | 336 |
| calib | `calib_512.csv` | 336 |
| lrsweep | `lrsweep_64.csv` | 2310 |
| lrsweep | `lrsweep_128.csv` | 2310 |
| layout | `layout_128.csv` | 300 |
| mse | `mse_128.csv` | 90 |
| calib | `calib_mse_128.csv` | 144 |

## 2.12 MNIST, 12 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.001 | 30 | 94.11 ± 0.39 | 94.19 | 94.01 ± 0.40 | 94.39 |
| tanh | 0.001 | 30 | 93.82 ± 0.24 | 93.84 | 93.71 ± 0.25 | 94.20 |
| leaky_relu | 0.001 | 30 | 94.15 ± 0.31 | 94.19 | 94.05 ± 0.31 | 94.48 |
| sigmoid | 0.003 | 30 | 93.74 ± 0.33 | 93.78 | 93.63 ± 0.33 | 93.95 |
| tanh+relu | 0.001 | 30 | 93.84 ± 0.29 | 93.80 | 93.73 ± 0.29 | 94.06 |
| tanh+leaky_relu | 0.001 | 30 | 93.92 ± 0.36 | 93.93 | 93.82 ± 0.36 | 94.11 |
| sigmoid+relu | 0.003 | 30 | 93.54 ± 0.46 | 93.49 | 93.44 ± 0.46 | 93.72 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.27 | [-0.45, -0.09] | 9/30 | 0.007 | 0.009 |
| tanh+relu vs relu | macro-F1 | 30 | -0.28 | [-0.47, -0.09] | 10/30 | 0.007 | 0.008 |
| tanh+relu vs tanh | accuracy | 30 | +0.02 | [-0.11, +0.15] | 16/30 | 0.804 | 0.812 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.01 | [-0.12, +0.14] | 17/30 | 0.857 | 0.824 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.23 | [-0.39, -0.05] | 7/30 | 0.014 | 0.017 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.23 | [-0.39, -0.06] | 7/30 | 0.015 | 0.016 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.11 | [-0.03, +0.23] | 19/30 | 0.132 | 0.141 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.11 | [-0.03, +0.24] | 19/30 | 0.136 | 0.158 |
| sigmoid+relu vs relu | accuracy | 30 | -0.57 | [-0.73, -0.41] | 3/30 | 1.5e-07 | 6.3e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.57 | [-0.74, -0.42] | 3/30 | 1.8e-07 | 2.0e-07 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.20 | [-0.39, -0.01] | 10/30 | 0.054 | 0.076 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.19 | [-0.39, -0.01] | 11/30 | 0.062 | 0.086 |

![](figures/fig_acc_12.png)
![](figures/fig_f1_12.png)
![](figures/fig_paired_acc_12.png)
![](figures/fig_paired_f1_12.png)
![](figures/fig_curves_12.png)

## 2.32 MNIST, 32 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 30 | 96.48 ± 0.25 | 96.56 | 96.43 ± 0.26 | 96.65 |
| tanh | 0.001 | 30 | 96.34 ± 0.17 | 96.33 | 96.28 ± 0.17 | 96.65 |
| leaky_relu | 0.003 | 30 | 96.52 ± 0.20 | 96.56 | 96.48 ± 0.20 | 96.65 |
| sigmoid | 0.003 | 30 | 96.27 ± 0.14 | 96.31 | 96.22 ± 0.14 | 96.49 |
| tanh+relu | 0.003 | 30 | 96.49 ± 0.21 | 96.48 | 96.44 ± 0.22 | 96.66 |
| tanh+leaky_relu | 0.003 | 30 | 96.49 ± 0.14 | 96.50 | 96.44 ± 0.14 | 96.67 |
| sigmoid+relu | 0.01 | 30 | 95.82 ± 0.25 | 95.82 | 95.76 ± 0.26 | 96.01 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.01 | [-0.10, +0.13] | 16/30 | 0.821 | 0.934 |
| tanh+relu vs relu | macro-F1 | 30 | +0.01 | [-0.10, +0.14] | 16/30 | 0.846 | 0.984 |
| tanh+relu vs tanh | accuracy | 30 | +0.16 | [+0.07, +0.25] | 23/30 | 0.003 | 0.004 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.16 | [+0.07, +0.26] | 23/30 | 0.002 | 0.002 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.03 | [-0.10, +0.04] | 14/30 | 0.403 | 0.271 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.04 | [-0.11, +0.04] | 14/30 | 0.330 | 0.229 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.15 | [+0.10, +0.21] | 24/30 | 2.3e-05 | 1.1e-04 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.16 | [+0.10, +0.21] | 26/30 | 1.7e-05 | 1.4e-05 |
| sigmoid+relu vs relu | accuracy | 30 | -0.66 | [-0.78, -0.53] | 1/30 | 3.3e-11 | 2.1e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.67 | [-0.79, -0.55] | 1/30 | 3.6e-11 | 3.7e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.45 | [-0.54, -0.36] | 1/30 | 1.4e-10 | 2.3e-06 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.46 | [-0.55, -0.37] | 1/30 | 1.4e-10 | 9.3e-09 |

![](figures/fig_acc_32.png)
![](figures/fig_f1_32.png)
![](figures/fig_paired_acc_32.png)
![](figures/fig_paired_f1_32.png)
![](figures/fig_curves_32.png)

## 2.64 MNIST, 64 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 30 | 97.61 ± 0.12 | 97.60 | 97.58 ± 0.12 | 97.78 |
| tanh | 0.01 | 30 | 97.40 ± 0.11 | 97.40 | 97.37 ± 0.11 | 97.66 |
| leaky_relu | 0.003 | 30 | 97.57 ± 0.10 | 97.57 | 97.55 ± 0.10 | 97.78 |
| sigmoid | 0.01 | 30 | 97.14 ± 0.11 | 97.14 | 97.10 ± 0.11 | 97.46 |
| tanh+relu | 0.01 | 30 | 97.61 ± 0.10 | 97.59 | 97.58 ± 0.10 | 97.90 |
| tanh+leaky_relu | 0.01 | 30 | 97.64 ± 0.08 | 97.64 | 97.61 ± 0.08 | 97.86 |
| sigmoid+relu | 0.003 | 30 | 96.53 ± 0.25 | 96.60 | 96.49 ± 0.26 | 96.72 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.00 | [-0.06, +0.06] | 11/30 | 0.920 | 0.820 |
| tanh+relu vs relu | macro-F1 | 30 | -0.01 | [-0.07, +0.05] | 11/30 | 0.830 | 0.715 |
| tanh+relu vs tanh | accuracy | 30 | +0.21 | [+0.15, +0.26] | 28/30 | 2.3e-08 | 8.0e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.21 | [+0.16, +0.26] | 28/30 | 1.8e-08 | 1.6e-07 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.07 | [+0.02, +0.11] | 23/30 | 0.006 | 0.004 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.07 | [+0.02, +0.11] | 23/30 | 0.008 | 0.003 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.24 | [+0.19, +0.29] | 28/30 | 1.7e-09 | 5.2e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.25 | [+0.19, +0.30] | 28/30 | 1.3e-09 | 1.0e-07 |
| sigmoid+relu vs relu | accuracy | 30 | -1.08 | [-1.17, -0.99] | 0/30 | 9.8e-20 | 1.7e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.10 | [-1.19, -1.00] | 0/30 | 1.2e-19 | 1.9e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.60 | [-0.70, -0.51] | 0/30 | 7.3e-13 | 1.7e-06 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.61 | [-0.71, -0.52] | 0/30 | 9.3e-13 | 1.9e-09 |

![](figures/fig_acc_64.png)
![](figures/fig_f1_64.png)
![](figures/fig_paired_acc_64.png)
![](figures/fig_paired_f1_64.png)
![](figures/fig_curves_64.png)

## 2.128 MNIST, 128 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 30 | 97.88 ± 0.08 | 97.88 | 97.86 ± 0.08 | 98.12 |
| tanh | 0.01 | 30 | 97.83 ± 0.11 | 97.83 | 97.81 ± 0.11 | 98.14 |
| leaky_relu | 0.003 | 30 | 97.85 ± 0.09 | 97.87 | 97.83 ± 0.09 | 98.12 |
| sigmoid | 0.03 | 30 | 97.68 ± 0.10 | 97.68 | 97.66 ± 0.10 | 98.03 |
| tanh+relu | 0.01 | 30 | 98.05 ± 0.07 | 98.04 | 98.03 ± 0.07 | 98.28 |
| tanh+leaky_relu | 0.01 | 30 | 98.00 ± 0.05 | 98.02 | 97.98 ± 0.05 | 98.28 |
| sigmoid+relu | 0.003 | 30 | 97.57 ± 0.11 | 97.57 | 97.55 ± 0.12 | 97.75 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.17 | [+0.14, +0.21] | 29/30 | 3.9e-10 | 2.0e-06 |
| tanh+relu vs relu | macro-F1 | 30 | +0.17 | [+0.14, +0.21] | 29/30 | 5.7e-10 | 5.6e-09 |
| tanh+relu vs tanh | accuracy | 30 | +0.22 | [+0.18, +0.26] | 28/30 | 6.5e-11 | 3.1e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.23 | [+0.18, +0.27] | 28/30 | 5.2e-11 | 9.3e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.15 | [+0.12, +0.18] | 28/30 | 3.3e-10 | 3.0e-06 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.15 | [+0.12, +0.18] | 28/30 | 8.4e-10 | 1.3e-08 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.17 | [+0.12, +0.21] | 27/30 | 6.1e-08 | 9.7e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.17 | [+0.13, +0.22] | 27/30 | 4.9e-08 | 1.3e-07 |
| sigmoid+relu vs relu | accuracy | 30 | -0.31 | [-0.35, -0.26] | 0/30 | 7.7e-14 | 1.7e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.31 | [-0.35, -0.26] | 0/30 | 8.2e-14 | 1.9e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.11 | [-0.16, -0.06] | 5/30 | 9.5e-05 | 2.9e-04 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.11 | [-0.15, -0.06] | 6/30 | 1.6e-04 | 1.5e-04 |

![](figures/fig_acc_128.png)
![](figures/fig_f1_128.png)
![](figures/fig_paired_acc_128.png)
![](figures/fig_paired_f1_128.png)
![](figures/fig_curves_128.png)

## 2.256 MNIST, 256 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.01 | 30 | 98.23 ± 0.06 | 98.23 | 98.21 ± 0.06 | 98.45 |
| tanh | 0.01 | 30 | 98.04 ± 0.07 | 98.06 | 98.02 ± 0.07 | 98.28 |
| leaky_relu | 0.01 | 30 | 98.20 ± 0.06 | 98.21 | 98.18 ± 0.06 | 98.48 |
| sigmoid | 0.1 | 30 | 98.00 ± 0.06 | 98.00 | 97.98 ± 0.06 | 98.22 |
| tanh+relu | 0.01 | 30 | 98.22 ± 0.07 | 98.22 | 98.21 ± 0.07 | 98.46 |
| tanh+leaky_relu | 0.01 | 30 | 98.20 ± 0.07 | 98.22 | 98.19 ± 0.07 | 98.46 |
| sigmoid+relu | 0.01 | 30 | 98.05 ± 0.12 | 98.08 | 98.04 ± 0.12 | 98.26 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.00 | [-0.03, +0.03] | 13/30 | 0.866 | 0.795 |
| tanh+relu vs relu | macro-F1 | 30 | -0.00 | [-0.03, +0.03] | 14/30 | 0.902 | 0.871 |
| tanh+relu vs tanh | accuracy | 30 | +0.18 | [+0.14, +0.22] | 30/30 | 6.3e-10 | 1.7e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.19 | [+0.15, +0.23] | 30/30 | 6.0e-10 | 1.9e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.00 | [-0.03, +0.03] | 15/30 | 0.984 | 0.943 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.00 | [-0.03, +0.04] | 15/30 | 0.903 | 0.839 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.16 | [+0.12, +0.20] | 28/30 | 1.7e-08 | 4.5e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.16 | [+0.12, +0.20] | 28/30 | 1.7e-08 | 1.0e-07 |
| sigmoid+relu vs relu | accuracy | 30 | -0.17 | [-0.23, -0.13] | 2/30 | 2.9e-07 | 3.9e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.18 | [-0.23, -0.13] | 3/30 | 4.1e-07 | 3.5e-08 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +0.06 | [+0.01, +0.10] | 22/30 | 0.016 | 0.007 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.06 | [+0.01, +0.10] | 22/30 | 0.014 | 0.005 |

![](figures/fig_acc_256.png)
![](figures/fig_f1_256.png)
![](figures/fig_paired_acc_256.png)
![](figures/fig_paired_f1_256.png)
![](figures/fig_curves_256.png)

## 2.512 MNIST, 512 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.01 | 30 | 98.30 ± 0.05 | 98.29 | 98.29 ± 0.05 | 98.50 |
| tanh | 0.01 | 30 | 98.05 ± 0.07 | 98.05 | 98.03 ± 0.07 | 98.31 |
| leaky_relu | 0.01 | 30 | 98.28 ± 0.07 | 98.27 | 98.26 ± 0.07 | 98.56 |
| sigmoid | 0.1 | 30 | 98.07 ± 0.07 | 98.07 | 98.05 ± 0.07 | 98.33 |
| tanh+relu | 0.01 | 30 | 98.32 ± 0.07 | 98.31 | 98.30 ± 0.07 | 98.56 |
| tanh+leaky_relu | 0.01 | 30 | 98.29 ± 0.07 | 98.28 | 98.28 ± 0.07 | 98.54 |
| sigmoid+relu | 0.03 | 30 | 98.14 ± 0.06 | 98.13 | 98.12 ± 0.06 | 98.42 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.02 | [-0.01, +0.05] | 17/30 | 0.230 | 0.236 |
| tanh+relu vs relu | macro-F1 | 30 | +0.02 | [-0.01, +0.05] | 17/30 | 0.219 | 0.198 |
| tanh+relu vs tanh | accuracy | 30 | +0.27 | [+0.23, +0.31] | 30/30 | 3.9e-14 | 1.7e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.27 | [+0.24, +0.31] | 30/30 | 3.0e-14 | 1.9e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.01 | [-0.02, +0.05] | 16/30 | 0.430 | 0.544 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.02 | [-0.02, +0.05] | 17/30 | 0.355 | 0.477 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.24 | [+0.21, +0.27] | 30/30 | 7.5e-14 | 1.7e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.25 | [+0.21, +0.28] | 30/30 | 5.6e-14 | 1.9e-09 |
| sigmoid+relu vs relu | accuracy | 30 | -0.16 | [-0.19, -0.13] | 0/30 | 3.3e-11 | 1.7e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.16 | [-0.19, -0.13] | 0/30 | 2.5e-11 | 1.9e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +0.08 | [+0.05, +0.10] | 24/30 | 9.6e-06 | 9.8e-05 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.08 | [+0.05, +0.11] | 25/30 | 9.8e-06 | 8.2e-05 |

![](figures/fig_acc_512.png)
![](figures/fig_f1_512.png)
![](figures/fig_paired_acc_512.png)
![](figures/fig_paired_f1_512.png)
![](figures/fig_curves_512.png)

## 3. MNIST crossover: paired hybrid − parent against width

One row per comparison and width. Δ is the paired mean difference in points (positive = the hybrid is ahead), CI is the 95% bootstrap interval.

**accuracy**

| comparison | width | n | Δ | 95% CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.27 | [-0.45, -0.09] | 9/30 | 0.007 | 0.009 |
| tanh+relu − relu | 32 | 30 | +0.01 | [-0.09, +0.13] | 16/30 | 0.821 | 0.934 |
| tanh+relu − relu | 64 | 30 | -0.00 | [-0.06, +0.05] | 11/30 | 0.920 | 0.820 |
| tanh+relu − relu | 128 | 30 | +0.17 | [+0.14, +0.21] | 29/30 | 3.9e-10 | 2.0e-06 |
| tanh+relu − relu | 256 | 30 | -0.00 | [-0.03, +0.03] | 13/30 | 0.866 | 0.795 |
| tanh+relu − relu | 512 | 30 | +0.02 | [-0.01, +0.05] | 17/30 | 0.230 | 0.236 |
| tanh+relu − tanh | 12 | 30 | +0.02 | [-0.11, +0.14] | 16/30 | 0.804 | 0.812 |
| tanh+relu − tanh | 32 | 30 | +0.16 | [+0.07, +0.25] | 23/30 | 0.003 | 0.004 |
| tanh+relu − tanh | 64 | 30 | +0.21 | [+0.15, +0.26] | 28/30 | 2.3e-08 | 8.0e-06 |
| tanh+relu − tanh | 128 | 30 | +0.22 | [+0.18, +0.26] | 28/30 | 6.5e-11 | 3.1e-06 |
| tanh+relu − tanh | 256 | 30 | +0.18 | [+0.14, +0.22] | 30/30 | 6.3e-10 | 1.7e-06 |
| tanh+relu − tanh | 512 | 30 | +0.27 | [+0.23, +0.31] | 30/30 | 3.9e-14 | 1.7e-06 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.23 | [-0.39, -0.05] | 7/30 | 0.014 | 0.017 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.03 | [-0.11, +0.04] | 14/30 | 0.403 | 0.271 |
| tanh+leaky_relu − leaky_relu | 64 | 30 | +0.07 | [+0.02, +0.11] | 23/30 | 0.006 | 0.004 |
| tanh+leaky_relu − leaky_relu | 128 | 30 | +0.15 | [+0.12, +0.18] | 28/30 | 3.3e-10 | 3.0e-06 |
| tanh+leaky_relu − leaky_relu | 256 | 30 | +0.00 | [-0.03, +0.03] | 15/30 | 0.984 | 0.943 |
| tanh+leaky_relu − leaky_relu | 512 | 30 | +0.01 | [-0.02, +0.05] | 16/30 | 0.430 | 0.544 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.11 | [-0.03, +0.24] | 19/30 | 0.132 | 0.141 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.15 | [+0.09, +0.21] | 24/30 | 2.3e-05 | 1.1e-04 |
| tanh+leaky_relu − tanh | 64 | 30 | +0.24 | [+0.18, +0.29] | 28/30 | 1.7e-09 | 5.2e-06 |
| tanh+leaky_relu − tanh | 128 | 30 | +0.17 | [+0.13, +0.22] | 27/30 | 6.1e-08 | 9.7e-06 |
| tanh+leaky_relu − tanh | 256 | 30 | +0.16 | [+0.12, +0.20] | 28/30 | 1.7e-08 | 4.5e-06 |
| tanh+leaky_relu − tanh | 512 | 30 | +0.24 | [+0.20, +0.28] | 30/30 | 7.5e-14 | 1.7e-06 |

**macro-F1**

| comparison | width | n | Δ | 95% CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.28 | [-0.47, -0.09] | 10/30 | 0.007 | 0.008 |
| tanh+relu − relu | 32 | 30 | +0.01 | [-0.10, +0.14] | 16/30 | 0.846 | 0.984 |
| tanh+relu − relu | 64 | 30 | -0.01 | [-0.06, +0.05] | 11/30 | 0.830 | 0.715 |
| tanh+relu − relu | 128 | 30 | +0.17 | [+0.14, +0.21] | 29/30 | 5.7e-10 | 5.6e-09 |
| tanh+relu − relu | 256 | 30 | -0.00 | [-0.03, +0.03] | 14/30 | 0.902 | 0.871 |
| tanh+relu − relu | 512 | 30 | +0.02 | [-0.01, +0.05] | 17/30 | 0.219 | 0.198 |
| tanh+relu − tanh | 12 | 30 | +0.01 | [-0.12, +0.14] | 17/30 | 0.857 | 0.824 |
| tanh+relu − tanh | 32 | 30 | +0.16 | [+0.07, +0.26] | 23/30 | 0.002 | 0.002 |
| tanh+relu − tanh | 64 | 30 | +0.21 | [+0.16, +0.26] | 28/30 | 1.8e-08 | 1.6e-07 |
| tanh+relu − tanh | 128 | 30 | +0.23 | [+0.18, +0.27] | 28/30 | 5.2e-11 | 9.3e-09 |
| tanh+relu − tanh | 256 | 30 | +0.19 | [+0.15, +0.23] | 30/30 | 6.0e-10 | 1.9e-09 |
| tanh+relu − tanh | 512 | 30 | +0.27 | [+0.23, +0.31] | 30/30 | 3.0e-14 | 1.9e-09 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.23 | [-0.40, -0.06] | 7/30 | 0.015 | 0.016 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.04 | [-0.11, +0.04] | 14/30 | 0.330 | 0.229 |
| tanh+leaky_relu − leaky_relu | 64 | 30 | +0.07 | [+0.02, +0.11] | 23/30 | 0.008 | 0.003 |
| tanh+leaky_relu − leaky_relu | 128 | 30 | +0.15 | [+0.12, +0.18] | 28/30 | 8.4e-10 | 1.3e-08 |
| tanh+leaky_relu − leaky_relu | 256 | 30 | +0.00 | [-0.03, +0.04] | 15/30 | 0.903 | 0.839 |
| tanh+leaky_relu − leaky_relu | 512 | 30 | +0.02 | [-0.02, +0.05] | 17/30 | 0.355 | 0.477 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.11 | [-0.03, +0.24] | 19/30 | 0.136 | 0.158 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.16 | [+0.10, +0.21] | 26/30 | 1.7e-05 | 1.4e-05 |
| tanh+leaky_relu − tanh | 64 | 30 | +0.25 | [+0.19, +0.30] | 28/30 | 1.3e-09 | 1.0e-07 |
| tanh+leaky_relu − tanh | 128 | 30 | +0.17 | [+0.13, +0.22] | 27/30 | 4.9e-08 | 1.3e-07 |
| tanh+leaky_relu − tanh | 256 | 30 | +0.16 | [+0.12, +0.20] | 28/30 | 1.7e-08 | 1.0e-07 |
| tanh+leaky_relu − tanh | 512 | 30 | +0.25 | [+0.21, +0.28] | 30/30 | 5.6e-14 | 1.9e-09 |

![](figures/fig_crossover.png)
![](figures/fig_crossover_f1.png)
![](figures/fig_width_acc.png)

## 3c. CIFAR-10 replication

The same protocol on a harder dataset: 3072 inputs instead of 784, the official 10 000-image test batch instead of a slice of the training file, and only the 5 000 validation images drawn out of the 50 000 training images by the same fixed `SPLIT_SEED`. Learning rates are calibrated separately for CIFAR-10 (`results/best_lr_c10.json`) because they do not transfer between datasets. Absolute accuracy is far below a convolutional network's — an MLP on raw pixels is not a competitive CIFAR-10 model — which does not affect the paired within-dataset comparisons this report is built on.

| config | W=12 | W=32 | W=64 | W=128 | W=256 | W=512 |
|---|---|---|---|---|---|---|
| relu | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh | 30 | 30 | 30 | 30 | 30 | 30 |
| leaky_relu | 30 | 30 | 30 | 30 | 30 | 30 |
| sigmoid | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh+relu | 30 | 30 | 30 | 30 | 30 | 30 |
| tanh+leaky_relu | 30 | 30 | 30 | 30 | 30 | 30 |
| sigmoid+relu | 30 | 30 | 30 | 30 | 30 | 30 |

## 3.1.12 CIFAR-10, 12 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.0003 | 30 | 43.32 ± 0.89 | 43.28 | 42.62 ± 0.99 | 42.72 |
| tanh | 0.0003 | 30 | 41.95 ± 0.68 | 41.88 | 41.15 ± 0.71 | 41.85 |
| leaky_relu | 0.0003 | 30 | 43.20 ± 0.86 | 43.25 | 42.44 ± 0.94 | 42.88 |
| sigmoid | 0.003 | 30 | 40.71 ± 0.75 | 40.83 | 39.98 ± 0.83 | 40.59 |
| tanh+relu | 0.0003 | 30 | 42.64 ± 0.74 | 42.70 | 41.84 ± 0.90 | 42.29 |
| tanh+leaky_relu | 0.0003 | 30 | 42.65 ± 0.95 | 42.92 | 41.90 ± 1.00 | 42.20 |
| sigmoid+relu | 0.001 | 30 | 41.71 ± 1.18 | 41.79 | 40.89 ± 1.31 | 41.10 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.68 | [-0.99, -0.36] | 9/30 | 2.7e-04 | 0.002 |
| tanh+relu vs relu | macro-F1 | 30 | -0.78 | [-1.17, -0.39] | 11/30 | 5.3e-04 | 0.003 |
| tanh+relu vs tanh | accuracy | 30 | +0.69 | [+0.39, +0.98] | 22/30 | 8.9e-05 | 2.4e-04 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.69 | [+0.39, +0.99] | 22/30 | 1.2e-04 | 1.4e-04 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.55 | [-0.95, -0.15] | 8/30 | 0.013 | 0.018 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.54 | [-0.95, -0.11] | 12/30 | 0.018 | 0.020 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.70 | [+0.36, +1.04] | 21/30 | 4.7e-04 | 0.001 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.75 | [+0.35, +1.13] | 25/30 | 8.3e-04 | 0.001 |
| sigmoid+relu vs relu | accuracy | 30 | -1.61 | [-2.09, -1.13] | 5/30 | 6.3e-07 | 2.6e-05 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.73 | [-2.27, -1.19] | 5/30 | 8.6e-07 | 1.7e-06 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +1.00 | [+0.56, +1.39] | 23/30 | 7.5e-05 | 3.9e-04 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.91 | [+0.41, +1.36] | 23/30 | 0.001 | 6.7e-04 |

![](figures/fig_acc_12_c10.png)
![](figures/fig_f1_12_c10.png)
![](figures/fig_paired_acc_12_c10.png)
![](figures/fig_paired_f1_12_c10.png)
![](figures/fig_curves_12_c10.png)

## 3.1.32 CIFAR-10, 32 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.0003 | 30 | 48.89 ± 0.83 | 48.89 | 48.38 ± 0.89 | 48.05 |
| tanh | 0.0003 | 30 | 47.48 ± 0.65 | 47.55 | 47.03 ± 0.65 | 47.01 |
| leaky_relu | 0.0003 | 30 | 48.63 ± 0.70 | 48.61 | 48.10 ± 0.67 | 47.91 |
| sigmoid | 0.003 | 30 | 46.35 ± 1.17 | 46.45 | 45.77 ± 1.27 | 46.04 |
| tanh+relu | 0.0003 | 30 | 48.27 ± 0.89 | 48.41 | 47.76 ± 0.91 | 47.78 |
| tanh+leaky_relu | 0.0003 | 30 | 48.09 ± 1.08 | 48.24 | 47.66 ± 1.02 | 47.33 |
| sigmoid+relu | 0.001 | 30 | 47.17 ± 0.75 | 47.21 | 46.65 ± 0.79 | 46.46 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.62 | [-0.89, -0.35] | 6/30 | 1.5e-04 | 5.5e-04 |
| tanh+relu vs relu | macro-F1 | 30 | -0.62 | [-0.92, -0.31] | 5/30 | 5.1e-04 | 5.5e-04 |
| tanh+relu vs tanh | accuracy | 30 | +0.79 | [+0.48, +1.08] | 25/30 | 1.7e-05 | 2.4e-04 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.73 | [+0.42, +1.01] | 24/30 | 5.7e-05 | 2.6e-04 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.54 | [-0.81, -0.29] | 10/30 | 3.4e-04 | 8.0e-04 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.45 | [-0.73, -0.16] | 10/30 | 0.005 | 0.009 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.61 | [+0.29, +0.92] | 24/30 | 7.8e-04 | 0.002 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.63 | [+0.30, +0.95] | 25/30 | 9.7e-04 | 0.001 |
| sigmoid+relu vs relu | accuracy | 30 | -1.72 | [-1.94, -1.49] | 0/30 | 7.9e-15 | 1.9e-09 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.73 | [-1.98, -1.46] | 0/30 | 2.1e-13 | 1.9e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +0.82 | [+0.44, +1.21] | 23/30 | 2.5e-04 | 6.7e-04 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.88 | [+0.46, +1.34] | 23/30 | 6.0e-04 | 2.6e-04 |

![](figures/fig_acc_32_c10.png)
![](figures/fig_f1_32_c10.png)
![](figures/fig_paired_acc_32_c10.png)
![](figures/fig_paired_f1_32_c10.png)
![](figures/fig_curves_32_c10.png)

## 3.1.64 CIFAR-10, 64 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.001 | 30 | 50.63 ± 0.62 | 50.63 | 50.23 ± 0.69 | 50.04 |
| tanh | 0.0003 | 30 | 50.14 ± 0.50 | 50.15 | 49.59 ± 0.71 | 49.67 |
| leaky_relu | 0.001 | 30 | 50.64 ± 0.65 | 50.70 | 50.23 ± 0.82 | 50.00 |
| sigmoid | 0.003 | 30 | 49.73 ± 0.50 | 49.79 | 49.24 ± 0.61 | 49.24 |
| tanh+relu | 0.0003 | 30 | 51.39 ± 0.67 | 51.47 | 50.92 ± 0.94 | 50.70 |
| tanh+leaky_relu | 0.0003 | 30 | 51.06 ± 0.73 | 51.05 | 50.47 ± 0.92 | 50.33 |
| sigmoid+relu | 0.001 | 30 | 49.92 ± 0.71 | 50.06 | 49.42 ± 0.74 | 49.11 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.77 | [+0.45, +1.10] | 23/30 | 9.7e-05 | 2.2e-04 |
| tanh+relu vs relu | macro-F1 | 30 | +0.69 | [+0.31, +1.08] | 22/30 | 0.002 | 0.002 |
| tanh+relu vs tanh | accuracy | 30 | +1.25 | [+1.05, +1.46] | 30/30 | 2.0e-12 | 1.7e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +1.33 | [+1.08, +1.58] | 30/30 | 4.4e-11 | 1.9e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.42 | [+0.13, +0.67] | 25/30 | 0.006 | 0.001 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.24 | [-0.08, +0.52] | 22/30 | 0.127 | 0.016 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.91 | [+0.70, +1.13] | 28/30 | 5.7e-09 | 5.2e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.88 | [+0.65, +1.12] | 29/30 | 6.5e-08 | 5.7e-07 |
| sigmoid+relu vs relu | accuracy | 30 | -0.70 | [-1.00, -0.41] | 6/30 | 8.4e-05 | 1.2e-04 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.81 | [-1.12, -0.50] | 6/30 | 3.1e-05 | 3.5e-05 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +0.19 | [-0.04, +0.42] | 18/30 | 0.119 | 0.118 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.18 | [-0.08, +0.43] | 18/30 | 0.203 | 0.158 |

![](figures/fig_acc_64_c10.png)
![](figures/fig_f1_64_c10.png)
![](figures/fig_paired_acc_64_c10.png)
![](figures/fig_paired_f1_64_c10.png)
![](figures/fig_curves_64_c10.png)

## 3.1.128 CIFAR-10, 128 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.0003 | 30 | 52.52 ± 0.76 | 52.66 | 52.07 ± 1.02 | 51.89 |
| tanh | 0.0003 | 30 | 51.32 ± 0.91 | 51.56 | 50.96 ± 1.03 | 50.82 |
| leaky_relu | 0.0003 | 30 | 52.40 ± 0.85 | 52.50 | 51.95 ± 1.00 | 51.71 |
| sigmoid | 0.003 | 30 | 51.18 ± 0.70 | 51.25 | 50.78 ± 0.96 | 50.58 |
| tanh+relu | 0.0003 | 30 | 52.75 ± 0.64 | 52.77 | 52.40 ± 0.79 | 52.01 |
| tanh+leaky_relu | 0.0003 | 30 | 52.49 ± 0.80 | 52.38 | 52.11 ± 0.95 | 51.80 |
| sigmoid+relu | 0.0003 | 30 | 51.33 ± 0.75 | 51.31 | 50.86 ± 0.85 | 50.56 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.23 | [+0.04, +0.42] | 19/30 | 0.029 | 0.030 |
| tanh+relu vs relu | macro-F1 | 30 | +0.33 | [+0.12, +0.56] | 22/30 | 0.008 | 0.009 |
| tanh+relu vs tanh | accuracy | 30 | +1.43 | [+1.18, +1.67] | 29/30 | 2.5e-12 | 2.1e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +1.44 | [+1.19, +1.67] | 29/30 | 3.1e-12 | 5.6e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.09 | [-0.13, +0.32] | 16/30 | 0.438 | 0.497 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.15 | [-0.09, +0.41] | 17/30 | 0.270 | 0.428 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +1.18 | [+0.94, +1.42] | 29/30 | 2.0e-10 | 2.4e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +1.15 | [+0.92, +1.38] | 28/30 | 2.4e-10 | 9.3e-09 |
| sigmoid+relu vs relu | accuracy | 30 | -1.19 | [-1.45, -0.92] | 2/30 | 1.4e-09 | 3.9e-06 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.21 | [-1.51, -0.89] | 2/30 | 3.7e-08 | 1.6e-07 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +0.15 | [-0.10, +0.39] | 17/30 | 0.257 | 0.249 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +0.08 | [-0.20, +0.36] | 16/30 | 0.572 | 0.570 |

![](figures/fig_acc_128_c10.png)
![](figures/fig_f1_128_c10.png)
![](figures/fig_paired_acc_128_c10.png)
![](figures/fig_paired_f1_128_c10.png)
![](figures/fig_curves_128_c10.png)

## 3.1.256 CIFAR-10, 256 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.0003 | 30 | 53.69 ± 0.63 | 53.74 | 53.35 ± 0.75 | 53.11 |
| tanh | 0.001 | 30 | 51.91 ± 0.71 | 52.04 | 51.58 ± 0.72 | 51.55 |
| leaky_relu | 0.0003 | 30 | 53.34 ± 0.74 | 53.45 | 52.99 ± 0.82 | 52.80 |
| sigmoid | 0.003 | 30 | 52.13 ± 0.87 | 52.25 | 51.59 ± 1.10 | 51.47 |
| tanh+relu | 0.0003 | 30 | 53.88 ± 0.60 | 54.00 | 53.49 ± 0.69 | 53.34 |
| tanh+leaky_relu | 0.0003 | 30 | 53.51 ± 0.69 | 53.48 | 53.11 ± 0.84 | 52.87 |
| sigmoid+relu | 0.0003 | 30 | 53.16 ± 0.53 | 53.23 | 52.79 ± 0.67 | 52.45 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.19 | [+0.01, +0.38] | 16/30 | 0.056 | 0.111 |
| tanh+relu vs relu | macro-F1 | 30 | +0.14 | [-0.05, +0.34] | 15/30 | 0.185 | 0.404 |
| tanh+relu vs tanh | accuracy | 30 | +1.98 | [+1.69, +2.27] | 30/30 | 5.7e-14 | 1.7e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +1.91 | [+1.62, +2.19] | 30/30 | 1.9e-13 | 1.7e-06 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | +0.17 | [-0.01, +0.35] | 19/30 | 0.074 | 0.066 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | +0.12 | [-0.06, +0.31] | 19/30 | 0.209 | 0.213 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +1.60 | [+1.31, +1.91] | 30/30 | 2.7e-11 | 1.7e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +1.53 | [+1.21, +1.85] | 29/30 | 5.8e-10 | 5.6e-09 |
| sigmoid+relu vs relu | accuracy | 30 | -0.53 | [-0.70, -0.37] | 4/30 | 7.5e-07 | 1.7e-05 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.56 | [-0.75, -0.38] | 3/30 | 2.1e-06 | 4.4e-06 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +1.04 | [+0.77, +1.30] | 27/30 | 2.3e-08 | 5.2e-06 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +1.19 | [+0.89, +1.49] | 26/30 | 1.5e-08 | 1.3e-07 |

![](figures/fig_acc_256_c10.png)
![](figures/fig_f1_256_c10.png)
![](figures/fig_paired_acc_256_c10.png)
![](figures/fig_paired_f1_256_c10.png)
![](figures/fig_curves_256_c10.png)

## 3.1.512 CIFAR-10, 512 hidden neurons, cross-entropy (30 seeds per config)

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.0003 | 30 | 54.33 ± 0.74 | 54.53 | 54.15 ± 0.84 | 53.73 |
| tanh | 0.001 | 30 | 52.19 ± 0.82 | 52.35 | 52.06 ± 0.81 | 51.76 |
| leaky_relu | 0.0003 | 30 | 54.00 ± 0.89 | 54.13 | 53.73 ± 1.03 | 53.35 |
| sigmoid | 0.003 | 30 | 52.89 ± 0.77 | 52.92 | 52.62 ± 0.99 | 52.01 |
| tanh+relu | 0.0003 | 30 | 54.36 ± 0.84 | 54.45 | 54.14 ± 0.93 | 53.73 |
| tanh+leaky_relu | 0.0003 | 30 | 53.90 ± 0.94 | 54.17 | 53.62 ± 1.03 | 53.36 |
| sigmoid+relu | 0.0003 | 30 | 53.94 ± 0.77 | 54.09 | 53.69 ± 0.85 | 53.19 |

Paired comparisons (hybrid minus each homogeneous parent, same seeds):

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | +0.03 | [-0.15, +0.20] | 16/30 | 0.757 | 0.607 |
| tanh+relu vs relu | macro-F1 | 30 | -0.02 | [-0.20, +0.16] | 16/30 | 0.843 | 0.935 |
| tanh+relu vs tanh | accuracy | 30 | +2.17 | [+1.93, +2.41] | 30/30 | 1.1e-16 | 1.7e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +2.08 | [+1.78, +2.36] | 30/30 | 1.5e-14 | 1.9e-09 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.10 | [-0.30, +0.14] | 10/30 | 0.410 | 0.165 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.11 | [-0.33, +0.14] | 7/30 | 0.394 | 0.100 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +1.71 | [+1.44, +1.98] | 30/30 | 5.1e-13 | 1.7e-06 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +1.56 | [+1.25, +1.86] | 29/30 | 1.0e-10 | 5.6e-09 |
| sigmoid+relu vs relu | accuracy | 30 | -0.39 | [-0.55, -0.23] | 4/30 | 3.8e-05 | 1.2e-04 |
| sigmoid+relu vs relu | macro-F1 | 30 | -0.47 | [-0.65, -0.28] | 4/30 | 5.1e-05 | 6.3e-05 |
| sigmoid+relu vs sigmoid | accuracy | 30 | +1.05 | [+0.82, +1.29] | 29/30 | 1.9e-09 | 3.7e-09 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | +1.07 | [+0.82, +1.32] | 30/30 | 4.8e-09 | 1.9e-09 |

![](figures/fig_acc_512_c10.png)
![](figures/fig_f1_512_c10.png)
![](figures/fig_paired_acc_512_c10.png)
![](figures/fig_paired_f1_512_c10.png)
![](figures/fig_curves_512_c10.png)

## 3.2. CIFAR-10 crossover: paired hybrid − parent against width

One row per comparison and width. Δ is the paired mean difference in points (positive = the hybrid is ahead), CI is the 95% bootstrap interval.

**accuracy**

| comparison | width | n | Δ | 95% CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.68 | [-1.00, -0.36] | 9/30 | 2.7e-04 | 0.002 |
| tanh+relu − relu | 32 | 30 | -0.62 | [-0.89, -0.34] | 6/30 | 1.5e-04 | 5.5e-04 |
| tanh+relu − relu | 64 | 30 | +0.77 | [+0.45, +1.10] | 23/30 | 9.7e-05 | 2.2e-04 |
| tanh+relu − relu | 128 | 30 | +0.23 | [+0.04, +0.42] | 19/30 | 0.029 | 0.030 |
| tanh+relu − relu | 256 | 30 | +0.19 | [+0.01, +0.38] | 16/30 | 0.056 | 0.111 |
| tanh+relu − relu | 512 | 30 | +0.03 | [-0.15, +0.20] | 16/30 | 0.757 | 0.607 |
| tanh+relu − tanh | 12 | 30 | +0.69 | [+0.40, +0.99] | 22/30 | 8.9e-05 | 2.4e-04 |
| tanh+relu − tanh | 32 | 30 | +0.79 | [+0.49, +1.08] | 25/30 | 1.7e-05 | 2.4e-04 |
| tanh+relu − tanh | 64 | 30 | +1.25 | [+1.05, +1.47] | 30/30 | 2.0e-12 | 1.7e-06 |
| tanh+relu − tanh | 128 | 30 | +1.43 | [+1.19, +1.66] | 29/30 | 2.5e-12 | 2.1e-06 |
| tanh+relu − tanh | 256 | 30 | +1.98 | [+1.69, +2.26] | 30/30 | 5.7e-14 | 1.7e-06 |
| tanh+relu − tanh | 512 | 30 | +2.17 | [+1.93, +2.42] | 30/30 | 1.1e-16 | 1.7e-06 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.55 | [-0.94, -0.15] | 8/30 | 0.013 | 0.018 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.54 | [-0.81, -0.30] | 10/30 | 3.4e-04 | 8.0e-04 |
| tanh+leaky_relu − leaky_relu | 64 | 30 | +0.42 | [+0.13, +0.67] | 25/30 | 0.006 | 0.001 |
| tanh+leaky_relu − leaky_relu | 128 | 30 | +0.09 | [-0.13, +0.32] | 16/30 | 0.438 | 0.497 |
| tanh+leaky_relu − leaky_relu | 256 | 30 | +0.17 | [-0.01, +0.35] | 19/30 | 0.074 | 0.066 |
| tanh+leaky_relu − leaky_relu | 512 | 30 | -0.10 | [-0.31, +0.13] | 10/30 | 0.410 | 0.165 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.70 | [+0.35, +1.04] | 21/30 | 4.7e-04 | 0.001 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.61 | [+0.28, +0.91] | 24/30 | 7.8e-04 | 0.002 |
| tanh+leaky_relu − tanh | 64 | 30 | +0.91 | [+0.69, +1.13] | 28/30 | 5.7e-09 | 5.2e-06 |
| tanh+leaky_relu − tanh | 128 | 30 | +1.18 | [+0.94, +1.41] | 29/30 | 2.0e-10 | 2.4e-06 |
| tanh+leaky_relu − tanh | 256 | 30 | +1.60 | [+1.31, +1.91] | 30/30 | 2.7e-11 | 1.7e-06 |
| tanh+leaky_relu − tanh | 512 | 30 | +1.71 | [+1.44, +1.98] | 30/30 | 5.1e-13 | 1.7e-06 |

**macro-F1**

| comparison | width | n | Δ | 95% CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.78 | [-1.17, -0.39] | 11/30 | 5.3e-04 | 0.003 |
| tanh+relu − relu | 32 | 30 | -0.62 | [-0.92, -0.32] | 5/30 | 5.1e-04 | 5.5e-04 |
| tanh+relu − relu | 64 | 30 | +0.69 | [+0.30, +1.07] | 22/30 | 0.002 | 0.002 |
| tanh+relu − relu | 128 | 30 | +0.33 | [+0.12, +0.56] | 22/30 | 0.008 | 0.009 |
| tanh+relu − relu | 256 | 30 | +0.14 | [-0.05, +0.35] | 15/30 | 0.185 | 0.404 |
| tanh+relu − relu | 512 | 30 | -0.02 | [-0.20, +0.17] | 16/30 | 0.843 | 0.935 |
| tanh+relu − tanh | 12 | 30 | +0.69 | [+0.39, +0.99] | 22/30 | 1.2e-04 | 1.4e-04 |
| tanh+relu − tanh | 32 | 30 | +0.73 | [+0.43, +1.01] | 24/30 | 5.7e-05 | 2.6e-04 |
| tanh+relu − tanh | 64 | 30 | +1.33 | [+1.08, +1.59] | 30/30 | 4.4e-11 | 1.9e-09 |
| tanh+relu − tanh | 128 | 30 | +1.44 | [+1.19, +1.67] | 29/30 | 3.1e-12 | 5.6e-09 |
| tanh+relu − tanh | 256 | 30 | +1.91 | [+1.62, +2.19] | 30/30 | 1.9e-13 | 1.7e-06 |
| tanh+relu − tanh | 512 | 30 | +2.08 | [+1.79, +2.36] | 30/30 | 1.5e-14 | 1.9e-09 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.54 | [-0.94, -0.13] | 12/30 | 0.018 | 0.020 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.45 | [-0.73, -0.16] | 10/30 | 0.005 | 0.009 |
| tanh+leaky_relu − leaky_relu | 64 | 30 | +0.24 | [-0.08, +0.51] | 22/30 | 0.127 | 0.016 |
| tanh+leaky_relu − leaky_relu | 128 | 30 | +0.15 | [-0.10, +0.42] | 17/30 | 0.270 | 0.428 |
| tanh+leaky_relu − leaky_relu | 256 | 30 | +0.12 | [-0.06, +0.30] | 19/30 | 0.209 | 0.213 |
| tanh+leaky_relu − leaky_relu | 512 | 30 | -0.11 | [-0.33, +0.14] | 7/30 | 0.394 | 0.100 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.75 | [+0.35, +1.12] | 25/30 | 8.3e-04 | 0.001 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.63 | [+0.29, +0.94] | 25/30 | 9.7e-04 | 0.001 |
| tanh+leaky_relu − tanh | 64 | 30 | +0.88 | [+0.65, +1.12] | 29/30 | 6.5e-08 | 5.7e-07 |
| tanh+leaky_relu − tanh | 128 | 30 | +1.15 | [+0.91, +1.38] | 28/30 | 2.4e-10 | 9.3e-09 |
| tanh+leaky_relu − tanh | 256 | 30 | +1.53 | [+1.20, +1.85] | 29/30 | 5.8e-10 | 5.6e-09 |
| tanh+leaky_relu − tanh | 512 | 30 | +1.56 | [+1.25, +1.86] | 29/30 | 1.0e-10 | 5.6e-09 |

![](figures/fig_crossover_c10.png)
![](figures/fig_crossover_f1_c10.png)
![](figures/fig_width_acc_c10.png)

## 3.3 Does the crossover replicate across datasets?

Paired Δ in test accuracy points at each width, MNIST beside CIFAR-10. A crossover that is a property of hybrid activation rather than of MNIST should change sign at a similar width in both columns.

**tanh+relu − relu**

| width | MNIST Δ | MNIST wins | CIFAR-10 Δ | CIFAR-10 wins | same sign |
|---|---|---|---|---|---|
| 12 | -0.27 [-0.45, -0.09] | 9/30 | -0.68 [-1.00, -0.36] | 9/30 | yes |
| 32 | +0.01 [-0.09, +0.13] | 16/30 | -0.62 [-0.89, -0.34] | 6/30 | **no** |
| 64 | -0.00 [-0.06, +0.05] | 11/30 | +0.77 [+0.45, +1.10] | 23/30 | **no** |
| 128 | +0.17 [+0.14, +0.21] | 29/30 | +0.23 [+0.04, +0.42] | 19/30 | yes |
| 256 | -0.00 [-0.03, +0.03] | 13/30 | +0.19 [+0.01, +0.38] | 16/30 | **no** |
| 512 | +0.02 [-0.01, +0.05] | 17/30 | +0.03 [-0.15, +0.20] | 16/30 | yes |

**tanh+relu − tanh**

| width | MNIST Δ | MNIST wins | CIFAR-10 Δ | CIFAR-10 wins | same sign |
|---|---|---|---|---|---|
| 12 | +0.02 [-0.11, +0.14] | 16/30 | +0.69 [+0.40, +0.99] | 22/30 | yes |
| 32 | +0.16 [+0.07, +0.25] | 23/30 | +0.79 [+0.49, +1.08] | 25/30 | yes |
| 64 | +0.21 [+0.15, +0.26] | 28/30 | +1.25 [+1.05, +1.47] | 30/30 | yes |
| 128 | +0.22 [+0.18, +0.26] | 28/30 | +1.43 [+1.19, +1.66] | 29/30 | yes |
| 256 | +0.18 [+0.14, +0.22] | 30/30 | +1.98 [+1.69, +2.26] | 30/30 | yes |
| 512 | +0.27 [+0.23, +0.31] | 30/30 | +2.17 [+1.93, +2.42] | 30/30 | yes |

**tanh+leaky_relu − leaky_relu**

| width | MNIST Δ | MNIST wins | CIFAR-10 Δ | CIFAR-10 wins | same sign |
|---|---|---|---|---|---|
| 12 | -0.23 [-0.39, -0.05] | 7/30 | -0.55 [-0.94, -0.15] | 8/30 | yes |
| 32 | -0.03 [-0.11, +0.04] | 14/30 | -0.54 [-0.81, -0.30] | 10/30 | yes |
| 64 | +0.07 [+0.02, +0.11] | 23/30 | +0.42 [+0.13, +0.67] | 25/30 | yes |
| 128 | +0.15 [+0.12, +0.18] | 28/30 | +0.09 [-0.13, +0.32] | 16/30 | yes |
| 256 | +0.00 [-0.03, +0.03] | 15/30 | +0.17 [-0.01, +0.35] | 19/30 | yes |
| 512 | +0.01 [-0.02, +0.05] | 16/30 | -0.10 [-0.31, +0.13] | 10/30 | **no** |

**tanh+leaky_relu − tanh**

| width | MNIST Δ | MNIST wins | CIFAR-10 Δ | CIFAR-10 wins | same sign |
|---|---|---|---|---|---|
| 12 | +0.11 [-0.03, +0.24] | 19/30 | +0.70 [+0.35, +1.04] | 21/30 | yes |
| 32 | +0.15 [+0.09, +0.21] | 24/30 | +0.61 [+0.28, +0.91] | 24/30 | yes |
| 64 | +0.24 [+0.18, +0.29] | 28/30 | +0.91 [+0.69, +1.13] | 28/30 | yes |
| 128 | +0.17 [+0.13, +0.22] | 27/30 | +1.18 [+0.94, +1.41] | 29/30 | yes |
| 256 | +0.16 [+0.12, +0.20] | 28/30 | +1.60 [+1.31, +1.91] | 30/30 | yes |
| 512 | +0.24 [+0.20, +0.28] | 30/30 | +1.71 [+1.44, +1.98] | 30/30 | yes |

## 4. Learning-rate robustness (seed-replicated, evaluation seeds)

**64 hidden, CE — mean validation accuracy (30 seeds)**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.05 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| relu | 95.07 | 97.02 | 97.53 | 97.78 | 97.94 | 96.95 | 95.50 | 58.43 | 19.77 | 11.06 | 10.07 |
| tanh | 94.76 | 96.99 | 97.48 | 97.62 | 97.71 | 97.66 | 96.73 | 95.91 | 91.89 | 69.40 | 11.26 |
| leaky_relu | 94.78 | 96.85 | 97.40 | 97.78 | 97.90 | 97.07 | 95.55 | 84.00 | 83.71 | 85.43 | 84.31 |
| sigmoid | 83.24 | 91.73 | 96.08 | 97.29 | 97.40 | 97.46 | 97.58 | 97.56 | 97.51 | 96.94 | 94.80 |
| tanh+relu | 94.96 | 97.00 | 97.51 | 97.71 | 97.81 | 97.90 | 96.80 | 96.39 | 91.05 | 61.28 | 12.24 |
| tanh+leaky_relu | 94.80 | 96.95 | 97.47 | 97.74 | 97.82 | 97.86 | 96.50 | 95.69 | 82.74 | 84.98 | 84.27 |
| sigmoid+relu | 94.07 | 96.32 | 96.65 | 96.72 | 96.63 | 96.73 | 96.70 | 96.45 | 96.15 | 81.76 | 76.57 |

**64 hidden, CE — collapsed seeds (final val acc < 50%), count / n**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.05 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 12/30 | 30/30 | 30/30 | 30/30 |
| tanh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 30/30 |
| leaky_relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| sigmoid | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| tanh+relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 10/30 | 30/30 |
| tanh+leaky_relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| sigmoid+relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 6/30 |

![](figures/fig_lrsweep_64.png)

**128 hidden, CE — mean validation accuracy (30 seeds)**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.05 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| relu | 95.41 | 97.44 | 97.87 | 98.12 | 98.26 | 98.25 | 96.50 | 10.61 | 20.24 | 10.84 | 10.16 |
| tanh | 94.51 | 97.02 | 97.79 | 97.94 | 98.01 | 98.14 | 97.47 | 95.11 | 86.42 | 54.08 | 10.47 |
| leaky_relu | 95.10 | 97.30 | 97.81 | 98.12 | 98.21 | 97.71 | 84.71 | 85.23 | 86.40 | 86.30 | 86.21 |
| sigmoid | 85.03 | 91.53 | 95.97 | 97.56 | 97.77 | 97.93 | 98.02 | 98.03 | 97.99 | 97.94 | 95.90 |
| tanh+relu | 95.16 | 97.28 | 97.84 | 98.07 | 98.18 | 98.28 | 97.57 | 96.76 | 69.27 | 31.60 | 11.43 |
| tanh+leaky_relu | 94.97 | 97.20 | 97.82 | 98.06 | 98.20 | 98.28 | 96.78 | 84.63 | 84.64 | 85.21 | 84.99 |
| sigmoid+relu | 94.68 | 96.96 | 97.44 | 97.75 | 97.95 | 97.31 | 97.84 | 97.78 | 96.97 | 69.48 | 71.57 |

**128 hidden, CE — collapsed seeds (final val acc < 50%), count / n**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.03 | lr=0.05 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| tanh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 9/30 | 30/30 |
| leaky_relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| sigmoid | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| tanh+relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 25/30 | 30/30 |
| tanh+leaky_relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| sigmoid+relu | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 | 8/30 |

![](figures/fig_lrsweep_128.png)

**Robustness score** — widest contiguous LR interval on the tested grid over which a config's mean validation accuracy stays within 1 point of its own best (shown as lo–hi, number of grid points, decades spanned):

| config | measure | W=64 | W=128 |
|---|---|---|---|
| relu | within 1 pt of own best | 0.0003–0.01 (5 pts, 1.52 dec) | 0.0003–0.01 (5 pts, 1.52 dec) |
| relu | no seed collapsed | 0.0001–0.02 (7 pts, 2.30 dec) | 0.0001–0.02 (7 pts, 2.30 dec) |
| tanh | within 1 pt of own best | 0.0003–0.02 (6 pts, 1.82 dec) | 0.001–0.02 (5 pts, 1.30 dec) |
| tanh | no seed collapsed | 0.0001–0.1 (10 pts, 3.00 dec) | 0.0001–0.05 (9 pts, 2.70 dec) |
| leaky_relu | within 1 pt of own best | 0.001–0.01 (4 pts, 1.00 dec) | 0.0003–0.01 (5 pts, 1.52 dec) |
| leaky_relu | no seed collapsed | 0.0001–0.3 (11 pts, 3.48 dec) | 0.0001–0.3 (11 pts, 3.48 dec) |
| sigmoid | within 1 pt of own best | 0.003–0.1 (7 pts, 1.52 dec) | 0.003–0.1 (7 pts, 1.52 dec) |
| sigmoid | no seed collapsed | 0.0001–0.3 (11 pts, 3.48 dec) | 0.0001–0.3 (11 pts, 3.48 dec) |
| tanh+relu | within 1 pt of own best | 0.0003–0.01 (5 pts, 1.52 dec) | 0.0003–0.02 (6 pts, 1.82 dec) |
| tanh+relu | no seed collapsed | 0.0001–0.05 (9 pts, 2.70 dec) | 0.0001–0.05 (9 pts, 2.70 dec) |
| tanh+leaky_relu | within 1 pt of own best | 0.0003–0.01 (5 pts, 1.52 dec) | 0.001–0.01 (4 pts, 1.00 dec) |
| tanh+leaky_relu | no seed collapsed | 0.0001–0.3 (11 pts, 3.48 dec) | 0.0001–0.3 (11 pts, 3.48 dec) |
| sigmoid+relu | within 1 pt of own best | 0.0003–0.05 (8 pts, 2.22 dec) | 0.0003–0.05 (8 pts, 2.22 dec) |
| sigmoid+relu | no seed collapsed | 0.0001–0.1 (10 pts, 3.00 dec) | 0.0001–0.05 (9 pts, 2.70 dec) |

## 5. Calibration LR grids (validation accuracy, mean over calibration seeds)

**12 hidden, CE — `calib_12.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 92.70 | 94.29 | 94.64 | 94.26 | 93.28 | 50.80 | 11.34 | 9.63 |
| tanh | 92.61 | 94.09 | 94.14 | 93.68 | 92.72 | 91.93 | 84.53 | 49.83 |
| leaky_relu | 92.68 | 94.17 | 94.41 | 94.00 | 94.02 | 93.45 | 82.51 | 84.27 |
| sigmoid | 73.32 | 89.48 | 92.96 | 93.77 | 93.54 | 92.90 | 91.72 | 88.49 |
| tanh+relu | 92.60 | 94.05 | 94.29 | 93.70 | 92.90 | 89.33 | 82.55 | 41.60 |
| tanh+leaky_relu | 92.54 | 93.96 | 94.08 | 94.07 | 93.45 | 92.07 | 75.47 | 78.31 |
| sigmoid+relu | 89.75 | 91.08 | 93.35 | 93.60 | 93.18 | 92.77 | 89.68 | 83.51 |

![](figures/fig_calib_12.png)

**32 hidden, CE — `calib_32.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 94.52 | 96.33 | 96.65 | 96.70 | 96.28 | 89.63 | 11.68 | 10.13 |
| tanh | 94.66 | 96.47 | 96.59 | 96.49 | 95.80 | 94.85 | 81.08 | 12.71 |
| leaky_relu | 94.28 | 96.30 | 96.62 | 96.67 | 96.15 | 94.38 | 84.14 | 85.38 |
| sigmoid | 80.86 | 91.51 | 95.60 | 96.56 | 96.27 | 96.23 | 95.60 | 93.29 |
| tanh+relu | 94.51 | 96.42 | 96.75 | 96.76 | 96.20 | 95.07 | 85.04 | 17.66 |
| tanh+leaky_relu | 94.46 | 96.42 | 96.72 | 96.81 | 96.11 | 95.35 | 82.76 | 83.21 |
| sigmoid+relu | 93.11 | 94.91 | 95.20 | 95.76 | 96.35 | 95.73 | 86.11 | 77.58 |

![](figures/fig_calib_32.png)

**64 hidden, CE — `calib_64.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 94.99 | 97.00 | 97.54 | 97.83 | 97.10 | 51.55 | 10.48 | 9.82 |
| tanh | 94.80 | 96.91 | 97.44 | 97.63 | 97.65 | 96.28 | 69.88 | 9.81 |
| leaky_relu | 94.77 | 96.92 | 97.48 | 97.83 | 96.90 | 84.68 | 85.24 | 86.61 |
| sigmoid | 83.96 | 91.83 | 96.07 | 97.30 | 97.59 | 97.56 | 96.68 | 94.69 |
| tanh+relu | 94.92 | 96.91 | 97.49 | 97.68 | 97.84 | 96.33 | 50.51 | 9.82 |
| tanh+leaky_relu | 94.85 | 96.84 | 97.53 | 97.73 | 97.82 | 95.86 | 83.20 | 83.77 |
| sigmoid+relu | 94.25 | 96.20 | 96.73 | 96.86 | 96.49 | 96.11 | 72.43 | 79.21 |

![](figures/fig_calib_64.png)

**128 hidden, CE — `calib_128.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 95.24 | 97.29 | 97.91 | 98.09 | 98.03 | 10.40 | 10.54 | 10.24 |
| tanh | 94.55 | 97.10 | 97.79 | 97.96 | 98.14 | 94.75 | 59.21 | 10.67 |
| leaky_relu | 94.95 | 97.23 | 97.83 | 97.99 | 97.55 | 84.98 | 85.65 | 85.80 |
| sigmoid | 84.36 | 91.51 | 95.94 | 97.53 | 97.92 | 98.06 | 97.99 | 95.62 |
| tanh+relu | 95.10 | 97.22 | 97.80 | 98.07 | 98.31 | 96.79 | 35.40 | 10.00 |
| tanh+leaky_relu | 94.90 | 97.16 | 97.75 | 97.97 | 98.29 | 83.93 | 84.31 | 84.67 |
| sigmoid+relu | 94.70 | 96.96 | 97.43 | 97.79 | 97.07 | 97.78 | 65.44 | 63.86 |

![](figures/fig_calib_128.png)

**256 hidden, CE — `calib_256.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 95.65 | 97.64 | 98.09 | 98.28 | 98.44 | 10.68 | 10.51 | 10.00 |
| tanh | 94.05 | 96.75 | 97.84 | 98.00 | 98.28 | 91.43 | 46.02 | 9.91 |
| leaky_relu | 95.34 | 97.60 | 98.08 | 98.15 | 98.40 | 87.14 | 86.39 | 86.94 |
| sigmoid | 86.04 | 91.21 | 95.46 | 97.54 | 98.10 | 98.27 | 98.32 | 96.38 |
| tanh+relu | 95.23 | 97.47 | 98.02 | 98.22 | 98.46 | 94.83 | 34.30 | 11.02 |
| tanh+leaky_relu | 94.99 | 97.29 | 97.94 | 98.21 | 98.44 | 85.30 | 86.39 | 85.80 |
| sigmoid+relu | 95.09 | 97.43 | 97.87 | 98.14 | 98.26 | 98.20 | 38.77 | 57.68 |

![](figures/fig_calib_256.png)

**512 hidden, CE — `calib_512.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 95.82 | 97.78 | 98.20 | 98.31 | 98.49 | 10.32 | 11.56 | 10.20 |
| tanh | 93.38 | 96.51 | 97.72 | 98.01 | 98.27 | 81.79 | 35.90 | 9.99 |
| leaky_relu | 95.47 | 97.62 | 98.13 | 98.28 | 98.55 | 85.71 | 88.08 | 87.10 |
| sigmoid | 86.90 | 91.03 | 94.94 | 97.20 | 98.00 | 98.28 | 98.36 | 80.97 |
| tanh+relu | 95.29 | 97.58 | 98.11 | 98.30 | 98.59 | 72.84 | 36.83 | 14.55 |
| tanh+leaky_relu | 94.99 | 97.45 | 98.06 | 98.29 | 98.51 | 87.06 | 87.58 | 88.05 |
| sigmoid+relu | 95.33 | 97.60 | 97.97 | 98.22 | 98.37 | 98.37 | 37.72 | 49.58 |

![](figures/fig_calib_512.png)

**CIFAR-10, 12 hidden, CE — `c10_calib_12.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 41.14 | 42.63 | 41.61 | 38.14 | 21.44 | 10.19 | 10.00 | 10.05 |
| tanh | 41.33 | 41.85 | 41.08 | 38.11 | 34.07 | 25.80 | 11.80 | 10.07 |
| leaky_relu | 40.63 | 42.58 | 42.47 | 41.95 | 39.71 | 32.86 | 10.13 | 10.59 |
| sigmoid | 25.95 | 35.00 | 39.95 | 40.15 | 38.43 | 35.50 | 28.04 | 15.23 |
| tanh+relu | 40.90 | 42.01 | 40.76 | 39.04 | 30.11 | 25.84 | 13.33 | 9.98 |
| tanh+leaky_relu | 41.11 | 42.19 | 41.40 | 39.76 | 36.47 | 28.23 | 9.56 | 10.08 |
| sigmoid+relu | 36.13 | 39.16 | 40.73 | 39.66 | 35.10 | 30.23 | 24.10 | 14.02 |

![](figures/fig_calib_12_c10.png)

**CIFAR-10, 32 hidden, CE — `c10_calib_32.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 46.32 | 47.94 | 47.66 | 44.09 | 33.04 | 11.37 | 10.11 | 10.11 |
| tanh | 45.93 | 46.74 | 46.49 | 42.74 | 35.74 | 22.82 | 11.04 | 9.84 |
| leaky_relu | 45.83 | 48.03 | 47.66 | 45.76 | 42.52 | 35.48 | 13.57 | 13.29 |
| sigmoid | 28.77 | 38.28 | 45.18 | 46.13 | 42.80 | 37.36 | 28.58 | 16.32 |
| tanh+relu | 46.14 | 47.66 | 46.83 | 44.08 | 30.98 | 25.35 | 14.69 | 9.84 |
| tanh+leaky_relu | 46.16 | 47.59 | 47.52 | 44.38 | 40.58 | 21.98 | 11.63 | 11.56 |
| sigmoid+relu | 43.33 | 45.37 | 46.73 | 45.52 | 41.23 | 35.13 | 25.10 | 15.15 |

![](figures/fig_calib_32_c10.png)

**CIFAR-10, 64 hidden, CE — `c10_calib_64.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 48.29 | 50.27 | 50.52 | 47.20 | 36.60 | 12.21 | 10.00 | 10.21 |
| tanh | 47.86 | 49.53 | 48.64 | 44.05 | 35.97 | 21.35 | 10.22 | 10.14 |
| leaky_relu | 47.79 | 50.24 | 50.57 | 48.39 | 43.90 | 36.49 | 15.82 | 18.18 |
| sigmoid | 30.60 | 39.51 | 47.75 | 48.63 | 45.01 | 39.03 | 27.21 | 15.67 |
| tanh+relu | 48.73 | 50.37 | 49.52 | 47.21 | 32.03 | 21.89 | 12.87 | 10.14 |
| tanh+leaky_relu | 48.36 | 50.55 | 48.96 | 46.83 | 41.30 | 18.39 | 13.87 | 15.13 |
| sigmoid+relu | 46.25 | 48.53 | 49.20 | 48.24 | 44.27 | 34.67 | 27.08 | 14.93 |

![](figures/fig_calib_64_c10.png)

**CIFAR-10, 128 hidden, CE — `c10_calib_128.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 50.00 | 52.20 | 51.18 | 48.92 | 37.40 | 10.98 | 9.67 | 9.85 |
| tanh | 48.45 | 51.02 | 50.65 | 45.64 | 34.43 | 13.82 | 9.81 | 9.81 |
| leaky_relu | 49.41 | 52.25 | 51.88 | 49.28 | 45.03 | 35.70 | 20.07 | 20.15 |
| sigmoid | 31.26 | 39.75 | 49.02 | 50.41 | 45.91 | 39.78 | 29.25 | 10.49 |
| tanh+relu | 49.85 | 52.35 | 51.64 | 49.06 | 26.33 | 20.11 | 9.82 | 9.81 |
| tanh+leaky_relu | 49.23 | 52.04 | 51.35 | 49.03 | 41.25 | 15.17 | 14.70 | 16.18 |
| sigmoid+relu | 47.95 | 50.58 | 50.45 | 49.42 | 44.64 | 18.30 | 27.07 | 15.71 |

![](figures/fig_calib_128_c10.png)

**CIFAR-10, 256 hidden, CE — `c10_calib_256.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 50.68 | 52.72 | 50.99 | 49.35 | 37.85 | 10.13 | 9.83 | 9.75 |
| tanh | 48.64 | 50.89 | 51.11 | 45.61 | 31.09 | 10.23 | 9.82 | 9.82 |
| leaky_relu | 50.16 | 52.67 | 52.50 | 50.10 | 44.10 | 34.58 | 17.65 | 18.83 |
| sigmoid | 31.56 | 39.99 | 48.72 | 50.81 | 47.28 | 36.89 | 23.80 | 9.82 |
| tanh+relu | 50.75 | 52.73 | 52.59 | 49.98 | 22.73 | 10.53 | 9.82 | 9.82 |
| tanh+leaky_relu | 50.19 | 52.71 | 52.64 | 49.93 | 39.18 | 17.93 | 18.60 | 16.14 |
| sigmoid+relu | 49.63 | 52.05 | 51.28 | 50.74 | 47.93 | 13.27 | 25.80 | 12.76 |

![](figures/fig_calib_256_c10.png)

**CIFAR-10, 512 hidden, CE — `c10_calib_512.csv`**

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 51.49 | 54.10 | 52.75 | 50.12 | 40.20 | 10.18 | 10.10 | 10.16 |
| tanh | 48.28 | 51.55 | 51.57 | 45.04 | 14.20 | 9.99 | 9.99 | 9.99 |
| leaky_relu | 50.85 | 53.59 | 52.67 | 50.42 | 42.99 | 34.60 | 19.20 | 20.56 |
| sigmoid | 31.84 | 39.91 | 48.36 | 51.63 | 49.31 | 12.94 | 9.99 | 9.99 |
| tanh+relu | 51.11 | 53.88 | 52.71 | 50.50 | 20.36 | 9.99 | 9.99 | 9.99 |
| tanh+leaky_relu | 50.71 | 53.47 | 52.95 | 50.93 | 37.71 | 19.77 | 17.86 | 20.21 |
| sigmoid+relu | 50.47 | 53.74 | 52.66 | 51.73 | 49.76 | 15.95 | 16.29 | 12.64 |

![](figures/fig_calib_512_c10.png)

**Selected learning rate per config and width (cross-entropy), from `results/best_lr.json`**

| config | W=12 | W=32 | W=64 | W=128 | W=256 | W=512 |
|---|---|---|---|---|---|---|
| relu | 0.001 | 0.003 | 0.003 | 0.003 | 0.01 | 0.01 |
| tanh | 0.001 | 0.001 | 0.01 | 0.01 | 0.01 | 0.01 |
| leaky_relu | 0.001 | 0.003 | 0.003 | 0.003 | 0.01 | 0.01 |
| sigmoid | 0.003 | 0.003 | 0.01 | 0.03 | 0.1 | 0.1 |
| tanh+relu | 0.001 | 0.003 | 0.01 | 0.01 | 0.01 | 0.01 |
| tanh+leaky_relu | 0.001 | 0.003 | 0.01 | 0.01 | 0.01 | 0.01 |
| sigmoid+relu | 0.003 | 0.01 | 0.003 | 0.003 | 0.01 | 0.03 |

## 6. Layout and ratio

**128 hidden, CE — `layout_128.csv`.** Each pair uses the LR calibrated for its interleave/0.5 variant, so ratios 0.25 and 0.75 are not separately tuned.

| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd | Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tanh+leaky_relu | block | 0.5 | 0.01 | 30 | 98.03 ± 0.08 | 98.02 ± 0.09 | +0.18 [+0.14, +0.22] | +0.20 [+0.15, +0.25] | +0.03 [-0.00, +0.07] |
| tanh+leaky_relu | interleave | 0.25 | 0.01 | 30 | 97.97 ± 0.08 | 97.95 ± 0.09 | +0.12 [+0.07, +0.16] | +0.14 [+0.10, +0.18] | -0.03 [-0.07, +0.00] |
| tanh+leaky_relu | interleave | 0.5 | 0.01 | 30 | 98.00 ± 0.05 | 97.98 ± 0.05 | +0.15 [+0.12, +0.18] | +0.17 [+0.12, +0.21] | +0.00 [+0.00, +0.00] |
| tanh+leaky_relu | interleave | 0.75 | 0.01 | 30 | 98.07 ± 0.08 | 98.05 ± 0.08 | +0.22 [+0.18, +0.26] | +0.24 [+0.19, +0.28] | +0.07 [+0.03, +0.10] |
| tanh+leaky_relu | random | 0.5 | 0.01 | 30 | 98.03 ± 0.08 | 98.01 ± 0.08 | +0.18 [+0.14, +0.22] | +0.20 [+0.15, +0.25] | +0.03 [-0.01, +0.07] |
| tanh+relu | block | 0.5 | 0.01 | 30 | 98.04 ± 0.10 | 98.02 ± 0.10 | +0.16 [+0.12, +0.20] | +0.21 [+0.16, +0.26] | -0.01 [-0.05, +0.03] |
| tanh+relu | interleave | 0.25 | 0.01 | 30 | 97.98 ± 0.08 | 97.96 ± 0.08 | +0.10 [+0.06, +0.14] | +0.15 [+0.10, +0.19] | -0.08 [-0.12, -0.04] |
| tanh+relu | interleave | 0.5 | 0.01 | 30 | 98.05 ± 0.07 | 98.03 ± 0.07 | +0.17 [+0.14, +0.21] | +0.22 [+0.18, +0.26] | +0.00 [+0.00, +0.00] |
| tanh+relu | interleave | 0.75 | 0.01 | 30 | 98.07 ± 0.07 | 98.05 ± 0.08 | +0.19 [+0.15, +0.22] | +0.24 [+0.19, +0.28] | +0.01 [-0.02, +0.05] |
| tanh+relu | random | 0.5 | 0.01 | 30 | 98.06 ± 0.08 | 98.05 ± 0.08 | +0.18 [+0.15, +0.22] | +0.23 [+0.18, +0.28] | +0.01 [-0.03, +0.05] |

## 7. Replication under the original loss (MSE, uniform sigmoid output)

**128 hidden, MSE — `mse_128.csv` (30 seeds)**

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.01 | 30 | 97.99 ± 0.09 | 97.99 | 97.98 ± 0.09 | 98.15 |
| tanh | 0.01 | 30 | 97.80 ± 0.08 | 97.79 | 97.78 ± 0.09 | 98.02 |
| tanh+relu | 0.03 | 30 | 97.98 ± 0.07 | 97.97 | 97.96 ± 0.07 | 98.22 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.02 | [-0.06, +0.02] | 12/30 | 0.412 | 0.446 |
| tanh+relu vs relu | macro-F1 | 30 | -0.02 | [-0.06, +0.02] | 12/30 | 0.343 | 0.349 |
| tanh+relu vs tanh | accuracy | 30 | +0.18 | [+0.14, +0.21] | 29/30 | 1.9e-10 | 2.5e-06 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.18 | [+0.14, +0.21] | 29/30 | 1.8e-10 | 3.7e-09 |

![](figures/fig_acc_128_mse.png)

MSE calibration grid, 128 hidden (validation accuracy, mean over calibration seeds):

| config | lr=0.0001 | lr=0.0003 | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|---|---|
| relu | 92.29 | 95.58 | 97.62 | 97.97 | 98.12 | 97.90 | 10.63 | 9.92 |
| tanh | 91.66 | 94.53 | 97.22 | 97.87 | 98.02 | 97.97 | 92.69 | 12.83 |
| tanh+relu | 91.85 | 95.18 | 97.44 | 97.92 | 98.20 | 98.26 | 36.32 | 10.96 |

![](figures/fig_calib_128_mse.png)

## 8. Rejected rows

No row was rejected: every run in every CSV has `status=ok` and a complete confusion matrix.

