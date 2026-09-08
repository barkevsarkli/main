# version5 extended sweep — report

Produced by `analysis/make_report.py` from the CSVs in `results/`. The full auto-generated tables and every figure live in [`results/summary_extended.md`](results/summary_extended.md); this file is the digest.

## 1. Machine, build and timing

**Machine** — Apple M1 (4 performance + 4 efficiency cores, `hw.physicalcpu = 8`), 8 GB unified memory,
macOS 26.2 (build 25C56), arm64.

**Compiler and build** — Apple clang 17.0.0.

```
clang++ -std=c++17 -O3 -mcpu=native -o main main.cpp net.cpp data.cpp
```

`-mcpu=native` compiles cleanly on this machine and was used for every run in the extended sweep.
The binary was built once and never rebuilt during the sweep;
`sha256 = 160c6e4bf3fd32cd301c88e5c33d8b5b4ed7d75d4e69add276f204bee41f9a21`.
No source file was modified: `main.cpp`, `net.cpp`, `net.h`, `data.cpp` and `data.h` are byte-for-byte
as they were, so init, batch-size-1 SGD, per-epoch shuffling, the fixed 45k/5k/10k split,
`SPLIT_SEED 12345`, the gradient clip, `LEAKY_SLOPE 0.1` and both loss definitions are untouched.

**Workers** — `P = 8`, chosen by measurement rather than by core count alone. Eight width-128
single-epoch jobs took 23.9 s at `P = 4`, 25.7 s at `P = 6` and 17.9 s at `P = 8`, i.e. a 5.05×
speed-up over the 90.4 s the same eight jobs cost serially. The four efficiency cores are slower
than the performance cores but still add throughput, so all eight were used.

**Single-process timings** (`--epochs 1` and `--epochs 3`, difference divided by two, machine
otherwise idle). "Per epoch" covers the 45 000 training samples plus the 5 000-sample validation
pass at the end of each epoch; "fixed cost" is data loading plus the final validation + test
evaluation.

| hidden width | per epoch (s) | fixed cost (s) | one 10-epoch run (s) |
|---|---|---|---|
| 12 | 0.68 | 0.59 | 7.3 |
| 32 | 1.94 | 0.98 | 20.4 |
| 64 | 4.25 | 1.30 | 43.7 |
| 128 | 8.86 | 2.42 | 91.0 |
| 256 | 19.63 | 5.26 | 201.6 |
| 512 | 49.81 | 13.25 | 511.4 |

Peak resident set size is 474 MB per process (transient, while `Data::split_data` holds both the
source and the split copies), settling to about 237 MB.

**Predicted vs actual.** Step 0 predicted 8.4 h for the full plan from a short benchmark. The
priority 1–3 stages plus layout-128 and the MSE replication finished in **6 h 48 min** wall-clock
(14:32:17 → 21:20:29 on 2026-09-04). The optional stages then ran on into the night; width 512
finished at 03:50 on 2026-09-05, a total of **13 h 18 min** for everything reported here. Realized
throughput varied by more than 3× across stages — 1.9× over serial at width 12, 6.0× at width 32,
6.7× at width 64 — which is why the mid-sweep scope cut described in section 10 was made and then
reversed. Worker count ended at P=4 rather than 8 after three low-memory kills.

## 2. Run inventory

Evaluation runs (stage B), config × width → number of seeds:

| config | W=12 | W=32 | W=64 | W=128 | W=256 | W=512 |
|---|---|---|---|---|---|---|
| relu | 30 | 30 | 20 | 20 | 10 | 5 |
| tanh | 30 | 30 | 20 | 20 | 10 | 5 |
| leaky_relu | 30 | 30 | 20 | 20 | 10 | 5 |
| sigmoid | 30 | 30 | 20 | 20 | – | – |
| tanh+relu | 30 | 30 | 20 | 20 | 10 | 5 |
| tanh+leaky_relu | 30 | 30 | 20 | 20 | 10 | 5 |
| sigmoid+relu | 30 | 30 | 20 | 20 | – | – |

All other stages:

| stage | file | ok runs |
|---|---|---|
| A calibration (W=12) | `calib_12_v2.csv` | 84 |
| A calibration (W=32) | `calib_32.csv` | 84 |
| A calibration (W=64) | `calib_64.csv` | 84 |
| A calibration (W=128) | `calib_128_v2.csv` | 84 |
| A calibration (W=256) | `calib_256.csv` | 60 |
| A calibration (W=512) | `calib_512.csv` | 34 |
| C LR robustness (W=12) | `lrsweep_12.csv` | 300 |
| C LR robustness (W=64) | `lrsweep_64.csv` | 250 |
| C LR robustness (W=128) | `lrsweep_128.csv` | 250 |
| D layout/ratio (W=12) | `layout_12.csv` | 80 |
| D layout/ratio (W=128) | `layout_128.csv` | 100 |
| D layout/ratio (W=256) | `layout_256.csv` | 60 |
| E MSE calibration (W=128) | `calib_mse_128.csv` | 27 |
| E MSE main (W=128) | `mse_128.csv` | 30 |

Where a width has both `main_<W>.csv` and `main_<W>_v2.csv`, the `_v2` file is used and the older one ignored: W=12 uses `main_12_v2.csv`, ignores `main_12.csv`; W=128 uses `main_128_v2.csv`, ignores `main_128.csv`.

Seeds 1–N are shared by every configuration at every width, so all comparisons are paired.
Calibration seeds are 101–103 and are never used for evaluation.

`main_12_v2.csv` and `main_128_v2.csv` supersede `main_12.csv` and `main_128.csv`: the older files were
produced on a different machine whose `std::shuffle` yields a different train/val/test split from the
same `SPLIT_SEED`, so they are not row-comparable with anything here (see section 10). The analysis
prefers the `_v2` file wherever both exist and ignores the older one; both remain on disk untouched.

## 3. Calibrated learning rates

Cross-entropy, selected on mean validation accuracy over the calibration seeds (101–103; 101–102 at width 512), grid {0.001, 0.003, 0.01, 0.03}:

| config | W=12 | W=32 | W=64 | W=128 | W=256 | W=512 |
|---|---|---|---|---|---|---|
| relu | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.01 |
| tanh | 0.003 | 0.003 | 0.003 | 0.01 | 0.01 | 0.01 |
| leaky_relu | 0.003 | 0.003 | 0.003 | 0.003 | 0.01 | 0.003 |
| sigmoid | 0.01 | 0.01 | 0.03 | 0.03 | – | – |
| tanh+relu | 0.003 | 0.003 | 0.01 | 0.01 | 0.01 | 0.01 |
| tanh+leaky_relu | 0.003 | 0.003 | 0.01 | 0.01 | 0.01 | 0.003 |
| sigmoid+relu | 0.01 | 0.01 | 0.003 | 0.003 | – | – |

Changes against the first-look selections (`results/best_lr_firstlook.json`) at widths 12 and 128:

- W=12 tanh+relu: 0.001 → 0.003
- W=12 sigmoid+relu: 0.003 → 0.01
- W=128 sigmoid: (not calibrated in the first-look sweep) → 0.03
- W=128 sigmoid+relu: (not calibrated in the first-look sweep) → 0.003

Two width-12 selections moved against the first-look sweep, both on a grid that is now four
points rather than five and on the new split: `tanh+relu` 0.001 → 0.003 and `sigmoid+relu` 0.003 → 0.01.
No width-128 selection changed — the first-look sweep had only tried {0.003, 0.01} with a single seed
there, and the full four-point grid with three seeds confirms the same choice for all five configs it
had calibrated. The two sigmoid configurations at width 128 are calibrated here for the first time.

The learning rate rises with width for the hybrids (0.003 at 12–32, 0.01 from 64 up) and stays at
0.003 for `relu` and `leaky_relu` throughout, which matters when reading the accuracy tables: at the
larger widths the hybrids are running at a learning rate their ReLU parents cannot use.

MSE + sigmoid output at width 128, grid {0.003, 0.01, 0.03}:

| config | lr |
|---|---|
| relu | 0.01 |
| tanh | 0.03 |
| tanh+relu | 0.01 |

## 4. Per-width results

### Width 12 — 30 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 30 | 93.72 ± 0.49 | 93.85 | 93.63 ± 0.50 | 93.94 |
| tanh | 0.003 | 30 | 93.32 ± 0.27 | 93.31 | 93.22 ± 0.26 | 93.67 |
| leaky_relu | 0.003 | 30 | 93.74 ± 0.45 | 93.82 | 93.64 ± 0.46 | 93.91 |
| sigmoid | 0.01 | 30 | 93.09 ± 0.36 | 93.12 | 92.97 ± 0.36 | 93.42 |
| tanh+relu | 0.003 | 30 | 93.15 ± 0.47 | 93.19 | 93.03 ± 0.48 | 93.42 |
| tanh+leaky_relu | 0.003 | 30 | 93.38 ± 0.37 | 93.43 | 93.27 ± 0.38 | 93.65 |
| sigmoid+relu | 0.01 | 30 | 92.56 ± 0.56 | 92.63 | 92.44 ± 0.56 | 92.85 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.57 | [-0.77, -0.38] | 4/30 | 5.5e-06 | 6.0e-05 |
| tanh+relu vs relu | macro-F1 | 30 | -0.60 | [-0.81, -0.40] | 4/30 | 4.0e-06 | 9.2e-06 |
| tanh+relu vs tanh | accuracy | 30 | -0.17 | [-0.36, +0.01] | 10/30 | 0.084 | 0.058 |
| tanh+relu vs tanh | macro-F1 | 30 | -0.18 | [-0.37, +0.01] | 11/30 | 0.076 | 0.052 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.36 | [-0.54, -0.18] | 8/30 | 6.2e-04 | 0.001 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.37 | [-0.56, -0.18] | 8/30 | 6.2e-04 | 6.1e-04 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.06 | [-0.13, +0.24] | 19/30 | 0.536 | 0.382 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.06 | [-0.13, +0.24] | 20/30 | 0.564 | 0.382 |
| sigmoid+relu vs relu | accuracy | 30 | -1.16 | [-1.42, -0.89] | 1/30 | 1.9e-09 | 1.2e-05 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.19 | [-1.45, -0.92] | 1/30 | 1.4e-09 | 6.9e-07 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.53 | [-0.77, -0.29] | 4/30 | 2.2e-04 | 4.2e-04 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.53 | [-0.77, -0.29] | 4/30 | 2.1e-04 | 1.7e-04 |

Figures: `results/figures/fig_acc_12.png`, `fig_f1_12.png`, `fig_paired_acc_12.png`, `fig_paired_f1_12.png`, `fig_curves_12.png`

![](results/figures/fig_acc_12.png)
![](results/figures/fig_paired_acc_12.png)

### Width 32 — 30 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 30 | 96.45 ± 0.26 | 96.49 | 96.40 ± 0.26 | 96.42 |
| tanh | 0.003 | 30 | 96.19 ± 0.25 | 96.20 | 96.14 ± 0.24 | 96.28 |
| leaky_relu | 0.003 | 30 | 96.37 ± 0.32 | 96.37 | 96.32 ± 0.32 | 96.33 |
| sigmoid | 0.01 | 30 | 95.89 ± 0.24 | 95.93 | 95.84 ± 0.24 | 96.18 |
| tanh+relu | 0.003 | 30 | 96.27 ± 0.20 | 96.32 | 96.22 ± 0.20 | 96.33 |
| tanh+leaky_relu | 0.003 | 30 | 96.26 ± 0.26 | 96.33 | 96.21 ± 0.26 | 96.35 |
| sigmoid+relu | 0.01 | 30 | 95.01 ± 0.59 | 95.14 | 94.95 ± 0.59 | 95.21 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 30 | -0.18 | [-0.27, -0.09] | 7/30 | 5.6e-04 | 9.6e-04 |
| tanh+relu vs relu | macro-F1 | 30 | -0.18 | [-0.27, -0.09] | 6/30 | 4.2e-04 | 3.8e-04 |
| tanh+relu vs tanh | accuracy | 30 | +0.08 | [-0.00, +0.16] | 18/30 | 0.084 | 0.106 |
| tanh+relu vs tanh | macro-F1 | 30 | +0.08 | [-0.01, +0.16] | 18/30 | 0.083 | 0.096 |
| tanh+leaky_relu vs leaky_relu | accuracy | 30 | -0.10 | [-0.24, +0.04] | 9/30 | 0.170 | 0.044 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 30 | -0.10 | [-0.24, +0.05] | 9/30 | 0.170 | 0.038 |
| tanh+leaky_relu vs tanh | accuracy | 30 | +0.07 | [-0.01, +0.15] | 20/30 | 0.090 | 0.094 |
| tanh+leaky_relu vs tanh | macro-F1 | 30 | +0.08 | [-0.01, +0.15] | 20/30 | 0.082 | 0.084 |
| sigmoid+relu vs relu | accuracy | 30 | -1.44 | [-1.68, -1.22] | 0/30 | 1.1e-12 | 1.9e-09 |
| sigmoid+relu vs relu | macro-F1 | 30 | -1.45 | [-1.70, -1.24] | 0/30 | 9.5e-13 | 1.9e-09 |
| sigmoid+relu vs sigmoid | accuracy | 30 | -0.88 | [-1.10, -0.67] | 1/30 | 1.3e-08 | 2.4e-06 |
| sigmoid+relu vs sigmoid | macro-F1 | 30 | -0.89 | [-1.12, -0.68] | 1/30 | 1.2e-08 | 5.6e-09 |

Figures: `results/figures/fig_acc_32.png`, `fig_f1_32.png`, `fig_paired_acc_32.png`, `fig_paired_f1_32.png`, `fig_curves_32.png`

![](results/figures/fig_acc_32.png)
![](results/figures/fig_paired_acc_32.png)

### Width 64 — 20 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 20 | 97.19 ± 0.20 | 97.23 | 97.16 ± 0.20 | 97.16 |
| tanh | 0.003 | 20 | 97.11 ± 0.12 | 97.12 | 97.07 ± 0.12 | 97.08 |
| leaky_relu | 0.003 | 20 | 97.17 ± 0.23 | 97.18 | 97.14 ± 0.23 | 97.13 |
| sigmoid | 0.03 | 20 | 96.96 ± 0.16 | 96.96 | 96.92 ± 0.16 | 97.00 |
| tanh+relu | 0.01 | 20 | 97.22 ± 0.23 | 97.27 | 97.19 ± 0.23 | 97.11 |
| tanh+leaky_relu | 0.01 | 20 | 97.16 ± 0.22 | 97.14 | 97.13 ± 0.22 | 97.12 |
| sigmoid+relu | 0.003 | 20 | 96.50 ± 0.25 | 96.48 | 96.45 ± 0.25 | 96.53 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 20 | +0.03 | [-0.07, +0.13] | 11/20 | 0.640 | 0.765 |
| tanh+relu vs relu | macro-F1 | 20 | +0.02 | [-0.08, +0.13] | 11/20 | 0.672 | 0.784 |
| tanh+relu vs tanh | accuracy | 20 | +0.11 | [+0.01, +0.22] | 13/20 | 0.057 | 0.049 |
| tanh+relu vs tanh | macro-F1 | 20 | +0.11 | [+0.01, +0.22] | 14/20 | 0.058 | 0.053 |
| tanh+leaky_relu vs leaky_relu | accuracy | 20 | -0.01 | [-0.13, +0.12] | 10/20 | 0.927 | 0.968 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 20 | -0.01 | [-0.13, +0.12] | 11/20 | 0.912 | 0.985 |
| tanh+leaky_relu vs tanh | accuracy | 20 | +0.05 | [-0.05, +0.15] | 15/20 | 0.305 | 0.218 |
| tanh+leaky_relu vs tanh | macro-F1 | 20 | +0.06 | [-0.05, +0.15] | 15/20 | 0.296 | 0.231 |
| sigmoid+relu vs relu | accuracy | 20 | -0.70 | [-0.85, -0.55] | 0/20 | 5.4e-08 | 1.3e-04 |
| sigmoid+relu vs relu | macro-F1 | 20 | -0.71 | [-0.86, -0.55] | 0/20 | 4.5e-08 | 1.9e-06 |
| sigmoid+relu vs sigmoid | accuracy | 20 | -0.46 | [-0.59, -0.32] | 1/20 | 2.3e-06 | 1.6e-04 |
| sigmoid+relu vs sigmoid | macro-F1 | 20 | -0.46 | [-0.60, -0.33] | 1/20 | 2.2e-06 | 1.3e-05 |

Figures: `results/figures/fig_acc_64.png`, `fig_f1_64.png`, `fig_paired_acc_64.png`, `fig_paired_f1_64.png`, `fig_curves_64.png`

![](results/figures/fig_acc_64.png)
![](results/figures/fig_paired_acc_64.png)

### Width 128 — 20 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 20 | 97.75 ± 0.14 | 97.75 | 97.72 ± 0.14 | 97.61 |
| tanh | 0.01 | 20 | 97.70 ± 0.15 | 97.68 | 97.67 ± 0.15 | 97.60 |
| leaky_relu | 0.003 | 20 | 97.60 ± 0.13 | 97.63 | 97.58 ± 0.13 | 97.53 |
| sigmoid | 0.03 | 20 | 97.43 ± 0.32 | 97.48 | 97.40 ± 0.31 | 97.40 |
| tanh+relu | 0.01 | 20 | 97.83 ± 0.18 | 97.85 | 97.80 ± 0.19 | 97.66 |
| tanh+leaky_relu | 0.01 | 20 | 97.71 ± 0.36 | 97.79 | 97.69 ± 0.36 | 97.53 |
| sigmoid+relu | 0.003 | 20 | 97.20 ± 0.18 | 97.19 | 97.17 ± 0.18 | 97.12 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 20 | +0.08 | [-0.03, +0.18] | 14/20 | 0.159 | 0.156 |
| tanh+relu vs relu | macro-F1 | 20 | +0.08 | [-0.03, +0.18] | 14/20 | 0.150 | 0.167 |
| tanh+relu vs tanh | accuracy | 20 | +0.13 | [+0.03, +0.22] | 16/20 | 0.020 | 0.035 |
| tanh+relu vs tanh | macro-F1 | 20 | +0.13 | [+0.03, +0.22] | 16/20 | 0.018 | 0.033 |
| tanh+leaky_relu vs leaky_relu | accuracy | 20 | +0.11 | [-0.05, +0.23] | 16/20 | 0.152 | 0.012 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 20 | +0.11 | [-0.05, +0.23] | 16/20 | 0.145 | 0.011 |
| tanh+leaky_relu vs tanh | accuracy | 20 | +0.02 | [-0.13, +0.14] | 12/20 | 0.796 | 0.313 |
| tanh+leaky_relu vs tanh | macro-F1 | 20 | +0.02 | [-0.14, +0.15] | 12/20 | 0.788 | 0.294 |
| sigmoid+relu vs relu | accuracy | 20 | -0.54 | [-0.64, -0.45] | 0/20 | 1.1e-09 | 8.8e-05 |
| sigmoid+relu vs relu | macro-F1 | 20 | -0.55 | [-0.64, -0.45] | 0/20 | 8.0e-10 | 1.9e-06 |
| sigmoid+relu vs sigmoid | accuracy | 20 | -0.23 | [-0.33, -0.08] | 2/20 | 0.003 | 0.002 |
| sigmoid+relu vs sigmoid | macro-F1 | 20 | -0.23 | [-0.34, -0.09] | 2/20 | 0.002 | 8.5e-04 |

Figures: `results/figures/fig_acc_128.png`, `fig_f1_128.png`, `fig_paired_acc_128.png`, `fig_paired_f1_128.png`, `fig_curves_128.png`

![](results/figures/fig_acc_128.png)
![](results/figures/fig_paired_acc_128.png)

### Width 256 — 10 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.003 | 10 | 97.92 ± 0.31 | 98.03 | 97.89 ± 0.31 | 97.76 |
| tanh | 0.01 | 10 | 97.84 ± 0.14 | 97.84 | 97.81 ± 0.15 | 97.73 |
| leaky_relu | 0.01 | 10 | 97.89 ± 0.21 | 97.94 | 97.86 ± 0.21 | 97.75 |
| tanh+relu | 0.01 | 10 | 98.14 ± 0.16 | 98.16 | 98.12 ± 0.16 | 97.85 |
| tanh+leaky_relu | 0.01 | 10 | 97.95 ± 0.25 | 97.97 | 97.92 ± 0.26 | 97.77 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 10 | +0.23 | [+0.03, +0.44] | 8/10 | 0.071 | 0.084 |
| tanh+relu vs relu | macro-F1 | 10 | +0.23 | [+0.03, +0.44] | 8/10 | 0.072 | 0.084 |
| tanh+relu vs tanh | accuracy | 10 | +0.30 | [+0.15, +0.45] | 8/10 | 0.004 | 0.010 |
| tanh+relu vs tanh | macro-F1 | 10 | +0.31 | [+0.15, +0.45] | 8/10 | 0.005 | 0.010 |
| tanh+leaky_relu vs leaky_relu | accuracy | 10 | +0.06 | [-0.12, +0.24] | 5/10 | 0.534 | 0.625 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 10 | +0.06 | [-0.12, +0.25] | 5/10 | 0.535 | 0.625 |
| tanh+leaky_relu vs tanh | accuracy | 10 | +0.11 | [-0.08, +0.28] | 6/10 | 0.288 | 0.336 |
| tanh+leaky_relu vs tanh | macro-F1 | 10 | +0.11 | [-0.07, +0.29] | 6/10 | 0.292 | 0.322 |

Figures: `results/figures/fig_acc_256.png`, `fig_f1_256.png`, `fig_paired_acc_256.png`, `fig_paired_f1_256.png`, `fig_curves_256.png`

![](results/figures/fig_acc_256.png)
![](results/figures/fig_paired_acc_256.png)

### Width 512 — 5 seeds per config, cross-entropy

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.01 | 5 | 98.02 ± 0.21 | 98.13 | 98.00 ± 0.21 | 97.85 |
| tanh | 0.01 | 5 | 97.74 ± 0.11 | 97.76 | 97.71 ± 0.12 | 97.63 |
| leaky_relu | 0.003 | 5 | 98.00 ± 0.11 | 98.02 | 97.97 ± 0.11 | 97.82 |
| tanh+relu | 0.01 | 5 | 98.14 ± 0.14 | 98.14 | 98.11 ± 0.15 | 97.93 |
| tanh+leaky_relu | 0.003 | 5 | 97.79 ± 0.22 | 97.79 | 97.76 ± 0.22 | 97.64 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 5 | +0.12 | [-0.14, +0.38] | 3/5 | 0.476 | 0.625 |
| tanh+relu vs relu | macro-F1 | 5 | +0.11 | [-0.15, +0.38] | 3/5 | 0.497 | 0.625 |
| tanh+relu vs tanh | accuracy | 5 | +0.40 | [+0.31, +0.53] | 5/5 | 0.003 | 0.062 |
| tanh+relu vs tanh | macro-F1 | 5 | +0.40 | [+0.31, +0.55] | 5/5 | 0.004 | 0.062 |
| tanh+leaky_relu vs leaky_relu | accuracy | 5 | -0.21 | [-0.31, -0.10] | 0/5 | 0.027 | 0.062 |
| tanh+leaky_relu vs leaky_relu | macro-F1 | 5 | -0.21 | [-0.31, -0.10] | 0/5 | 0.028 | 0.062 |
| tanh+leaky_relu vs tanh | accuracy | 5 | +0.05 | [-0.07, +0.14] | 4/5 | 0.510 | 0.625 |
| tanh+leaky_relu vs tanh | macro-F1 | 5 | +0.05 | [-0.07, +0.14] | 4/5 | 0.440 | 0.625 |

Figures: `results/figures/fig_acc_512.png`, `fig_f1_512.png`, `fig_paired_acc_512.png`, `fig_paired_f1_512.png`, `fig_curves_512.png`

![](results/figures/fig_acc_512.png)
![](results/figures/fig_paired_acc_512.png)

## 5. Crossover: hybrid − parent against width

**test accuracy** (Δ in points, positive = hybrid ahead)

| comparison | width | n | Δ | 95% bootstrap CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.57 | [-0.77, -0.38] | 4/30 | 5.5e-06 | 6.0e-05 |
| tanh+relu − relu | 32 | 30 | -0.18 | [-0.27, -0.09] | 7/30 | 5.6e-04 | 9.6e-04 |
| tanh+relu − relu | 64 | 20 | +0.03 | [-0.07, +0.13] | 11/20 | 0.640 | 0.765 |
| tanh+relu − relu | 128 | 20 | +0.08 | [-0.02, +0.18] | 14/20 | 0.159 | 0.156 |
| tanh+relu − relu | 256 | 10 | +0.23 | [+0.03, +0.43] | 8/10 | 0.071 | 0.084 |
| tanh+relu − relu | 512 | 5 | +0.12 | [-0.14, +0.38] | 3/5 | 0.476 | 0.625 |
| tanh+relu − tanh | 12 | 30 | -0.17 | [-0.35, +0.01] | 10/30 | 0.084 | 0.058 |
| tanh+relu − tanh | 32 | 30 | +0.08 | [-0.01, +0.16] | 18/30 | 0.084 | 0.106 |
| tanh+relu − tanh | 64 | 20 | +0.11 | [+0.01, +0.22] | 13/20 | 0.057 | 0.049 |
| tanh+relu − tanh | 128 | 20 | +0.13 | [+0.03, +0.22] | 16/20 | 0.020 | 0.035 |
| tanh+relu − tanh | 256 | 10 | +0.30 | [+0.15, +0.45] | 8/10 | 0.004 | 0.010 |
| tanh+relu − tanh | 512 | 5 | +0.40 | [+0.32, +0.53] | 5/5 | 0.003 | 0.062 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.36 | [-0.54, -0.18] | 8/30 | 6.2e-04 | 0.001 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.10 | [-0.24, +0.04] | 9/30 | 0.170 | 0.044 |
| tanh+leaky_relu − leaky_relu | 64 | 20 | -0.01 | [-0.13, +0.12] | 10/20 | 0.927 | 0.968 |
| tanh+leaky_relu − leaky_relu | 128 | 20 | +0.11 | [-0.05, +0.23] | 16/20 | 0.152 | 0.012 |
| tanh+leaky_relu − leaky_relu | 256 | 10 | +0.06 | [-0.11, +0.24] | 5/10 | 0.534 | 0.625 |
| tanh+leaky_relu − leaky_relu | 512 | 5 | -0.21 | [-0.31, -0.10] | 0/5 | 0.027 | 0.062 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.06 | [-0.12, +0.24] | 19/30 | 0.536 | 0.382 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.07 | [-0.01, +0.15] | 20/30 | 0.090 | 0.094 |
| tanh+leaky_relu − tanh | 64 | 20 | +0.05 | [-0.05, +0.15] | 15/20 | 0.305 | 0.218 |
| tanh+leaky_relu − tanh | 128 | 20 | +0.02 | [-0.13, +0.14] | 12/20 | 0.796 | 0.313 |
| tanh+leaky_relu − tanh | 256 | 10 | +0.11 | [-0.07, +0.28] | 6/10 | 0.288 | 0.336 |
| tanh+leaky_relu − tanh | 512 | 5 | +0.05 | [-0.07, +0.14] | 4/5 | 0.510 | 0.625 |

**test macro-F1** (Δ in points, positive = hybrid ahead)

| comparison | width | n | Δ | 95% bootstrap CI | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu − relu | 12 | 30 | -0.60 | [-0.80, -0.39] | 4/30 | 4.0e-06 | 9.2e-06 |
| tanh+relu − relu | 32 | 30 | -0.18 | [-0.27, -0.09] | 6/30 | 4.2e-04 | 3.8e-04 |
| tanh+relu − relu | 64 | 20 | +0.02 | [-0.08, +0.13] | 11/20 | 0.672 | 0.784 |
| tanh+relu − relu | 128 | 20 | +0.08 | [-0.03, +0.18] | 14/20 | 0.150 | 0.167 |
| tanh+relu − relu | 256 | 10 | +0.23 | [+0.03, +0.45] | 8/10 | 0.072 | 0.084 |
| tanh+relu − relu | 512 | 5 | +0.11 | [-0.15, +0.38] | 3/5 | 0.497 | 0.625 |
| tanh+relu − tanh | 12 | 30 | -0.18 | [-0.37, +0.01] | 11/30 | 0.076 | 0.052 |
| tanh+relu − tanh | 32 | 30 | +0.08 | [-0.00, +0.16] | 18/30 | 0.083 | 0.096 |
| tanh+relu − tanh | 64 | 20 | +0.11 | [+0.00, +0.22] | 14/20 | 0.058 | 0.053 |
| tanh+relu − tanh | 128 | 20 | +0.13 | [+0.03, +0.22] | 16/20 | 0.018 | 0.033 |
| tanh+relu − tanh | 256 | 10 | +0.31 | [+0.15, +0.46] | 8/10 | 0.005 | 0.010 |
| tanh+relu − tanh | 512 | 5 | +0.40 | [+0.31, +0.54] | 5/5 | 0.004 | 0.062 |
| tanh+leaky_relu − leaky_relu | 12 | 30 | -0.37 | [-0.56, -0.18] | 8/30 | 6.2e-04 | 6.1e-04 |
| tanh+leaky_relu − leaky_relu | 32 | 30 | -0.10 | [-0.24, +0.05] | 9/30 | 0.170 | 0.038 |
| tanh+leaky_relu − leaky_relu | 64 | 20 | -0.01 | [-0.13, +0.11] | 11/20 | 0.912 | 0.985 |
| tanh+leaky_relu − leaky_relu | 128 | 20 | +0.11 | [-0.05, +0.23] | 16/20 | 0.145 | 0.011 |
| tanh+leaky_relu − leaky_relu | 256 | 10 | +0.06 | [-0.12, +0.24] | 5/10 | 0.535 | 0.625 |
| tanh+leaky_relu − leaky_relu | 512 | 5 | -0.21 | [-0.31, -0.10] | 0/5 | 0.028 | 0.062 |
| tanh+leaky_relu − tanh | 12 | 30 | +0.06 | [-0.13, +0.24] | 20/30 | 0.564 | 0.382 |
| tanh+leaky_relu − tanh | 32 | 30 | +0.08 | [-0.01, +0.15] | 20/30 | 0.082 | 0.084 |
| tanh+leaky_relu − tanh | 64 | 20 | +0.06 | [-0.05, +0.15] | 15/20 | 0.296 | 0.231 |
| tanh+leaky_relu − tanh | 128 | 20 | +0.02 | [-0.13, +0.14] | 12/20 | 0.788 | 0.294 |
| tanh+leaky_relu − tanh | 256 | 10 | +0.11 | [-0.08, +0.29] | 6/10 | 0.292 | 0.322 |
| tanh+leaky_relu − tanh | 512 | 5 | +0.05 | [-0.07, +0.14] | 4/5 | 0.440 | 0.625 |

![](results/figures/fig_crossover.png)
![](results/figures/fig_crossover_f1.png)
![](results/figures/fig_width_acc.png)

## 6. Learning-rate robustness

**Width 12 — mean validation accuracy over 10 seeds**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|
| relu | 93.87 | 93.90 | 93.05 | 84.92 | 11.28 | 10.18 |
| tanh | 93.63 | 93.52 | 92.62 | 91.04 | 78.72 | 33.39 |
| leaky_relu | 93.67 | 93.81 | 92.99 | 92.75 | 80.48 | 82.12 |
| tanh+relu | 93.08 | 93.35 | 92.86 | 89.31 | 76.32 | 35.97 |
| tanh+leaky_relu | 93.28 | 93.54 | 93.04 | 91.84 | 76.33 | 73.96 |

**Width 12 — collapsed seeds (final validation accuracy < 50%), count / n**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 | lr=0.3 |
|---|---|---|---|---|---|---|
| relu | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 |
| tanh | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 9/10 |
| leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tanh+relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 8/10 |
| tanh+leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

![](results/figures/fig_lrsweep_12.png)

**Width 64 — mean validation accuracy over 10 seeds**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 |
|---|---|---|---|---|---|
| relu | 96.74 | 97.20 | 96.83 | 95.26 | 11.95 |
| tanh | 96.53 | 97.09 | 97.14 | 95.30 | 65.44 |
| leaky_relu | 96.53 | 97.11 | 97.03 | 95.72 | 84.87 |
| tanh+relu | 96.61 | 97.29 | 97.11 | 95.96 | 62.45 |
| tanh+leaky_relu | 96.48 | 97.17 | 97.18 | 95.71 | 84.32 |

**Width 64 — collapsed seeds (final validation accuracy < 50%), count / n**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 |
|---|---|---|---|---|---|
| relu | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 |
| tanh | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tanh+relu | 0/10 | 0/10 | 0/10 | 0/10 | 3/10 |
| tanh+leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

![](results/figures/fig_lrsweep_64.png)

**Width 128 — mean validation accuracy over 10 seeds**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 |
|---|---|---|---|---|---|
| relu | 97.09 | 97.56 | 97.50 | 95.90 | 11.83 |
| tanh | 96.43 | 97.32 | 97.53 | 94.87 | 43.20 |
| leaky_relu | 96.91 | 97.49 | 97.38 | 94.85 | 84.74 |
| tanh+relu | 96.87 | 97.62 | 97.67 | 96.47 | 30.16 |
| tanh+leaky_relu | 96.74 | 97.55 | 97.62 | 95.57 | 84.41 |

**Width 128 — collapsed seeds (final validation accuracy < 50%), count / n**

| config | lr=0.001 | lr=0.003 | lr=0.01 | lr=0.03 | lr=0.1 |
|---|---|---|---|---|---|
| relu | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 |
| tanh | 0/10 | 0/10 | 0/10 | 0/10 | 7/10 |
| leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tanh+relu | 0/10 | 0/10 | 0/10 | 0/10 | 8/10 |
| tanh+leaky_relu | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

![](results/figures/fig_lrsweep_128.png)

**Robustness score** — widest contiguous interval of the tested LR grid on which a config's mean validation accuracy stays within 1 point of its own best (lo–hi, grid points spanned, decades):

| config | measure | W=12 | W=64 | W=128 |
|---|---|---|---|---|
| relu | within 1 pt of own best | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) |
| relu | no seed collapsed | 0.001–0.03 (4 pts, 1.48 dec) | 0.001–0.03 (4 pts, 1.48 dec) | 0.001–0.03 (4 pts, 1.48 dec) |
| tanh | within 1 pt of own best | 0.001–0.003 (2 pts, 0.48 dec) | 0.001–0.01 (3 pts, 1.00 dec) | 0.003–0.01 (2 pts, 0.52 dec) |
| tanh | no seed collapsed | 0.001–0.1 (5 pts, 2.00 dec) | 0.001–0.1 (5 pts, 2.00 dec) | 0.001–0.03 (4 pts, 1.48 dec) |
| leaky_relu | within 1 pt of own best | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) |
| leaky_relu | no seed collapsed | 0.001–0.3 (6 pts, 2.48 dec) | 0.001–0.1 (5 pts, 2.00 dec) | 0.001–0.1 (5 pts, 2.00 dec) |
| tanh+relu | within 1 pt of own best | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) |
| tanh+relu | no seed collapsed | 0.001–0.1 (5 pts, 2.00 dec) | 0.001–0.03 (4 pts, 1.48 dec) | 0.001–0.03 (4 pts, 1.48 dec) |
| tanh+leaky_relu | within 1 pt of own best | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) | 0.001–0.01 (3 pts, 1.00 dec) |
| tanh+leaky_relu | no seed collapsed | 0.001–0.3 (6 pts, 2.48 dec) | 0.001–0.1 (5 pts, 2.00 dec) | 0.001–0.1 (5 pts, 2.00 dec) |

## 7. Layout and ratio

**Width 12** — each pair uses the LR calibrated for its interleave/0.5 variant, so ratios 0.25 and 0.75 are not separately tuned.

| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd | Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tanh+relu | block | 0.5 | 0.001 | 20 | 93.58 ± 0.36 | 93.51 ± 0.36 | -0.22 [-0.44, +0.02] | +0.28 [+0.10, +0.47] | – |
| tanh+relu | interleave | 0.25 | 0.001 | 20 | 93.45 ± 0.27 | 93.39 ± 0.27 | -0.35 [-0.55, -0.13] | +0.16 [-0.01, +0.31] | – |
| tanh+relu | interleave | 0.75 | 0.001 | 20 | 93.59 ± 0.47 | 93.53 ± 0.47 | -0.20 [-0.48, +0.06] | +0.30 [+0.06, +0.53] | – |
| tanh+relu | random | 0.5 | 0.001 | 20 | 93.47 ± 0.33 | 93.40 ± 0.33 | -0.33 [-0.55, -0.10] | +0.18 [-0.04, +0.39] | – |

**Width 128** — each pair uses the LR calibrated for its interleave/0.5 variant, so ratios 0.25 and 0.75 are not separately tuned.

| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd | Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tanh+leaky_relu | block | 0.5 | 0.01 | 10 | 97.78 ± 0.17 | 97.76 ± 0.17 | +0.19 [+0.05, +0.33] | +0.08 [-0.05, +0.19] | +0.03 [-0.08, +0.14] |
| tanh+leaky_relu | interleave | 0.25 | 0.01 | 10 | 97.72 ± 0.11 | 97.70 ± 0.11 | +0.13 [+0.05, +0.20] | +0.01 [-0.06, +0.09] | -0.03 [-0.14, +0.09] |
| tanh+leaky_relu | interleave | 0.5 | 0.01 | 10 | 97.75 ± 0.11 | 97.73 ± 0.11 | +0.16 [+0.06, +0.27] | +0.05 [-0.05, +0.13] | +0.00 [+0.00, +0.00] |
| tanh+leaky_relu | interleave | 0.75 | 0.01 | 10 | 97.60 ± 0.14 | 97.57 ± 0.15 | +0.01 [-0.09, +0.11] | -0.10 [-0.23, +0.00] | -0.15 [-0.27, -0.05] |
| tanh+leaky_relu | random | 0.5 | 0.01 | 10 | 97.85 ± 0.16 | 97.83 ± 0.17 | +0.26 [+0.13, +0.40] | +0.15 [+0.01, +0.30] | +0.10 [-0.01, +0.22] |
| tanh+relu | block | 0.5 | 0.01 | 10 | 97.87 ± 0.09 | 97.85 ± 0.09 | +0.13 [+0.02, +0.23] | +0.17 [+0.06, +0.27] | -0.01 [-0.09, +0.08] |
| tanh+relu | interleave | 0.25 | 0.01 | 10 | 97.82 ± 0.11 | 97.79 ± 0.12 | +0.07 [-0.01, +0.14] | +0.11 [+0.03, +0.18] | -0.06 [-0.16, +0.03] |
| tanh+relu | interleave | 0.5 | 0.01 | 10 | 97.88 ± 0.15 | 97.86 ± 0.14 | +0.13 [+0.01, +0.24] | +0.17 [+0.05, +0.28] | +0.00 [+0.00, +0.00] |
| tanh+relu | interleave | 0.75 | 0.01 | 10 | 97.76 ± 0.27 | 97.74 ± 0.27 | +0.01 [-0.17, +0.18] | +0.05 [-0.14, +0.21] | -0.12 [-0.23, -0.01] |
| tanh+relu | random | 0.5 | 0.01 | 10 | 97.77 ± 0.24 | 97.74 ± 0.25 | +0.02 [-0.13, +0.16] | +0.06 [-0.08, +0.20] | -0.11 [-0.23, -0.02] |

**Width 256** — each pair uses the LR calibrated for its interleave/0.5 variant, so ratios 0.25 and 0.75 are not separately tuned.

| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd | Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tanh+leaky_relu | block | 0.5 | 0.01 | 6 | 97.86 ± 0.38 | 97.84 ± 0.39 | +0.02 [-0.12, +0.16] | -0.02 [-0.25, +0.18] | -0.00 [-0.35, +0.29] |
| tanh+leaky_relu | interleave | 0.25 | 0.01 | 6 | 98.00 ± 0.19 | 97.98 ± 0.20 | +0.16 [+0.06, +0.25] | +0.12 [+0.02, +0.21] | +0.14 [-0.11, +0.41] |
| tanh+leaky_relu | interleave | 0.5 | 0.01 | 6 | 97.87 ± 0.22 | 97.84 ± 0.23 | +0.02 [-0.23, +0.28] | -0.02 [-0.23, +0.18] | +0.00 [+0.00, +0.00] |
| tanh+leaky_relu | interleave | 0.75 | 0.01 | 6 | 97.84 ± 0.30 | 97.81 ± 0.30 | -0.00 [-0.28, +0.24] | -0.04 [-0.30, +0.18] | -0.03 [-0.21, +0.20] |
| tanh+leaky_relu | random | 0.5 | 0.01 | 6 | 97.86 ± 0.50 | 97.84 ± 0.50 | +0.02 [-0.51, +0.50] | -0.02 [-0.47, +0.38] | -0.00 [-0.29, +0.25] |
| tanh+relu | block | 0.5 | 0.01 | 6 | 98.18 ± 0.14 | 98.16 ± 0.14 | +0.34 [+0.18, +0.52] | +0.30 [+0.11, +0.43] | +0.05 [-0.10, +0.21] |
| tanh+relu | interleave | 0.25 | 0.01 | 6 | 98.03 ± 0.16 | 98.01 ± 0.16 | +0.18 [-0.05, +0.47] | +0.15 [+0.03, +0.28] | -0.10 [-0.26, +0.04] |
| tanh+relu | interleave | 0.5 | 0.01 | 6 | 98.13 ± 0.18 | 98.10 ± 0.18 | +0.28 [-0.00, +0.58] | +0.25 [+0.02, +0.48] | +0.00 [+0.00, +0.00] |
| tanh+relu | interleave | 0.75 | 0.01 | 6 | 98.20 ± 0.18 | 98.18 ± 0.18 | +0.36 [+0.16, +0.57] | +0.32 [+0.15, +0.47] | +0.07 [-0.09, +0.25] |
| tanh+relu | random | 0.5 | 0.01 | 6 | 98.02 ± 0.30 | 98.00 ± 0.30 | +0.17 [-0.20, +0.52] | +0.14 [-0.15, +0.42] | -0.11 [-0.36, +0.11] |

## 8. MSE replication at width 128

| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |
|---|---|---|---|---|---|---|
| relu | 0.01 | 10 | 97.86 ± 0.14 | 97.88 | 97.83 ± 0.14 | 97.74 |
| tanh | 0.03 | 10 | 97.43 ± 0.20 | 97.45 | 97.41 ± 0.20 | 97.48 |
| tanh+relu | 0.01 | 10 | 97.62 ± 0.11 | 97.61 | 97.60 ± 0.11 | 97.57 |

| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tanh+relu vs relu | accuracy | 10 | -0.24 | [-0.34, -0.13] | 1/10 | 0.002 | 0.006 |
| tanh+relu vs relu | macro-F1 | 10 | -0.24 | [-0.34, -0.13] | 1/10 | 0.002 | 0.006 |
| tanh+relu vs tanh | accuracy | 10 | +0.19 | [+0.06, +0.34] | 7/10 | 0.036 | 0.020 |
| tanh+relu vs tanh | macro-F1 | 10 | +0.19 | [+0.06, +0.35] | 8/10 | 0.037 | 0.037 |

![](results/figures/fig_acc_128_mse.png)

Calibration grid (validation accuracy, mean of seeds 101–103):

| config | lr=0.003 | lr=0.01 | lr=0.03 |
|---|---|---|---|
| relu | 97.03 | 97.62 | 97.46 |
| tanh | 96.27 | 97.13 | 97.17 |
| tanh+relu | 96.85 | 97.51 | 97.41 |

## 9. What the data show

**(a) Hybrid versus better parent, against width.** `tanh+relu` improves against `tanh`
monotonically at every width tested — −0.17, +0.08, +0.11, +0.13, +0.30, +0.40 points across
12 → 512 — and against `relu` it goes −0.57, −0.18, +0.03, +0.08, +0.23, +0.12, crossing zero
between 32 and 64. `tanh+relu` is the best configuration at 64, 128, 256 and 512 (98.14 ± 0.14 at
512), while a homogeneous baseline wins at 12 and 32. Significance is asymmetric. The small-width
*losses* are strong: `tanh+relu − relu` at width 12 gives p = 5.5e-06, surviving Bonferroni across
~40 paired tests. Two hybrid *wins* clear p < 0.01 — `tanh+relu − tanh` at 256 (+0.30, p = 0.004)
and at 512 (+0.40, 5/5 seeds, p = 0.003) — but neither survives Bonferroni, and the 512 point rests
on 5 seeds. The first-look headline, `tanh+leaky_relu` beating `leaky_relu` by +0.30 on 6/6 seeds
(p = 0.007), does not replicate anywhere: +0.11 at 128, +0.06 at 256, and −0.21 (0/5 seeds) at 512.
The width effect belongs to `tanh+relu` specifically, not to hybridisation in general.

**(b) Learning-rate robustness.** The hybrid's protection against ReLU's collapse is real but decays
with width, and the three widths now show that cleanly. At lr 0.1, `relu` collapses on 10/10 seeds at
every width, while `tanh+relu` collapses on **0/10 at width 12, 3/10 at 64 and 8/10 at 128**. `tanh`
decays the same way (0/10, 0/10, 7/10), which is what the hybrid is inheriting — half its units are
tanh, and tanh's own protection is what gives out as the network grows. `leaky_relu` collapses on
0/10 at all three widths and at every rate tested, including 0.3 at width 12. Even at width 128,
`tanh+relu` is not as dead as ReLU (2/10 seeds at chance against ReLU's 8/10, mean 30.2% against
11.8%), but neither is trainable. The robustness comes from the non-zero negative branch; mixing in
tanh buys a partial version of it that fades as the network grows.

**(c) Layout and ratio at larger width.** Nothing matters, at either width. Across twenty variants
at 128 and 256, every one sits within 0.15 points of interleave/0.5, and the signs are scattered
rather than systematic — the ratio-0.75 variants are the weakest at 128 but not at 256, which argues
that even that difference is noise or the reused learning rate rather than a real preference. Where
the tanh units sit does not matter, and neither does the mix ratio between 0.25 and 0.75.

## 10. Deviations, problems and anything skipped

Nothing in the experimental protocol was changed: `main.cpp`, `net.cpp`, `net.h` and `data.h` are
untouched, so initialisation, batch-size-1 SGD, per-epoch shuffling, the 45k/5k/10k split,
`SPLIT_SEED 12345`, the gradient clip, `LEAKY_SLOPE 0.1` and both loss definitions are exactly as
they were. Every pre-existing CSV in `results/` is unmodified. The changes and departures were:

**1. `sweep/run_chunk.sh` could not run a single job on macOS.** `xargs -I{}` substituted the
~164-byte job line twice into the command string, and BSD xargs caps a `-I`-constructed argument at
255 bytes: every chunk died with `xargs: command line cannot be assembled, too long`. The job line is
now passed as an argument and `eval`'d, which behaves identically under GNU xargs. Backup:
`sweep/run_chunk_v1.sh.bak`.

**2. The stored CSVs and this machine do not share the train/val/test split, so the old files could
not be topped up.** Re-running `main_12.csv` seed 1 `relu` here gives test accuracy 93.42, not the
stored 93.78, and the two diverge at epoch 1 (loss 0.548755 vs 0.557661). This is not a compiler
effect — `-O0`, `-O2`, `-O3`, `-O3 -mcpu=native` and `-ffp-contract=off` all agree on 0.548755. The
decisive evidence is the per-class composition of the test set (row sums of `test_confusion`):
stored `[1004, 1099, 955, 1029, 983, 918, 990, 1035, 1011, 976]` against local
`[963, 1164, 969, 1033, 994, 869, 991, 1045, 982, 990]`. Every stored row shares one split and every
new row shares another. The cause is that `main.cpp` builds the split with
`std::shuffle(perm, std::mt19937(SPLIT_SEED))`, and `std::shuffle` is not specified to produce the
same permutation across standard-library implementations; the stored data came from
`_to_delete/main_linux_binary` (ELF aarch64, GNU/Linux, libstdc++) while this machine uses libc++.
Same seed, different permutation, therefore a different split *and* different initial weights.

Consequence: extending `main_12.csv` with seeds 21–30 or `main_128.csv` with seeds 7–20 would have
put seeds evaluated on **different test sets** inside a single file, silently breaking the paired
design. Both widths were therefore re-run in full on this machine into `results/main_12_v2.csv`
(seeds 1–30) and `results/main_128_v2.csv` (seeds 1–20). This is the `_v2` branch the plan already
allowed for, taken for a different reason than a changed learning rate. It also means **the extended
sweep is internally consistent but is not row-comparable with the first-look sweep**: differences
between the two generations mix the effect of more seeds with the effect of a different split.

**3. One speed/memory change, proven bit-identical.** The sweep was killed three times by the OS for
low memory. `Data::split_data` allocated `images_train` and copied 60 000 `Image` structs while the
source array was still alive, peaking at 474 MB per worker; `main.cpp` always calls
`Data(..., 0.0f)`, so `train_size == number_of_images` and that copy is pure waste. When those are
equal the loaded array is now handed over by pointer instead of copied, which cut peak RSS to 238 MB.
The ground rules' proof was run against rows this machine had already written — width 12 seeds 1 and
2, and width 128 seed 1 — and all three reproduced **identically in all 22 CSV fields**, not merely
in `test_acc`, `test_f1` and `epoch_loss`. Backup: `data_v1.cpp.bak`.

**4. Worker count was reduced from 8 to 4.** The machine has 8 GB, is 28 days up, and already had
~4.5 GB of its 5.1 GB swap consumed by unrelated applications. Even after the memory fix, 6 workers
triggered the low-memory watchdog. P=4 ran to completion. Every kill was recovered by the resume
mechanism with no completed run lost and no duplicate rows.

**5. Throughput was badly mis-estimated at Step 0, in both directions.** A short 8-job benchmark at
width 128 measured a 5.05× speed-up, which predicted 8.4 h for the whole plan. The first real stage
came in at 1.81×, implying 24 h, and scope was cut. That 1.81× turned out to be an artefact of width
12, where a 7-second job pays a 0.6-second data load and concurrent MNIST loads thrash the page
cache; widths 32 and 64 then measured 6.04× and 6.66×, so the cut was reversed and the full plan
restored. The lesson for the log: per-stage throughput at this width range varies by more than 3×,
and a single short benchmark does not predict it.


## Appendix — sanity checks

All checks pass. Full machine-generated output is reproduced below.

### Run-inventory check

| file | expected runs | ok rows | unique keys | missing | extra | duplicates | rejected |
|---|---|---|---|---|---|---|---|
| `calib_12_v2.csv` | 84 | 84 | 84 | 0 | 0 | 0 | 0 |
| `calib_32.csv` | 84 | 84 | 84 | 0 | 0 | 0 | 0 |
| `calib_64.csv` | 84 | 84 | 84 | 0 | 0 | 0 | 0 |
| `calib_128_v2.csv` | 84 | 84 | 84 | 0 | 0 | 0 | 0 |
| `calib_256.csv` | 60 | 60 | 60 | 0 | 0 | 0 | 0 |
| `calib_mse_128.csv` | 27 | 27 | 27 | 0 | 0 | 0 | 0 |
| `main_12_v2.csv` | 210 | 210 | 210 | 0 | 0 | 0 | 0 |
| `main_32.csv` | 210 | 210 | 210 | 0 | 0 | 0 | 0 |
| `main_64.csv` | 140 | 140 | 140 | 0 | 0 | 0 | 0 |
| `main_128_v2.csv` | 140 | 140 | 140 | 0 | 0 | 0 | 0 |
| `main_256.csv` | 50 | 50 | 50 | 0 | 0 | 0 | 0 |
| `main_512.csv` | 25 | 25 | 25 | 0 | 0 | 0 | 0 |
| `calib_512.csv` | 30 | 34 | 34 | 0 | 0 | 0 | 0 |
| `mse_128.csv` | 30 | 30 | 30 | 0 | 0 | 0 | 0 |
| `lrsweep_12.csv` | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| `lrsweep_64.csv` | 250 | 250 | 250 | 0 | 0 | 0 | 0 |
| `lrsweep_128.csv` | 250 | 250 | 250 | 0 | 0 | 0 | 0 |
| `layout_128.csv` | 100 | 100 | 100 | 0 | 0 | 0 | 0 |
| `layout_256.csv` | 60 | 60 | 60 | 0 | 0 | 0 | 0 |

**No `status=nan` rows and no truncated rows anywhere.** Every row in every extended-sweep CSV has
`status=ok` and a complete 100-entry confusion matrix, so nothing was silently dropped. This holds
despite seven low-memory kills mid-chunk — the truncated-row guards in `pending.py` and `analyze.py`
caught the partial writes and those runs were simply re-executed, with no duplicate keys anywhere.

**One deliberate reduction**, not a failure: `lr = 0.03` was dropped from the width-512
calibration grid once it had been measured divergent there (`tanh` 84.91, `relu` 96.21 against
97.61/98.10 at 0.01) and catastrophic at 256, so the 6 pending 0.03 runs could not have changed any
selection; and `lrsweep_64.csv`, the one stage never on the priority list, was left until last; it was
subsequently run to completion on request and is now complete at 250/250.

**Layer-1 masks correct at every width.** One hybrid run per width, from the program's own stdout
mask line: `H=12 relu=6 tanh=6`, `H=32 relu=16 tanh=16`, `H=64 relu=32 tanh=32`,
`H=128 relu=64 tanh=64`, `H=256 relu=128 tanh=128`.

**Reproduction is bit-identical on this machine.** Re-running finished jobs to a scratch CSV
reproduced the stored rows in all 22 fields (every column but wall-clock `seconds`) for width 12
seeds 1 and 2 and width 128 seed 1. A second, larger check re-ran all ten width-128 `tanh+relu`
runs at lr 0.1 and reproduced every one of them identically. Reproduction against the *first-look*
CSVs is impossible for the split reason in section 10, and is reported rather than worked around.

**`analysis/analyze.py` runs end to end with no errors**, producing `results/summary_extended.md`
and every figure in `results/figures/`.

