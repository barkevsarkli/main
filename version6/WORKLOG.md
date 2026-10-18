# version6 work log

All times local (Europe/Istanbul). Machine: Apple M1 (4P + 4E cores, 8 GB), macOS 26.6,
Apple clang 21.0.0, CMake 4.4.3, Python 3.11.9.

## 2026-10-18

### Phase 0 — git snapshot
- Repository state: branch `master`, up to date with `origin/master`
  (`git@github.com:barkevsarkli/main.git`). Untracked: `version5.zip` (43.6 MB, now
  ignored by `*.zip`) and the version6 prompt.
- Files > 5 MB in the tree: six CIFAR-10 batches (30.7 MB each, already tracked),
  MNIST training images (47.0 MB, tracked), `version5.zip` (43.6 MB), two PDFs in
  `Claude outputs/` (6.5 and 9.1 MB, unrelated). Nothing above 50 MB.
- `version5/results_aws/` was entirely gitignored. Barkev decided (DECISIONS 2–3):
  datasets stay tracked, results_aws is committed without zips, render temp files,
  LaTeX intermediates, `Claude outputs/` or `financial_analyst_colab/`.
- Secret scan (AWS keys, private keys, tokens, .env, .pem, SSH details): nothing found.
  `aws_command.txt` (already tracked) mentions only `<PUBLIC_IP>` placeholders and a
  key *file name*.
- Snapshot commit **`b9a723d`** (280 files, ~45 MB), tag **`v5-snapshot-2026-10`**,
  both pushed and checked with `git ls-remote`. Branch **`v6-cnn`** created and pushed.

### Phase 1–4 — code
- **`fb0dc5c`** `version6: import version5 sources unchanged`: main.cpp, net.*, rng_util.h,
  dataset.h, mnist_data.*, cifar10_data.*.
- **`eb4a13e`**: trainer (`mlp1`/`mlp2`/`cnn`), platform file layer, split RNG streams,
  kernel banks + `kernel_dump` + gallery, diagnostics, C++ tests, CMake with build-time
  provenance.
- **`26e4f0d`**: sweep runner, merger, job generators, regression / PyTorch / run tests,
  faster conv and dense kernels.
- Kernel banks generated: 48 JSON files (2 datasets × 6 families × 4 K) in 19 s. τ_K:
  MNIST 0.173 / 0.156 / 0.169 / 0.170 and CIFAR-10 0.136 / 0.132 / 0.149 / 0.151 for
  K = 3 / 5 / 7 / 9.

### Findings while testing
- **MSE gradient scale**: the first gradient-check failures were exactly ×10, only with
  MSE. Cause: version5 back-propagates the derivative of the *sum* of squared errors but
  logs the *mean* (DECISIONS 9). The behaviour is kept; the check now differentiates the
  sum.
- **FMA contraction**: version6 `legacy` matched version5 built with the same flags 8/8.
  Against `version5/reference/determinism_mac.csv` the ReLU and sigmoid rows differed.
  version5 rebuilt with plain `-O3` (Apple clang contracts by default) reproduces that
  reference exactly, so the reference came from an FMA-contracted build (DECISIONS 5).
- **PyTorch**: a constant 2.2e-8 weight difference after the first step came from version5's
  float32 learning rate. With the same rounding in torch, the error is 5e-16 to 1e-15
  after 200 steps.
- **Gabor K = 3** with the specified λ = 2 gives two all-zero filters (DECISIONS 17).

### Performance (single process, `bin/profile_layers`, CIFAR-10 H = 128 K = 9, tanh+relu, s/epoch = 45k steps + 15k evaluation forwards)

| layer | first version | final |
|---|---|---|
| conv forward (incl. activation and pool) | 16.4 | 11.1 |
| conv backward | 5.9 | 5.8 |
| hidden dense (fwd + bwd) | 6.8 | 5.2 |
| output dense | 0.04 | 0.04 |
| loss, data | 0.5 | 0.5 |
| **total** | **29.7** | **~22.7** |

Changes: the conv forward unrolls the kernel row with K known at compile time, which
keeps the (ci, ky, kx) order and is bit-identical to the reference loops; a 16-lane fixed-
order dense dot product (DECISIONS 6); the conv backward unrolls for K = 3, 5, 7 (generic
loop at K = 9, which measured faster). One attempt (an 8-wide register block over x with
a runtime K) was 2.2× *slower* and was dropped.

### Phase 5 — verification (final binary, commit d97603f, clean build; log: `results/final_verification.log`)

| # | test | result | detail |
|---|---|---|---|
| 1 | loader equivalence | PASS | 60,000 MNIST + 60,000 CIFAR-10 images: mmap vs ifstream identical floats and labels; CIFAR plane means 0.4914 / 0.4822 / 0.4465 confirm R, G, B order; known first labels |
| 2 | version5 regression | PASS (POSIX variant) | version5 built unchanged with the same compiler and flags vs version6 `mlp2 --rng-mode legacy`: 8/8 rows bit-identical (MNIST relu, MNIST tanh+relu, CIFAR tanh+relu, MNIST tanh+relu random layout; seeds 1, 2; 3 epochs). Also 8/8 on Ubuntu 24.04 / GCC 13.3 / aarch64 (Docker) |
| 3 | gradient check | PASS | 5,768 parameters, float64 production templates, CNN both input shapes × K {3, 9} × P {1, 2, 4} × 4 activation pairs, frozen CNN, mlp1/mlp2 × 4 activations × CE/MSE; update = −lr·grad to 1e-9 |
| 4 | reference convolution | PASS | float vs double reference: worst 2.3e-7 (forward) / 2.5e-7 (backward) relative to Σ\|terms\| (limit 1e-6); forward bit-identical to same-order reference; bit-reproducible |
| 5 | PyTorch equivalence | PASS | torch 2.8.0 float64, 200 SGD steps, random / bank_init / frozen on MNIST and CIFAR-10: max relative weight error 4.6e-16 to 1.5e-15 |
| 6 | kernel banks | PASS | 1,547 checks: shapes, unit norm, zero sum, K=3 Sobel = classic, Gaussians ≥ 0, stored banks = make_bank, gains finite > 0, galleries exist |
| 7 | RNG alignment | PASS | 20 configurations (4 kernel modes/families × 5 activation setups incl. random layout) share one dense init hash; epoch-1 order identical incl. mlp1 |
| 8 | frozen mode | PASS | conv W and b bit-identical after 3,000 steps; bank_init conv changes |
| 9 | determinism / workers | PASS | same job twice identical; 4 jobs with 1 vs 4 workers identical; resumption skips finished runs |
| 10 | diagnostics invariance | PASS | rows identical with stats on/off; hand-built network: dead / nearly-dead / saturated counts exact, conv dead channel found |
| 11 | collapse abort | PASS | fires after N collapsed epochs, resets on a good epoch, respects both thresholds, N = 0 never fires |
| 12 | smoke | PASS | 108 runs (mlp1, mlp2, every cnn mode × family × K, both datasets, H = 64, tanh+relu, 2 epochs): all ok; learned modes 47–98% validation accuracy |
| – | selection rules | PASS | `tests/test_selection.py`: ties, 2 pp window, stress-LR fallback, lower-LR tie-break |
| – | non-finite | PASS | lr 1e30 → status nan in epoch 1, row still written |

Extra cross-checks:
- GCC 16 + libstdc++ (macOS) and Apple clang + libc++ give bit-identical CNN rows. Ubuntu
  24.04 / GCC 13.3 / glibc (Docker, aarch64) gives the same row too (DECISIONS 38).
- Mini stage (81 runs, 3 epochs) exercised `summarise.py`, `figures.py` and
  `select_stage2.py` end to end (`results/mini/`).

### Phase 7 — benchmark (`results/bench/bench.md`, 42.7 min)
- Worker scaling on the M1: 380 / 547 / 673 jobs/h at 4 / 6 / 8 workers → 8 chosen.
- No thermal drift. Low Power Mode was on (left as found).
- Stats: 4.4 KB per measurement point, about 180 KB per 40-epoch run.
- AWS ratio 2.25 (version5 code path: 43.5 s/epoch here vs 19.4 on c8g.48xlarge).
- Estimates (`results/bench/estimates.md`): full plan ≈ 14 h on one c8g.48xlarge (≈ $37
  spot). On the laptop Stage 1 needs 198 h and Stage 2 495 h; the prescribed reductions
  only reach 78 h and 275 h.

## 2026-10-18 — independent review and fixes (after f49ddbb)
- Independent pre-launch review: `ON_KONTROL_2026-10-18.md` (Linux aarch64, GCC 11.4; C++
  tests 6/6; version5 bit-identity; NumPy float64 reference of the CNN ≤ 5e-15; kernel
  gains; determinism; stats/mmap invariance; runner resume; one 40-epoch Stage-0-type run
  per model: ReLU H=128 LR 0.03 collapses with hidden2 128/128 dead and input-independent
  logits, tanh+relu 96.9%).
- Seed sets made disjoint (DECISIONS 43): Stage 1 101–103, Stage 3 31–45; job files
  regenerated (4,554 / 450). `select_stage2.py --seeds`; `summarise.py --seeds` for the
  pre-registered confirmatory tables on seeds 11–30 (DECISIONS 49).
- Answers to the open questions recorded (DECISIONS 42–48); PLAN.md and README updated
  (all stages on AWS, stats archives outside git, AWS runbook).
- `version6/.cache/` removed from the index and ignored.
- `python3 tests/test_selection.py`: PASS.
