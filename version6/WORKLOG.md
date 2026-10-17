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
