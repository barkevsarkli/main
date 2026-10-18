# version6 — decisions and open questions

Every design choice that PROMPT.md leaves open, every place where version6 departs from
version5, and every question for Barkev. Dated; **OPEN** marks items that need an answer.

## Environment and repository (2026-10-18)

1. **Machine. OPEN (partly answered).** PROMPT.md assumes a Windows laptop (Core Ultra 9
   275HX). This work ran on an Apple M1 (4 performance + 4 efficiency cores, 8 GB, macOS
   26). Barkev's answer on 2026-10-18: *the heavy runs will go to AWS again; write the code
   for that.* Consequences:
   - All tests ran on macOS/arm64 with Apple clang 21. The Windows code paths
     (`MapViewOfFile` loader, `SetThreadExecutionState`, MSVC flags, `build.ps1`) are
     written but **not compiled or run** here.
   - The benchmark (`results/bench/`) is a provisional M1 measurement. AWS times are
     extrapolated (item 28) and should be re-measured on the instance (20 minutes).
2. **Datasets stay tracked in git** (Barkev, 2026-10-18). `CIFAR10/*.bin` and the MNIST
   training files were committed before this work. `.gitignore` only blocks *new* raw data
   (`t10k-*`, `cifar-10-batches-bin/`). History was not rewritten.
3. **`version5/results_aws/` is now tracked.** It was fully ignored before, so the TMLR
   draft, the review notes and every AWS result CSV were missing from git. The snapshot
   commit tracks it except for `*.zip`, `tmlr_draft/tmp/` (26 MB of render PNGs), the LaTeX
   intermediates, `Claude outputs/` (PDFs from another project), and
   `financial_analyst_colab/`, which Barkev confirmed is unrelated (2026-10-18).
4. **The prompt file** `version5/CLAUDE_CODE_PROMPT_version6_CNN.md` is untracked inside
   `version5/`. Committing it there would modify version5, so it was copied to
   `version6/PROMPT.md` and the original was left alone. Barkev may delete it.

## Numerics

5. **FMA contraction is off (`-ffp-contract=off`), and this matters for version5
   comparisons.** Building unchanged version5 with plain `-O3` (Apple clang's default
   contraction is `on`) reproduces `version5/reference/determinism_mac.csv` exactly. Building
   it with `-ffp-contract=off` reproduces version6 `--rng-mode legacy` exactly, but differs
   from that reference in the ReLU and sigmoid rows (the hybrid rows happen to match). So
   the version5 Mac reference was produced *with* FMA contraction. The AWS build used `g++
   -O3 -mcpu=native -std=c++17`. GCC documents `-ffp-contract=off` as the default only for
   ISO C, so the version5 AWS runs most likely used `fast` contraction. version6 forbids
   contraction everywhere, as PROMPT.md requires. Test 2 therefore compares version5
   against version6 built with the same flags. **Open:** say in the article that version5's
   AWS numbers came from a contracted build. The source code is unchanged; only the last
   bits of some runs differ.
6. **Dense summation order.** `--rng-mode legacy` keeps version5's sequential order
   (`z = b; z += x_j w_j` for j = 0, 1, …). All other runs use a fixed 16-lane blocked dot
   product: partial sums over j mod 16, a balanced tree, then the tail in order. The order
   is platform independent and lets the compiler vectorise without reassociating. It gave
   1.7× on the dense forward. Choosing this before any version6 result existed costs
   nothing.
7. **Conv summation order.** Forward: bias, then (ci, ky, kx) in lexicographic order for
   each output. This is bit-identical to the plain reference loops (checked in
   `test_conv_reference`). Backward: dW sums over pooled cells in row-major order at each
   cell's arg-max position. This differs from the reference's row-major order over all
   positions, so the check there uses a tolerance. Zero gradients are skipped, which is
   bit-safe: adding ±0 changes nothing.
8. **Learning rate is a float32**, as in version5 (`std::stof`). For example, `--lr 0.01`
   trains with 0.0099999998. The PyTorch test applies the same rounding. Without it, the
   two implementations differ by a constant 2.2e-8.
9. **MSE gradient scale (version5 behaviour, kept).** With `--loss mse` the logged loss is
   the *mean* squared error over the 10 outputs, but the back-propagated derivative is
   `2(o − y)`, the gradient of the *sum*. That is 10× the gradient of the logged loss. The
   gradient check differentiates the sum. The review notes already point this out.
10. **Maximum pooling acts on h = act(z), as specified.** Ties go to the first element in
    row-major order (strict `>`). Pooling on z would be cheaper for monotone activations.
    It was not used, because it breaks ties differently when float tanh/sigmoid saturate.
11. **Non-finite runs.** The check covers every logit and the loss at every step. On
    failure: `status=nan`, `status_epoch` = the epoch in progress, training stops, and the
    partial epoch is **not** evaluated or recorded. version5 still evaluated and recorded
    that epoch. The final-test columns are evaluated on the non-finite weights, and status
    flags them as unusable.

## Random numbers

12. **Split streams.** `stream_seed(seed, id) = high 32 bits of splitmix64((uint32(seed)
    << 32) | id)`, where splitmix64 adds 0x9E3779B97F4A7C15 and then applies the standard
    finaliser. Ids: 1 = initial weights, 2 = masks, 3 = data order. Each id seeds its own
    `std::mt19937`. `test_rng` checks the SplitMix64 reference values and that the 30,000
    seeds for 10,000 run seeds × 3 streams contain no collisions.
13. **Uniform floats:** `u = (draw >> 8) · 2⁻²⁴` (exact), and `w = limit · (2u − 1)`, which
    has exactly one rounding. **Normals:** Box–Muller in double precision from two draws,
    with `u1 = ((draw >> 8) + 1) · 2⁻²⁴` in (0, 1]. Both values of each pair are used.
    These are used only to build the `random_frozen` bank, which is stored (item 16).
    `std::uniform_real_distribution` survives only in `--rng-mode legacy`, because
    version5's weights came from it.
14. **Init order:** the conv weights come first in every kernel mode, then the dense layers
    in order. `bank_init`/`frozen` then overwrite the conv weights, so the dense layers'
    initial weights are identical across kernel modes and activation configurations
    (test 7: 20 configurations share one `init_hash_dense`).
15. Stage 0 uses `split`, so its runs are **not** seed-for-seed replicas of version5.

## Kernel banks

16. **The trainer reads the stored banks; it never rebuilds them.** `kernel_dump` builds the
    filters (in double, then rounded to float), calibrates the gains, and writes
    `kernels/<dataset>_<family>_k<K>.json`. The file holds the filters as `%.9g`, which
    round-trips float exactly, plus the gains, τ and an FNV hash that is checked at load
    time. libm differences between platforms therefore cannot change a run's kernels. A
    `random_frozen` run's `--kernel-seed` must match the seed stored in its file.
17. **Gabor at K = 3. OPEN.** With the specified λ = max(2, (K+1)/2) = 2, the odd-phase
    filters at θ = 0 and π/2 are sin(πu) = 0 at every integer u, so they are all zeros and
    cannot be normalised. Provisional choice: λ = 2.5 at K = 3 only. K = 5, 7 and 9 follow
    the specification. Alternatives would be other phases at K = 3, or dropping Gabor K = 3.
18. **Laplacian at K = 3 (observation, no change).** σ = σ_K·{0.5, 0.75} = 0.25 and 0.375.
    At these scales the neighbour weights are ~e⁻⁸, so after mean subtraction the first two
    filters are close to "delta minus mean". The gallery shows this.
19. **Bank order within each family:** sharpen j = 2·α-index + σ-index; laplacian 0–3
    on-centre (positive centre) at increasing σ, then 4–7 the same filters negated; gabor
    j = 2·θ-index + ψ-index; sobel θ_j = 2πj/8. That gives 4 sign-flipped pairs, as
    specified.
20. **Gain calibration.** "Pre-activation std" is the population std, over the 1,000
    calibration images × all positions, of one channel's z (bias 0). The CIFAR-10 filters
    include the 1/√3. τ_K uses exactly the conv weights that a `split` run with seeds 1–5
    draws first. Smoothing banks get small gains because their outputs have large means and
    small spreads (PROMPT note: do not "fix" this).
21. **CIFAR-10 frozen banks see colour only through the channel sum** (same 2-D filter on R,
    G, B, scaled by 1/√3). Learned modes can mix colours.
22. **`--conv-mask-offset 1`** rotates the mask by one channel: m′[i] = m[(i+1) mod C]. At
    ratio 0.5 with interleave this is exactly the complementary pattern. At other ratios it
    keeps the counts. A homogeneous conv mask canonicalises the offset to 0.

## Run identity and outputs

23. **Canonical identity.** A mask is written (a, a, 0) when it is homogeneous, whichever
    way it was requested. Conv fields are blanked (`none`/0) for MLPs. `kernel_family` is
    `none` in random mode, and `kernel_seed` is 0 unless the family is `random_frozen`.
    `abort_on_collapse` affects results, so it is part of the identity. The run key covers
    every identity column; `run_id` = FNV-1a 64 of the run key (16 hex digits). After every
    job, the runner checks that the row's identity columns equal the job's canonical
    identity.
24. **Columns beyond PROMPT.md 6.10:** `abort_on_collapse`, `status_epoch`,
    `init_hash_dense` (dense layers only, for test 7), `dtype`, and `epoch_seconds` (per
    epoch wall time, used for the benchmark and the drift check).
25. **Extra CLI flags:** `--kernel-dir`, `--out`, `--stage`, `--quiet`, `--help`, and the
    test hooks `--dtype float64`, `--max-steps`, `--dump-final`, `--debug-order`,
    `--no-clip`. Conv flags given with an MLP are errors, and so is `--kernel-family` in
    random mode.
26. **Diagnostics definitions:**
    - probe set = first 1,000 validation indices;
    - groups = one per activation present in the layer, plus `all` when the layer mixes;
    - percentiles are nearest-rank;
    - "active" means z > 0; a dead unit has z ≤ 0 on every probe element; leaky-ReLU units
      use the same definition;
    - a clip hit means |dL/dz| > 10 before the clip;
    - conv gradient means count every position, including the exact zeros at non-arg-max
      positions;
    - `dw_rel` = ‖W_end − W_start‖_F / ‖W_start‖_F over weights only (no biases);
    - epoch 0 records weight norms and probe statistics only.

    Measured size: about 4 KB per measurement point, roughly 170 KB for a 40-epoch run (see
    the benchmark).
27. **Abort-on-collapse** is implemented and tested, off by default, and unused by every plan.
    **OPEN:** enable it for any stage? It would save compute on collapsed runs, but it
    truncates their curves.

## Plan, selection, statistics

28. **AWS extrapolation.** The benchmark also runs version5's code path (`mlp2 --rng-mode
    legacy`, CIFAR H = 128) under the same load. version5 measured 19.3 s/epoch for that job
    on a c8g.48xlarge at P = 190, and the ratio between the two converts this machine's
    timings. This assumes the CNN speeds up between the machines by the same factor as the
    MLP.
29. **"Below 50% of its best grid mean"** (stress LR) is read as: below 50% of the best,
    over the grid, of ReLU's mean *final* validation accuracy for that configuration.
30. **Stage 1 tie-breaks.** After K and family order, frozen comes before bank_init. PROMPT
    did not order the modes.
31. **Holm families** = (dataset, metric, comparison) within a stage. Paired t-test p values
    with a paired bootstrap CI (10,000 resamples, seed 20261018).
32. **A selected LR on the grid edge is flagged, not fatal.** version5's `select_lr.py`
    refused to proceed in that case. Here the grids are pre-registered and fixed.
33. **Stage 3's ratio-0.5 model duplicates Stage 2 runs** (cfg2, H = 128, tanh+relu, at the
    same LRs and seeds 1–10). They are re-run in Stage 3's own directory for simplicity,
    about 20% of Stage 3. **OPEN:** reuse the Stage 2 rows instead?
34. **The Stage 2 extension** de-duplicates LRs when the selected LR coincides with the
    stress LR or the LR just below it. The estimates assume three distinct LRs, an upper
    bound.
35. **Not done, recorded as ideas:** He or activation-specific initialisation (PROMPT 6.5:
    not in this phase), and caching frozen-conv features to skip the conv forward (a memory
    trade-off: 0.4–0.5 GB per process).

## Tooling

36. **PyTorch** is used only by `tests/torch_equiv.py`. `version6/.venv` was created with
    `--system-site-packages`, reusing the torch 2.8.0 already installed on this Mac instead
    of downloading it again. pytest was installed into the venv. `.venv/` is gitignored.
37. Job files (`results/*/jobs*.jsonl`) are committed as the record of what was planned.
    Those with placeholders cannot be run (`run_jobs.py` refuses them).

## Added after the benchmark (2026-10-18)

38. **Cross-platform reproducibility, measured.** On the same CNN job (MNIST, H = 32, frozen
    gabor K = 5, tanh+relu, 1 epoch), three builds gave rows identical in every metric,
    curve and hash: Apple clang 21 + libc++ (macOS), GCC 16 + libstdc++ (macOS), and GCC
    13.3 + glibc (Ubuntu 24.04 aarch64, Docker; same OS and arch as the AWS AMI). The
    version5 MLP rows do differ between macOS and Linux in the tanh runs (libm `tanhf`).
    So pooling runs from different machines is safe only when the compiler/libm pairs have
    been compared. **Recommendation:** run every stage of a comparison on one platform, and
    record `platform` and `compiler` (both are in every row).
39. **Stats storage. OPEN.** About 180 KB per run means ≈ 0.8 GB (Stage 1) and ≈ 2.3 GB
    (Stage 2) of small CSV files. Options: commit them compressed (`.csv.gz` per stage),
    keep them in S3 and commit only the merged result rows, or record stats every 2
    epochs. Provisional: commit rows; archive stats separately.
40. **The laptop cannot meet the Stage 1/2 targets** even with every prescribed reduction
    (PLAN.md, Budget). On AWS no reduction is needed (~14 instance-hours in total).
    **OPEN:** confirm AWS for Stages 1–3, and whether Stage 0 runs on the laptop (1.5 h).
41. Benchmark noise: one cell (CIFAR-10 H = 128, K = 5) measured the frozen conv slower
    than the learned one (39.2 vs 34.2 s). That is impossible as a cost ordering (frozen
    skips work), so it shows the run-to-run spread on a laptop with mixed P/E cores under
    full load. The estimates use the measured values as they are.
