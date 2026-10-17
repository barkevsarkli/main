# Prompt for Claude Code — version6: CNN extension of the mixed-activation study

> **For Barkev (not part of the prompt):** Start Claude Code in the repository root `article/` (the folder that contains `README.md`, `version1/` … `version5/`). Paste everything below the line. The agent commits and pushes the current state before it writes any code, builds and verifies version6, runs a short benchmark, writes an experiment plan with time estimates, and then **stops and waits for your approval**. It does not launch the large sweeps.

---

## 1. Role and goal

You are a careful research engineer working on a from-scratch C++ neural-network study (no ML libraries in the training code). The study asks whether **fixed per-neuron mixtures of activation functions** (e.g. half tanh, half ReLU in a layer) make training more stable — in particular, whether they prevent the *late training collapse* that ReLU networks show at high learning rates — and how they affect accuracy.

`version5/` contains the finished MLP study and an advisor-review TMLR draft. Your job is to create **`version6/`**, which extends the study to a small convolutional network, adds activation/gradient diagnostics, and prepares (but does not run) the new experiments.

Your work ends at a clearly defined stop point (Section 12). Correctness, reproducibility and an honest record of every decision matter more than speed.

## 2. Read before doing anything

Read these files completely and summarise them to yourself before you start:

- `README.md` (repository root)
- `version5/main.cpp`, `version5/net.h`, `version5/net.cpp`, `version5/rng_util.h`, `version5/dataset.h`, `version5/mnist_data.*`, `version5/cifar10_data.*`
- `version5/sweep/make_jobs.py`, `version5/sweep/pending.py`, `version5/sweep/run_chunk.sh`, `version5/sweep/select_lr.py`
- `version5/results_aws/tmlr_draft/main.tex` (the current article draft)
- `version5/results_aws/inceleme_2026-10-01/kacirilan_bulgular_inceleme.md` (a review of findings missed by the draft; Turkish, numbers are readable)
- `version5/CLAUDE_CODE_PROMPT.md` (the prompt that produced version5's sweep, for conventions)

Key facts from version5 that version6 must respect or improve on:

1. **Network and training protocol (version5):** `D → H → H → 10` MLP; per-neuron activation masks (`ratio`, `layout`); weights `U(−1/√fan_in, +1/√fan_in)`, biases 0; batch-size-1 SGD; 40 epochs; per-epoch shuffle; CE loss with linear outputs + softmax; element-wise clip of dL/dz to [−10, 10]; leaky-ReLU slope 0.1; fixed split with `SPLIT_SEED 12345` (MNIST: 45k/5k/10k slices of the 60k training file; CIFAR-10: 45k/5k from the training batches + official 10k test batch); best-validation checkpoint plus final epoch both tested; clean training accuracy on a fixed 10k training subset.
2. **Main result:** at MNIST H=128, LR=0.03, ReLU collapses in 30/30 runs (final val ≈10.6%), 50:50 tanh+ReLU in 0/30 (≈96.8%).
3. **Mechanism hints already in the logs:** collapsed ReLU runs end with training CE = 2.3081 ± 0.0003 ≈ ln 10, i.e. an input-independent output, consistent with a dead ReLU layer. Leaky-ReLU fails differently: a sudden one-epoch loss spike (+3 to +6) at a nearly fixed epoch across all seeds, then a plateau near 85% accuracy with CE > ln 10 (over-confident errors). Training loss starts rising ~12 epochs before validation collapses.
4. **Weaknesses the draft and the review identified:** (a) the 15% "failure" threshold misses partial collapses (e.g. tanh+ReLU loses >10 pp in 30/30 runs at LR=0.05); (b) LR tuning used *final* validation while testing used the *best-validation checkpoint*; (c) no activation, dead-unit or gradient measurements; (d) only one NaN check on one output; (e) source/binary provenance not recorded; (f) no ratio × learning-rate experiment; (g) concurrent CSV appends produced repeated header lines.
5. **Common random numbers:** in version5, `interleave`/`block` masks consume no RNG, so models with the same seed share initial weights and data order. Keep this property and make it explicit (Section 6.7).
6. **Portability problems in version5:** the data loaders use POSIX `mmap` (`sys/mman.h`, `unistd.h`) and the job runner is bash + `xargs`. **The new machine runs Windows** (Asus ROG Strix SCAR G18, Intel Core Ultra 9 275HX, 24 cores without SMT). version6 must build and run natively on Windows **and** on Linux/macOS (the heavy runs may later move to AWS Graviton Linux).

## 3. Non-negotiable rules

- **Do not modify anything in `version1/` … `version5/`.** They are the record for the article. version6 may copy files from them.
- All new code, scripts, results and docs go in `version6/`.
- **Commit and push the current state before writing any code** (Section 4).
- Training code stays dependency-free C++17. Python is allowed for job generation, running, analysis and tests. PyTorch is allowed **only** in the optional equivalence test, installed into `version6/.venv` (gitignored).
- Never use `-ffast-math`, `/fp:fast`, OpenMP inside a run, or any non-deterministic reduction. One process = one run = one thread. Parallelism comes from running many processes.
- Never consume RNG in a place that would make two models with the same seed diverge in initial weights or data order, unless the design explicitly requires it (random layout, Section 6.7).
- Never commit datasets, binaries, build directories, virtual environments or ZIP files. Never force-push or rewrite history.
- Do not launch Stage 1–3 sweeps. Only tests, smoke runs, the benchmark and (optionally) Stage 0 if the user later approves.
- If a requirement is ambiguous or seems wrong, do not guess silently: write the question and your provisional choice into `version6/DECISIONS.md`, and list it in your final report. If it blocks progress, stop and ask.
- Keep a dated log of what you did in `version6/WORKLOG.md`.

## 4. Phase 0 — Git snapshot before any code

Work from the repository root `article/`.

1. Inspect: `git status`, `git remote -v`, `git branch --show-current`, `git log -5 --oneline`, `git fetch`. If the local branch is behind or has diverged from its upstream, **stop and ask**.
2. Inventory everything untracked or modified. Print a table of files larger than 5 MB and every file larger than 50 MB (GitHub rejects >100 MB).
3. Update the root `.gitignore` (create it if missing). It must exclude at least:
   - compiled binaries and backups: `version*/main`, `version*/main.exe`, `*.prebuild.bak`, `*.o`, `*.obj`, `*.exe`, `*.pdb`, `build/`, `out/`, `_to_delete/`
   - archives: `*.zip`
   - LaTeX intermediates: `*.aux`, `*.log` **only inside `tmlr_draft/` and `paper/`** (sweep `.log` files are results and must stay tracked), `*.fls`, `*.fdb_latexmk`, `*.out` inside the draft folders, `*.blg`, `*.synctex.gz`, `results_aws/tmlr_draft/tmp/`
   - Python/OS clutter: `__pycache__/`, `*.pyc`, `.venv/`, `.DS_Store`, `.Rhistory`
   - datasets: `train-images-idx3-ubyte/`, `train-labels-idx1-ubyte/`, `t10k-*`, `CIFAR10/`, `cifar-10-batches-bin/` and any other raw data directory you find (check with `find` for files > 20 MB)
   - files unrelated to this article, e.g. `version5/results_aws/Claude outputs/` (PDFs from a different project). List anything else that looks unrelated and ask before committing it.
   **Keep tracked:** all source code, all results CSVs and `.log` files under `results*/`, `best_lr.json` files, analysis scripts and their outputs, Markdown notes, the TMLR draft sources (`.tex`, `.bib`, `.sty`, `.bst`, tables, figures) and its compiled `main.pdf`, figure PDFs/PNGs.
4. Scan everything that is about to be committed for secrets: AWS keys (`AKIA`, `aws_secret_access_key`), private keys (`BEGIN .* PRIVATE KEY`, `*.pem`, `id_rsa`), tokens, `.env` files, SSH host details inside `sweep/*.sh`. If you find any, **stop and ask**; do not commit them.
5. Show the final list of files to be added (`git status --short` after `git add`) and the total added size. Commit with the message:
   `Snapshot before version6 (CNN): version5 results, TMLR draft, review notes`
6. Create an annotated tag `v5-snapshot-2026-10` on that commit. Push the branch and the tag. Verify with `git ls-remote --heads --tags origin` that both exist remotely. Record the commit hash in `WORKLOG.md` later.
7. Create and push a working branch `v6-cnn` from that commit. All later commits go there. Push after every phase that passes its checks.

Only after step 7 succeeds may you create `version6/`.

## 5. Phase 1 — version6 scaffold and portability

### 5.1 Layout

```
version6/
  README.md            what version6 is, how to build, run, test (Windows + POSIX)
  PROMPT.md            a copy of this prompt
  DECISIONS.md         every design decision and open question, dated
  WORKLOG.md           chronological log of work, commit hashes, test outcomes
  PLAN.md              experiment plan with counts and time estimates (Phase 8)
  CMakeLists.txt
  src/                 C++ sources (training binary)
  tests/               C++ test executables + Python tests
  tools/               kernel dump, benchmark helpers
  sweep/               job generators, runner, pending detection, LR selection
  analysis/            loaders, summaries, figures
  kernels/             generated kernel banks (CSV + JSON gains + PNG gallery)
  results/             one subfolder per stage; benchmark results
```

Copy the version5 sources you need into `version6/src/` as the starting point and commit that copy unmodified first (`version6: import version5 sources unchanged`), so every later change is visible in the diff.

### 5.2 Portability

- Replace the POSIX `mmap` loaders with a small platform layer: `CreateFileMapping`/`MapViewOfFile` on Windows (`_WIN32`), `mmap` on POSIX. Optionally provide a plain `std::ifstream` fallback (`--no-mmap`). All three must yield **identical normalised floats**; add a test that checks this over the full MNIST training file and all CIFAR-10 batches.
- Open files in binary mode everywhere. Use `std::filesystem` for paths. No `/tmp`, no hard-coded separators.
- Data location: the binary takes `--data <dir>` as in version5 (`<dir>/train-images-idx3-ubyte/train-images-idx3-ubyte`, `<dir>/train-labels-idx1-ubyte/train-labels-idx1-ubyte`, `<dir>/CIFAR10/...` — confirm exact paths from the version5 loaders). Default `--data ..` relative to `version6/`. If the datasets are missing on this machine, stop and tell the user exactly which files to place where.
- Build with CMake (≥ 3.20). Support MSVC or clang-cl on Windows and GCC/Clang on Linux/macOS. Release flags:
  - GCC/Clang: `-O3 -march=native -ffp-contract=off -fno-fast-math`
  - MSVC/clang-cl: `/O2 /arch:AVX2 /fp:precise` (no `/fp:fast`, no `/fp:contract`)
  - Provide `build.ps1` and `build.sh` wrappers.
- Embed provenance at build time (generated header): git commit hash, dirty flag, compiler id/version, build flags, platform. Every result row records them.

### 5.3 Job runner (replaces bash/xargs)

Write `sweep/run_jobs.py` (Python ≥ 3.10, standard library only):

- Input: a job file (one JSON object per line describing all CLI arguments) and `--workers N`.
- Detects finished runs via a **run key** that includes *every* argument that affects results (Section 6.10), so re-running continues where it stopped.
- Each subprocess writes its result row and diagnostics to **its own** files (`results/<stage>/rows/<run_id>.csv`, `results/<stage>/stats/<run_id>.csv`). A separate `sweep/merge.py` concatenates them into one CSV per stage with exactly one header. No concurrent appends to a shared file.
- Prints a progress line after every completed job and a summary every 5 minutes: done/remaining, failures, elapsed, ETA from the observed throughput.
- Keeps the machine awake: on Windows call `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` via `ctypes`; on macOS wrap with `caffeinate -i` if available; on Linux try `systemd-inhibit` if available.
- Handles Ctrl-C cleanly (kills children, leaves no half-written rows).
- Logs failed jobs with their stderr tail to `results/<stage>/failures.log`.

## 6. Phase 2 — the network

### 6.1 Architectures (`--arch`)

- `mlp2`: the version5 network `D → H → H → 10` (needed for regression tests and the Stage 0 mechanism check).
- `mlp1`: `D → H → 10` (no conv). The CNN has one hidden layer, so this is its depth-matched baseline.
- `cnn`: **input → Conv2D → activation (per-channel mask) → MaxPool → flatten → Dense H (per-neuron mask) → Dense 10 (linear) → softmax CE.** Exactly one convolutional layer and one hidden dense layer.

### 6.2 Convolution layer

- Input tensor in CHW order: MNIST 1×28×28, CIFAR-10 3×32×32. Verify the order produced by the loaders (CIFAR binary stores the R plane, then G, then B).
- Kernel size `--conv-k K`, K ∈ {3, 5, 7, 9}; stride 1; "same" zero padding (`(K−1)/2`); `--conv-channels C` (default **8**, fixed for all planned experiments); one bias per output channel.
- Full channel mixing (`C × C_in × K × K` weights) for learned modes.
- MaxPool `--pool P` with P ∈ {1, 2, 4}, default 2 (stride = P). MNIST → 8×14×14 = 1,568 features; CIFAR-10 → 8×16×16 = 2,048. Backward sends the gradient to the arg-max position; break ties by the first element in row-major order (deterministic).
- Flatten in CHW order.

### 6.3 Kernel modes (`--kernel-mode`)

- `random` — learned conv, random init `U(−1/√(C_in·K²), +1/√(C_in·K²))`, bias 0. This is the standard-CNN baseline.
- `bank_init` — learned conv whose initial weights are a fixed kernel bank (Section 7), bias 0; all conv weights and biases train.
- `frozen` — conv weights fixed to a kernel bank, bias fixed at 0; nothing in the conv layer trains. Skip the conv/pool backward entirely in this mode (the hidden layer does not need to compute input gradients).

`--kernel-family {gauss, sharpen, sobel, laplacian, gabor, random_frozen}` selects the bank for `bank_init`/`frozen`.

**RNG alignment:** in every mode, first draw the random conv weights from the init stream exactly as `random` would, then overwrite them for `bank_init`/`frozen`. This keeps the hidden- and output-layer initial weights identical across kernel modes for the same seed. Add a test for it (Section 9).

### 6.4 Activation masks

- Separate flags per layer: `--conv-act1 --conv-act2 --conv-ratio` (mask over the C channels) and `--hid-act1 --hid-act2 --hid-ratio` (mask over the H neurons), plus `--layout {interleave, block, random}` as in version5. Convenience flags `--act1 --act2 --ratio` set both layers.
- `--conv-mask-offset {0,1}`: 1 uses the complementary interleave pattern (swaps which channels get act1/act2). For fixed banks, channel index = filter identity, so this control separates "which activation" from "which filter".
- Activations: relu, leaky_relu (slope 0.1), tanh, sigmoid, linear (output only) — same definitions and derivative formulas as version5.
- Named model configurations for the job generator:
  - `relu`, `leaky_relu`, `tanh`, `sigmoid` — homogeneous in both layers;
  - `tanh+relu` — conv and hidden mixed, ratio 0.5 (main hybrid);
  - `tanh+relu@conv` — conv mixed 0.5, hidden all ReLU (control);
  - `tanh+relu@hidden` — conv all ReLU, hidden mixed 0.5 (control);
  - `tanh+relu@ρ` — both layers mixed with ratio ρ ∈ {0.25, 0.5, 0.75} (ratio is the share of the second activation, ReLU). With C = 8, ρ = 0.25/0.5/0.75 gives 2/4/6 ReLU channels.

### 6.5 Initialisation of dense layers

Unchanged from version5: `U(−1/√fan_in, +1/√fan_in)`, bias 0. Do not add He or other schemes in this phase (record any idea in `DECISIONS.md`).

### 6.6 Training

Identical semantics to version5 unless stated:

- batch-size-1 SGD, `--epochs 40`, fixed LR, CE loss, per-epoch shuffle of the training indices;
- element-wise clip of dL/dz to [−10, 10] at every pre-activation, **including every conv output position**;
- dense layers: compute input gradients with the pre-update weights, then update in the same pass (version5 behaviour);
- conv layer: the kernel is shared across positions, so **accumulate dW and db over all positions first, then apply one update** per sample;
- best-validation snapshot of all layers (including conv) for `test_acc_best`, plus final-epoch test;
- clean training accuracy on the same fixed 10k training subset as version5;
- **NaN/Inf check on every logit and on the loss at every step** (version5 checked only `o[0]`). On failure set `status=nan`, record the epoch, stop.
- New per-epoch outputs: **validation CE loss**, and `epochs_completed`.
- Optional `--abort-on-collapse N` (default **off**): stop if, for N consecutive epochs, mean training loss is within 0.01 of ln 10 and validation accuracy is below 15%; set `status=collapsed_abort`. Implement and test it, but do not enable it in any plan without the user's approval.

### 6.7 Random-number streams (`--rng-mode`)

- `legacy`: one `std::mt19937(seed)` used exactly as version5 (only for `mlp2` regression tests).
- `split` (default for all new runs): derive three independent 32-bit seeds from `(seed, stream_id)` with SplitMix64 (stream ids: 1 = initial weights, 2 = masks, 3 = data order) and seed three `std::mt19937`. Document the exact formula in `DECISIONS.md`.
- Use only raw `mt19937` output plus the project's own `detu::bounded/shuffle`. **Do not use `std::uniform_real_distribution` or `std::normal_distribution` for anything new**: their output is not specified across standard libraries. Implement uniform floats and Box–Muller normals from raw 32-bit draws, document them, and test them. (Note in `DECISIONS.md` that version5 used `std::uniform_real_distribution` for weights; keep that call only in `legacy` mode.)
- Result: with `split`, models that share a seed share initial dense weights, conv random weights and data order, even under `random` layout.

### 6.8 Performance

- Implement a plain, readable reference convolution (forward + backward) in `tests/` and an optimised production version in `src/` (loop order and memory layout tuned for cache; still single-threaded, fixed summation order). The production outputs must match the reference within 1e-6 relative error per element on random inputs; a run must be bit-reproducible with itself.
- Profile one CIFAR H=128 K=9 epoch and record seconds per epoch for each layer in `WORKLOG.md`.

### 6.9 CLI

Keep version5 flags where they still make sense and add the new ones: `--arch --conv-k --conv-channels --pool --kernel-mode --kernel-family --kernel-seed --conv-act1 --conv-act2 --conv-ratio --conv-mask-offset --hid-act1 --hid-act2 --hid-ratio --layout --rng-mode --stats-out --stats-every --abort-on-collapse --dump-init <path> --run-id <id> --no-mmap`. Unknown flags are errors. `--help` prints all flags with defaults.

### 6.10 Result row (CSV, one row per run, written to its own file)

Columns, at minimum:

- identity: `run_id, stage, arch, dataset, seed, lr, epochs, loss, rng_mode, conv_k, conv_channels, pool, kernel_mode, kernel_family, kernel_seed, conv_act1, conv_act2, conv_ratio, conv_mask_offset, hid_act1, hid_act2, hid_ratio, layout, hidden`
- outcome: `status, epochs_completed, seconds, train_acc, val_acc, val_f1, test_acc, test_f1, best_val_acc, best_val_epoch, test_acc_best, test_f1_best, train_acc_clean`
- curves (pipe-separated): `epoch_loss, epoch_train_acc, epoch_train_acc_clean, epoch_val_acc, epoch_val_f1, epoch_val_loss`
- confusion matrices: `test_confusion, test_confusion_best` (100 pipe-separated counts each)
- provenance: `git_hash, git_dirty, compiler, build_flags, platform, n_params_trainable, n_params_total, init_hash, kernel_hash, split_hash` (FNV-1a 64 over the initial weights of all layers, over the conv bank actually used, and over the train/val/test index lists).

The run key used for resumption is the tuple of all identity columns except `run_id` and `stage`.

## 7. Phase 3 — fixed kernel banks

Implement `src/kernels.{h,cpp}` and `tools/kernel_dump`. Notation: K odd, radius r = (K−1)/2, coordinates u, v ∈ {−r … r}, base scale σ_K = K/6. Every bank has exactly **8 filters**, so all banks feed the same 1,568 (MNIST) or 2,048 (CIFAR-10) features into the hidden layer.

| Family | Purpose | The 8 filters |
|---|---|---|
| `gauss` | smoothing | isotropic Gaussians, σ_j = σ_K·(0.5 + 0.25 j), j = 0…7; each normalised to sum 1 before gain |
| `sharpen` | sharpening | unsharp masks δ + α(δ − G_σ), α ∈ {0.5, 1, 2, 4} × σ ∈ {σ_K, 1.5 σ_K} (G_σ normalised to sum 1) |
| `sobel` | first-derivative edges | generalised Sobel: Gx = s ⊗ d with s = binomial coefficients of length K and d = (binomial of length K−2) ∗ [−1, 0, 1]; Gy = Gxᵀ; 8 steered orientations θ_j = 2πj/8, G_θ = cos θ·Gx + sin θ·Gy. At K=3, θ=0 must equal the classic Sobel kernel up to sign/scale. |
| `laplacian` | second-derivative edges | Laplacian of Gaussian at σ ∈ σ_K·{0.5, 0.75, 1.0, 1.25}, each made exactly zero-sum (subtract mean), in both signs (on-centre and off-centre) |
| `gabor` | oriented texture/edges | g = exp(−(u′² + γ²v′²)/(2σ²))·cos(2πu′/λ + ψ), u′ = u cos θ + v sin θ, v′ = −u sin θ + v cos θ; θ ∈ {0, π/4, π/2, 3π/4}, ψ ∈ {0, π/2}; λ = max(2, (K+1)/2), σ = 0.56λ, γ = 0.5; subtract the mean so each filter is zero-mean |
| `random_frozen` | control: "any fixed filter" | i.i.d. standard normals from a dedicated `mt19937(kernel_seed)` (default `kernel_seed = 20261018`) via your portable Box–Muller; the same bank for every run seed |

Scaling (important — activation statistics depend on the input scale):

1. L2-normalise every filter.
2. **CIFAR-10:** apply the same 2-D filter to all three colour channels, `W[c, ch, :, :] = f_c / √3`. Note in `DECISIONS.md` that frozen banks therefore see colour only through the channel sum, while learned modes can mix colours.
3. Per-filter gain: choose a scalar g_c for each filter so that its pre-activation standard deviation, measured on a **calibration set = the first 1,000 indices of the training split**, equals τ_K. τ_K is the mean, over seeds 1–5 and over the 8 channels, of the pre-activation std of the `random` mode conv with the same K, dataset and init. Gains are computed once per (dataset, family, K), written to `kernels/<dataset>_<family>_k<K>.json`, and loaded by the binary. Never compute gains from validation or test images.
4. `kernel_dump` writes every bank to `kernels/*.csv` and a PNG gallery (`kernels/gallery_<dataset>.png`, via a Python script) so the filters can be inspected visually.

Interpretation note for later analysis (do not "fix" it): smoothing filters on [0, 1] images give mostly positive outputs, so ReLU barely rectifies them while tanh compresses them.

## 8. Phase 4 — diagnostics ("activation distribution")

Enabled with `--stats-out <path>` and `--stats-every E` (default 1). Diagnostics must **not change training in any way**: no RNG use, no change to any training buffer, forward-only passes after the epoch's updates. A test must show identical result rows with stats on and off.

Measured at epoch 0 (initialisation) and after every E epochs, on a **probe set = the first 1,000 indices of the validation split**:

For each layer (conv post-activation measured **before** pooling; hidden layer) and separately for each activation group in that layer's mask (e.g. `tanh` channels vs `relu` channels) plus `all`:

- pre-activation z: mean, std, p01, p50, p99, max |z|
- post-activation h: mean, std
- ReLU / leaky-ReLU units: fraction of elements with z < 0; **dead units** (z ≤ 0 for every probe image and, for conv, every position); nearly-dead units (active on < 1% of probe elements). A conv "unit" is a channel.
- tanh units: fraction of elements with |h| > 0.99; units saturated on > 95% of probe elements
- sigmoid units: fraction of elements with h < 0.01 or h > 0.99
- histogram of z: 48 equal bins on [−12, 12] plus underflow and overflow counts

Output layer: mean and std of logits; **between-input std of logits** (std across probe images, averaged over classes — near 0 means input-independent output); number of distinct predicted classes; largest predicted-class share; mean max-softmax probability; mean prediction entropy.

Training-time statistics accumulated during the epoch's SGD steps (no extra passes): per layer, mean |dL/dz|, max |dL/dz|, fraction of dL/dz elements that hit the clip; weight Frobenius norm at epoch end; ‖ΔW‖/‖W‖ over the epoch.

Format: long CSV, one row per (run_id, epoch, layer, group, metric) with a `value` column; histograms as one row with pipe-separated counts. Keep a run's stats file below ~200 KB; report the measured size per run in the benchmark.

## 9. Phase 5 — verification (all must pass before the benchmark)

Write each as an executable or `pytest` test, run them, and record the outcome in `WORKLOG.md` and in the final report table.

1. **Loader equivalence:** mmap (both platforms) vs `--no-mmap` give identical floats and labels for all MNIST and CIFAR-10 images.
2. **Version5 regression (`--arch mlp2 --rng-mode legacy`):**
   - If a POSIX toolchain is available (macOS, Linux, WSL, or AWS): build version5 unchanged and version6 with the same compiler and flags; for MNIST H=12 relu, MNIST H=12 tanh+relu, CIFAR H=12 tanh+relu (3 epochs, 2 seeds each) every metric column and curve must be **bit-identical** (ignore `seconds`).
   - On Windows only: run the same configurations and compare with the matching rows already stored in `version5/results_aws/` (AWS/ARM). Bit identity is not expected (different libm and platform); report the maximum absolute difference of `epoch_val_acc` over the first 3 epochs and treat it as a smoke check. State clearly which variant you ran.
3. **Gradient check:** finite differences (central, in double precision via a templated or duplicated test path, clip disabled) against analytic gradients for conv weights and biases, hidden and output layers; every activation; mixed masks in both layers; pool sizes 1/2/4; K = 3 and 9; both datasets' input shapes. Relative error < 1e-4 in double precision. Also check that the production update equals −lr × gradient.
4. **Reference convolution:** optimised vs reference forward and backward, relative error ≤ 1e-6.
5. **PyTorch equivalence (optional, recommended):** if PyTorch (CPU) can be installed in `version6/.venv`, rebuild the same `cnn` in float64 PyTorch, load identical initial weights from `--dump-init`, run 200 batch-size-1 SGD steps on the same images in the same order (clip at 10 implemented identically), and compare all weights after training (relative error < 1e-5 against a float64 build of our path, or explain the tolerance used). Cover `random`, `bank_init` and `frozen`.
6. **Kernel bank checks:** shapes; zero-sum where required (sobel, laplacian, gabor); K=3 Sobel matches the classic kernel up to sign/scale; Gaussians are non-negative; gains are finite and positive; the PNG gallery exists.
7. **RNG alignment:** for the same seed, initial hidden/output weights are identical across `random`, `bank_init` and `frozen`, and across all activation configurations (compare `init_hash` of those layers). Data order is identical across models (log the first 20 indices of epoch 1 in a debug mode and compare).
8. **Frozen mode:** conv weights and biases are bit-identical before and after training.
9. **Determinism:** the same job run twice gives identical rows (except `seconds`), and results do not depend on `--workers`.
10. **Diagnostics invariance:** rows identical with `--stats-out` on and off. Hand-check dead-unit counting on a tiny network where the answer is known.
11. **Collapse-abort logic:** a synthetic test triggers and does not trigger it correctly.
12. **Smoke runs:** 2 epochs of every (arch, kernel mode, kernel family, K) combination on MNIST H=64 and CIFAR H=64 with tanh+relu; all statuses `ok`, accuracies above chance after 2 epochs for learned modes.

## 10. Phase 6 — experiment plan as job generators (generate, do not run)

Write `sweep/make_jobs.py` with one sub-command per stage, emitting JSON-lines job files under `results/<stage>/jobs.jsonl`, plus `sweep/select_stage2.py` and `sweep/select_lr.py`. Selection scripts use **validation metrics only, never test metrics**. All runs use `--rng-mode split`, 40 epochs, CE, and stats enabled.

Seed conventions: evaluation seeds 1–30; if a stage needs separate tuning seeds, use 101–106 as in version5.

### Stage 0 — MLP mechanism check (cheap, high value for the current paper)

`mlp2`, MNIST, H ∈ {64, 128}, LR ∈ {0.02, 0.03, 0.05}, models {relu, leaky_relu, tanh, tanh+relu}, seeds 1–10, stats every epoch. Purpose: measure directly whether ReLU collapse coincides with dead hidden units and input-independent outputs, and what happens in the leaky-ReLU loss spike. (`split` RNG means these runs are not seed-for-seed replicas of version5; state that in the plan.)

### Stage 1 — kernel screen (H = 128, both datasets)

- Kernel configurations (48): `random` × K ∈ {3,5,7,9} (4); each designed family {gauss, sharpen, sobel, laplacian, gabor} × K ∈ {3,5,7,9} × mode ∈ {frozen, bank_init} (40); `random_frozen` × K ∈ {3,5,7,9} × frozen (4).
- Models: relu, tanh, tanh+relu.
- LR grids (5 points, roughly ×3 apart, chosen to cover each dataset's optimum and its collapse region in version5): MNIST {0.001, 0.003, 0.01, 0.03, 0.1}; CIFAR-10 {0.0003, 0.001, 0.003, 0.01, 0.03}. All three models run on the full grid.
- Seeds 1–3.
- Mask control: for every frozen bank, `tanh+relu` with `--conv-mask-offset 1` at the middle grid LR (MNIST 0.01, CIFAR-10 0.003), seeds 1–3.
- Also `mlp1` with the same models, LRs and seeds as a no-conv reference.

**Pre-registered selection for Stage 2** (implement in `select_stage2.py`, document in `PLAN.md`), per dataset, using validation only:

1. **Standard CNN:** `random` mode at the K with the highest mean best-validation accuracy for ReLU at its best grid LR (ties → smaller K).
2. **Most informative bank:** among `frozen` and `bank_init` configurations, the one with the largest mean (tanh+relu − relu) difference in **final** validation accuracy at that configuration's **Stage 1 stress LR**, provided ReLU's mean best-validation accuracy for that configuration is within 2 pp of configuration 1's. Ties → smaller K, then the family order in Section 7.
   The Stage 1 stress LR of a configuration is the smallest grid LR at which ReLU's mean final validation accuracy (seeds 1–3) is below 50% of its best grid mean for that configuration; if there is none, use the largest grid LR. Do not use a fixed LR such as 0.03: the collapse region of the CNN is unknown and may differ between kernel configurations.
3. If both datasets select different configurations, run each dataset with its own selection.

### Stage 2 — main CNN study

- Architectures: the two selected CNN configurations + `mlp1`.
- Widths: H ∈ {64, 128, 256} for both datasets and for all architectures in this stage (including `mlp1`). H = 512 is deliberately excluded from version6 to save compute; the article's H = 512 evidence comes from version5's MLP. State this in `PLAN.md`.
- Models: relu, leaky_relu, tanh, sigmoid, tanh+relu; plus the controls tanh+relu@conv and tanh+relu@hidden for CNN configuration 2 at H ∈ {128, 256}.
- LR grids: MNIST {0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1}; CIFAR-10 {0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05}.
- Sweep: seeds 1–10 on the full grid.
- LR selection (pre-registered): per (arch, dataset, H, model) choose the LR with the highest **mean best-validation accuracy** over seeds 1–10 (checkpoint-aligned, fixing version5's tuning mismatch). Ties → lower LR.
- **Stress LR** (pre-registered): per (arch, dataset, H), the smallest grid LR at which ReLU's mean final validation accuracy over seeds 1–10 is below 50% of its best grid mean. If there is none, use the largest grid LR.
- Extension to seeds 1–30 for every model at three LRs: the selected LR, the stress LR, and the grid LR just below the stress LR.

### Stage 3 — activation ratio × learning rate

- CNN configuration 2 (from Stage 1) at H = 128, both datasets.
- Models: tanh+relu@ρ for ρ ∈ {0.25, 0.5, 0.75}, plus relu and tanh as references.
- LRs: Stage 2's stress LR for that setting, plus the grid LRs just below and just above it.
- Seeds 1–15.

Stage 2 and 3 job files can only be finalised after Stages 1 and 2 have results. For now, generate them with placeholders (e.g. `<SELECTED_K>`) and compute their counts and time estimates under the assumption "K = 5, mode = bank_init" for configuration 2 and "K = 3, random" for configuration 1. State the assumption.

### Outcome measures the analysis must support (write them into `PLAN.md` now)

Per run: final and best validation accuracy, peak-to-final drop, `final < 15%`, `drop > 10 pp`, `ever below 15%` (any epoch), first collapse epoch, final training loss vs ln 10, between-input logit std, dead-unit fractions over time, checkpoint and final test accuracy and macro-F1. Paired comparisons by seed (paired bootstrap, 10,000 resamples, seed 20261018) with Holm correction inside pre-declared families. Report learning rates also **relative to each model's own best LR**, because version5 showed that the hybrid's LR curve is shifted as well as widened.

## 11. Phase 7 — benchmark and budget

Run only after Section 9 passes.

1. Worker count: on one mid-size job (CIFAR H=128, `cnn`, K=5, `random`, 1 epoch) measure total throughput (jobs/hour) with 8, 16 and 24 concurrent workers. The 275HX has 8 performance and 16 efficiency cores; pick the worker count with the best throughput and record per-worker time spread.
2. Per-configuration cost: with the chosen worker count, all workers running copies of the same job, measure seconds per epoch for every (dataset, H) used in the plan × {`mlp1`, `mlp2`, `cnn` K ∈ {3,5,7,9} in `random` and in `frozen`}. 1 epoch each is enough. Cap the whole benchmark at 2 hours; if it would exceed that, measure K ∈ {3, 9} and interpolate in K² for the conv part, and say so.
3. Note the conditions: AC power, Windows power mode, ambient notes, any thermal throttling you can detect (per-epoch time drift across consecutive epochs).
4. Compute, per stage, the number of runs and the estimated wall-clock time on this machine (`runs × epochs × seconds_per_epoch ÷ workers`, plus 5% overhead). Also estimate for an AWS host with N cores using version5's measured AWS speed as a reference (state the method).
5. If a stage exceeds these targets on this machine — Stage 0 ≤ 4 h, Stage 1 ≤ 72 h, Stage 2 ≤ 120 h, Stage 3 ≤ 24 h — propose a reduced variant **in addition to** the full one. Default reduction order: Stage 1 — seeds 3 → 2, then drop K = 7, then drop the lowest grid LR (MNIST 0.001, CIFAR-10 0.0003); Stage 2 — sweep seeds 10 → 6, then extension 30 → 20 seeds, then drop the @conv/@hidden controls at H=256, then drop sigmoid from CIFAR H=256. Never drop relu, tanh or tanh+relu, the stress LR, or the `random` CNN baseline.

## 12. Phase 8 — analysis scaffold, plan, report, STOP

1. `analysis/load.py` (reads merged stage CSVs and stats files), `analysis/summarise.py` (tables of the outcome measures above), `analysis/figures.py` (LR curves, failure maps with both the 15% and the 10 pp rules, dead-unit fraction and logit-std over epochs next to validation accuracy, kernel-family × K heatmaps). Test them on the smoke and benchmark outputs.
2. Write `PLAN.md`: stages, exact factor levels, run counts, full and reduced time estimates, pre-registered selection rules, outcome measures, and what each stage can and cannot show.
3. Update `README.md`, `DECISIONS.md` and `WORKLOG.md`. Commit and push.
4. **Stop and report** to the user with:
   - commit hashes of the snapshot, the unchanged import and the final commit; tag and branch names;
   - a pass/fail table of all tests in Section 9 (with which regression variant ran);
   - the benchmark table (seconds per epoch per configuration, chosen worker count);
   - per-stage run counts and time estimates (full and reduced);
   - the list of open questions from `DECISIONS.md`;
   - what you need from the user to continue (e.g. approval of a stage variant, the `--abort-on-collapse` decision, where to run the heavy stages).

Do not start Stage 0–3 until the user explicitly approves.
