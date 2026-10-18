# version6 — mixed activations in a small CNN

version5 showed that fixed per-neuron mixtures of activation functions change how a
small MLP trained by batch-size-1 SGD fails at high learning rates. At MNIST, H = 128 and
LR = 0.03, ReLU collapsed in 30 of 30 runs and a 50:50 tanh+ReLU mix in none. version6:

* extends the study to a CNN with one conv block (`cnn`) and adds a depth-matched MLP
  (`mlp1`). The version5 network stays available as `mlp2`;
* adds fixed, gain-calibrated **kernel banks** (gauss, sharpen, sobel, laplacian, gabor,
  random_frozen) that the conv layer can be frozen to (`frozen`) or start from
  (`bank_init`);
* adds **diagnostics**: per-layer and per-activation-group pre-activation statistics, dead
  and saturated units, a z histogram, input-independence of the output, and gradient,
  clip and weight-change statistics;
* fixes the version5 weaknesses: independent RNG streams, a NaN/Inf check on every logit,
  provenance and hashes in every row, one result file per run (no concurrent CSV
  appends), and a checkpoint-aligned LR selection;
* defines Stages 0–3 as job generators, with a benchmark and time estimates. **Nothing
  of Stages 0–3 has been run.** See `PLAN.md`.

Files: `PROMPT.md` (the task), `DECISIONS.md` (every choice and open question),
`WORKLOG.md` (what was done, test outcomes, commits), `PLAN.md` (the experiment plan).

## Build

The training code is dependency-free C++17 built with CMake ≥ 3.20. Floating-point policy:
no fast-math, no FMA contraction, no OpenMP, one thread per process.

```bash
./build.sh          # Linux / macOS: GCC or Clang, -O3 -march=native (or -mcpu=native on aarch64)
./build.sh test     # ... and run the C++ tests
```
```powershell
.\build.ps1         # Windows: MSVC or clang-cl, /O2 /arch:AVX2 /fp:precise   (untested so far)
.\build.ps1 -Test
```

Binaries go to `version6/bin/` (`main`, `kernel_dump`, `profile_layers`, `test_*`). The
build embeds provenance (git commit, dirty flag, compiler, flags, platform), and every
result row records it.

## Data

`--data <dir>` (default `..`, i.e. the repository root when run from `version6/`) must hold
```
<dir>/train-images-idx3-ubyte/train-images-idx3-ubyte
<dir>/train-labels-idx1-ubyte/train-labels-idx1-ubyte
<dir>/CIFAR10/train_images/data_batch_{1..5}.bin
<dir>/CIFAR10/test_images/test_batch.bin
```
The split is version5's: SPLIT_SEED 12345. MNIST is 45k/5k/10k slices of the 60k
training file; CIFAR-10 is 45k/5k from the training batches plus the official test batch.
Files are memory-mapped (`mmap`, or `MapViewOfFile` on Windows); `--no-mmap` reads them
with `std::ifstream` instead. Both give identical floats (`test_loader`).

## Run one model

```bash
bin/main --help
bin/main --arch cnn --dataset cifar10 --hidden 128 --conv-k 5 --kernel-mode frozen \
         --kernel-family gabor --act1 tanh --act2 relu --ratio 0.5 --lr 0.003 --seed 1 \
         --epochs 40 --out row.csv --stats-out stats.csv
bin/main --arch mlp2 --rng-mode legacy --hid-act1 tanh --hid-act2 relu --hid-ratio 0.5 \
         --hidden 128 --lr 0.03 --seed 1 --epochs 40 --out row.csv     # version5, bit for bit
```
`--out` gets a header and one row, written atomically. The kernel banks are in
`kernels/`. Regenerate them with `bin/kernel_dump --data .. --out kernels` and
`python3 tools/kernel_gallery.py`.

## Sweeps

```bash
python3 sweep/make_jobs.py stage0                         # -> results/stage0/jobs.jsonl
python3 sweep/run_jobs.py results/stage0/jobs.jsonl --workers 24
python3 sweep/merge.py results/stage0                     # -> results/stage0/stage0.csv
python3 analysis/summarise.py results/stage0              # -> results/stage0/summary/
python3 analysis/figures.py results/stage0                # -> results/stage0/figures/
```
`run_jobs.py` uses only the standard library. Each run writes `results/<stage>/rows/<run_id>.csv`
and `stats/<run_id>.csv`. A rerun skips finished runs: the run key covers every result-
affecting argument. Ctrl-C kills the workers cleanly, failures go to
`results/<stage>/failures.log`, and the machine is kept awake (caffeinate /
systemd-inhibit / SetThreadExecutionState). Stage-to-stage selection uses validation data
only: `sweep/select_stage2.py` after Stage 1, `sweep/select_lr.py` after the Stage 2 sweep.

### On AWS (all stages run there; DECISIONS 42)

One instance type, one AMI and one build for every stage (results from different
compiler/libm pairs can differ from epoch 1; DECISIONS 50).

```bash
sudo apt update && sudo apt install -y cmake g++ git python3 tmux
git clone <repo> article && cd article && git checkout v6-cnn   # the datasets are in git
cd version6 && ./build.sh test && python3 tests/test_selection.py
python3 tools/bench.py --workers-list 96,144,192               # ~20 min; replaces the M1 ratio
tmux new -s v6                                                 # then, inside tmux:
W=192 bash sweep/run_all_aws.sh                               # Stages 0-3 in order, resumable
```
Step-by-step console and command guide (Turkish): `AWS_RUNBOOK_TR.md`.
Order: Stage 0 → Stage 1 → `select_stage2.py` → `make_jobs.py stage2` → Stage 2 sweep →
`select_lr.py` → `make_jobs.py stage2 --part ext` → extension → `make_jobs.py stage3` →
Stage 3. Seeds: Stage 1 101–103, Stage 2 1–30, Stage 3 31–45 (PLAN.md, *Seed sets*).

Results: commit the merged `results/<stage>/<stage>.csv` and summaries; keep the
diagnostics out of git as `stats_<stage>.tar.gz` (≈ 0.8 GB Stage 1, 2.3 GB Stage 2 before
compression), e.g. `tar czf stats_stage1.tar.gz results/stage1/stats` then copy to S3 or
`rsync` to the laptop. Stage 2's confirmatory tables: `python3 analysis/summarise.py
results/stage2 --seeds 11-30` (needs numpy, pandas, scipy; run it on the laptop).

## Tests (PROMPT.md section 9)

| # | test | how |
|---|---|---|
| 1 | loader equivalence (mmap vs ifstream, all images; CIFAR plane order) | `bin/test_loader ..` |
| 2 | version5 regression, bit-identical (POSIX variant) | `python3 tests/regress_v5.py` |
| 3 | finite-difference gradient check, float64, production code | `bin/test_gradcheck` |
| 4 | optimised vs reference convolution | `bin/test_conv_reference` |
| 5 | PyTorch float64 equivalence, 200 SGD steps | `.venv/bin/python tests/torch_equiv.py` |
| 6 | kernel banks | `bin/test_kernels kernels` |
| 7–10 | RNG alignment, frozen mode, determinism / workers, stats invariance | `python3 tests/test_runs.py` |
| 10, 11 | dead-unit hand check, collapse-abort logic, in-model RNG alignment | `bin/test_misc` |
| 12 | smoke runs (every arch × mode × family × K) | `make_jobs.py smoke`, `run_jobs.py`, `merge.py`, `tests/check_smoke.py` |

`ctest` (inside `build/`) runs the C++ ones. `bin/profile_layers` times each layer;
`tools/bench.py` and `tools/estimate.py` produce `results/bench/`.
