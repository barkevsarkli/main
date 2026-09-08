# Prompt for Claude Code — extended hybrid-activation sweep (version5)

Copy everything below the line into Claude Code, started in the repository root (the folder that contains `README.md` and `version5/`).

---

## Context

This repository is a from-scratch C++ MLP project (no ML libraries). Read `README.md` first, then `version5/FINDINGS.md` and `version5/results/summary.md`. The hypothesis under test: **mixing activation functions across neurons within a hidden layer ("hybrid activation") changes accuracy, macro-F1 and training robustness compared to homogeneous layers.**

A first-look sweep in `version5/` (≈1 h of compute) found:

1. At width 12 the hybrid `tanh+relu` is ~0.46 points *below* `relu` and tied with `tanh` — the earlier v2 result (+6.5 points) was an artefact of confounds that version5 removed.
2. Hybrids are much more robust to the learning rate: `relu` collapses at high LR (17.8% at lr 0.1 under CE; 76% at lr 0.03 under MSE) while `tanh+relu` keeps training (79% / 92.6%).
3. At width 128, `tanh+leaky_relu` beat `leaky_relu` on 6/6 seeds (+0.30 points, p = 0.007) and was the best configuration; `tanh+relu` tied `tanh` and led `relu` on 5/6 seeds.

Points 2 and 3 are the candidate article claims. They need far more data: more widths (to locate the crossover), more seeds, and a seed-replicated learning-rate study. **Your job is to run that sweep, extend the analysis, and report the results. Do not change the experimental protocol.**

## Ground rules

- Work only inside `version5/`. Do not touch `version1`–`version4` (they are the historical record for the article).
- **Do not change the training semantics** in `main.cpp` / `net.cpp` (init, SGD batch-size 1, per-epoch shuffle, fixed 45k/5k/10k split with `SPLIT_SEED 12345`, gradient clip, LEAKY_SLOPE 0.1, loss definitions). If you optimise anything for speed, prove on two seeds at width 12 and one at width 128 that the CSV rows (test_acc, test_f1, epoch_loss) are bit-identical before and after.
- **Never overwrite or delete existing files in `version5/results/`.** New runs go to new CSV files (naming below). The existing `main_12.csv`, `main_128.csv`, `layout_12.csv`, `mse_12.csv`, `calib_*.csv` stay as they are.
- Everything must be resumable. Use the existing `sweep/run_chunk.sh` + `sweep/pending.py` mechanism (a run is identified by seed/act1/act2/ratio/layout/hidden/lr/loss; finished runs are skipped automatically). If the machine sleeps or a job dies, re-running the same command continues where it left off.
- Keep the machine awake while running: prefix long commands with `caffeinate -i`.
- Log every stage to `version5/results/sweep_extended.log` with timestamps. Print a short progress line after each chunk (remaining count, elapsed time, ETA).
- Do not commit to git unless asked.

## Step 0 — build and timing (do this first, report before continuing)

```bash
cd version5
clang++ -std=c++17 -O3 -o main main.cpp net.cpp data.cpp        # add -mcpu=native on Apple Silicon if it compiles; skip -march=native (Apple clang may reject it)
./main --act1 tanh --act2 relu --ratio 0.5 --hidden 12 --seed 1 --epochs 1 --lr 0.003 --quiet --data .. --out /tmp/smoke.csv
```

Then time **one epoch** for widths 12, 32, 64, 128, 256, 512 (`--epochs 1`, any config, `--out /tmp/timing.csv`) and record seconds/epoch. Also check `sysctl -n hw.physicalcpu` and decide the worker count `P` = number of physical cores (memory-bandwidth-bound loops don't benefit from hyper-threads). Using those timings, compute the wall-clock estimate for the full plan below (10 epochs per run) and print it as a table before launching anything. If the total exceeds **12 hours**, reduce seeds at widths 256/512 first (never below 5), and say so.

Reference: in a 2-core cloud container one epoch took ~1.4 s at width 12 and ~7 s at width 128. Width 256 should be roughly 3.5–4× the 128 cost, 512 roughly 4× that again.

## Step 1 — generalise the sweep tooling (small code changes)

`sweep/make_jobs.py` currently hard-codes stages for widths 12 and 128. Replace its stage logic with a general generator driven by CLI flags, keeping the same command-line format for `./main` and the same key semantics used by `pending.py`:

```
make_jobs.py --stage calib --hidden 64 --seeds 101,102,103 --lrs 0.001,0.003,0.01,0.03 --configs all --out results/calib_64.csv
make_jobs.py --stage main  --hidden 64 --seeds 1-20 --configs all --lr-json results/best_lr.json --out results/main_64.csv
make_jobs.py --stage lrsweep --hidden 128 --seeds 1-10 --lrs 0.001,0.003,0.01,0.03,0.1 --configs core --out results/lrsweep_128.csv
make_jobs.py --stage layout --hidden 128 --seeds 1-10 --lr-json results/best_lr.json --out results/layout_128.csv
```

Config sets:
- `all`  = relu, tanh, leaky_relu, sigmoid, tanh+relu, tanh+leaky_relu, sigmoid+relu (homogeneous = `--ratio 0`, hybrid = `--ratio 0.5 --layout interleave`)
- `core` = relu, tanh, leaky_relu, tanh+relu, tanh+leaky_relu
- `layout` stage = tanh+relu and tanh+leaky_relu at layouts {interleave, block, random} × ratio 0.5, plus interleave at ratios {0.25, 0.75}; use the lr calibrated for interleave/0.5 of the same pair.

`--seeds` accepts a comma list or an `a-b` range. `--loss` defaults to `ce`. `--data ..` and `--epochs 10` stay the defaults.

`sweep/select_lr.py` already merges into `results/best_lr.json` keyed by `act1/act2/ratio/layout/hidden/loss`; keep that format so `analysis/analyze.py` and the existing JSON stay valid. Selection criterion stays **mean validation accuracy** over the calibration seeds.

Keep `sweep/run_chunk.sh` as is (it takes `<jobs file> <max jobs this call> <workers>`); for unattended runs wrap it in a loop:

```bash
until [ "$(python3 sweep/pending.py < sweep/jobs_X.txt | wc -l)" -eq 0 ]; do bash sweep/run_chunk.sh sweep/jobs_X.txt 200 $P; done
```

## Step 2 — the sweep plan

All runs: 10 epochs, cross-entropy, `--data ..`. Calibration seeds are 101–103 and are never used for evaluation. Evaluation seeds start at 1 so every width shares seeds 1–N and every comparison stays paired.

| stage | widths | configs | seeds | LR grid | output |
|---|---|---|---|---|---|
| **A. calibration** | 12*, 32, 64, 128*, 256, 512 | all (drop `sigmoid`, `sigmoid+relu` at 256/512) | 101–103 (101–102 at 512) | 0.001, 0.003, 0.01, 0.03 | `results/calib_<W>.csv` |
| **B. main** | 32, 64, 256, 512 (new) + top-ups of 12 and 128 (see note) | all (drop sigmoid pair at 256/512) | 32: 1–30, 64: 1–20, 256: 1–10, 512: 1–5 | best per config from A | `results/main_<W>.csv` |
| **C. LR robustness** | 12, 64, 128 | core | 1–10 | 0.001, 0.003, 0.01, 0.03, 0.1 (add 0.3 at width 12) | `results/lrsweep_<W>.csv` |
| **D. layout/ratio** | 128, 256 | layout set | 1–10 (256: 1–6) | calibrated | `results/layout_<W>.csv` |
| **E. MSE replication** | 128 | relu, tanh, tanh+relu | 1–10 | calibrate {0.003, 0.01, 0.03} on 101–103 first | `results/calib_mse_128.csv`, `results/mse_128.csv` |

*Notes*

- Width 12 and 128 already have calibration CSVs, but on a narrower grid (128 only tried {0.003, 0.01} with one seed). Run stage A for them again on the full grid into **new** files `results/calib_12_v2.csv` and `results/calib_128_v2.csv`, then run `select_lr.py` on those so `best_lr.json` is refreshed. If the selected LR for a width-128 config changes, the existing `main_128.csv` rows are at the old LR: re-run the affected configs for seeds 1–6 into `results/main_128_v2.csv` and extend all configs to seeds 1–20 there. If nothing changes, just extend `main_128.csv` from seed 7 to 20 (same file is fine — `pending.py` will skip 1–6). Width 12: extend `main_12.csv` from seed 21 to 30 if its LRs are unchanged, otherwise start `main_12_v2.csv` with seeds 1–30.
- Run order: A(all widths) → select LR → B for 32/64 (fast) → C for 12 (fast) → B for 256 → C for 64, 128 → D → B for 512 → E. Cheap stages first so partial results are useful early.
- Width 512 is optional; if the Step 0 estimate says the whole plan exceeds 12 h, drop 512 first, then reduce 256 to 6 seeds.

## Step 3 — extend the analysis

`analysis/analyze.py` currently has hand-written sections for widths 12 and 128. Refactor it so it discovers every `results/main_*.csv` (and `_v2` variants — prefer `_v2` when both exist for a width and say so in the report) and produces, per width:

- summary table (config, lr, n, test acc mean ± sd, median, macro-F1 mean ± sd, val acc mean);
- paired comparisons hybrid − each parent for accuracy and macro-F1 with mean Δ, 95% bootstrap CI (10k resamples), wins, paired t p, Wilcoxon p (the existing `paired()` / `paired_table()` functions already do this — reuse them);
- the strip/box figure and the paired-delta figure (existing functions).

Add these new outputs:

1. **Crossover figure** (`fig_crossover.png`): x = width (log scale), y = paired Δ (hybrid − parent) in accuracy points with 95% CI error bars, one line per comparison (`tanh+relu − relu`, `tanh+relu − tanh`, `tanh+leaky_relu − leaky_relu`, `tanh+leaky_relu − tanh`), dashed zero line. Same again for macro-F1. This is the article's key figure.
2. **Absolute accuracy vs width** (`fig_width_acc.png`): mean ± sd per config across widths, one line per config.
3. **LR robustness figures** from stage C (`fig_lrsweep_<W>.png`): val accuracy vs LR per config, mean with a shaded 10–90 percentile band across the 10 seeds, plus a table of "fraction of seeds that collapsed" (define collapse as final val accuracy < 50%) per config × LR.
4. **Robustness score table**: for each config and width, the widest contiguous LR range over which mean val accuracy stays within 1 point of that config's best — this quantifies "more robust to the learning rate".
5. **Layout table** for 128/256, same format as the existing section 3.
6. **Learning-curve figures** per width (existing function).

Write everything to `results/summary_extended.md` (leave `results/summary.md` untouched) and figures to `results/figures/`. Keep the existing palette and styling in `analyze.py` (fixed colour per config; do not cycle colours).

Use the same rules the script already follows: skip rows whose `status` is not `ok`, skip truncated rows (`test_confusion` must have 100 entries), de-duplicate on the run key keeping the first row, and drop repeated header lines (parallel workers can write the header twice).

## Step 4 — sanity checks before reporting

- Every evaluation config at a width has exactly the intended seed set (print a table config × width → n).
- No `status=nan` rows; if any appear, list them with config/seed/lr — do not silently drop them from the report.
- For one hybrid run per width, confirm from the program's stdout mask line that the layer-1 mask has the intended counts (e.g. `relu=64 tanh=64` at width 128).
- Bit-identical reproduction check: rerun one finished job (any width-12 config, seed 1) to a scratch CSV and diff the row against the stored one.
- Run `analysis/analyze.py` end to end with no errors.

## Step 5 — report back (this is what I will paste into the other session)

Write `version5/REPORT_EXTENDED.md` containing, in this order:

1. Machine, compiler flags, cores used, total wall-clock time, and the timing table from Step 0.
2. The run-inventory table (config × width → n, per stage).
3. The refreshed `best_lr.json` as a table (config × width → chosen LR), and whether any width-12/128 selections changed from the first-look sweep.
4. For each width: the summary table and the paired-comparison table (accuracy and macro-F1).
5. The crossover data as a table: comparison × width → Δ, CI, wins/n, p.
6. The LR-robustness collapse table and the robustness-score table.
7. The layout/ratio tables for 128 and 256.
8. The MSE replication at 128.
9. A plain-language paragraph (no more than 200 words) stating what the data show about the three candidate claims: (a) hybrid vs. better parent as a function of width, (b) learning-rate robustness, (c) does layout/ratio matter at larger width. Be explicit about which comparisons are significant at p < 0.01 after considering that ~40 paired tests were run, and which are not.
10. Anything that went wrong, was skipped, or was reduced from the plan, with the reason.

Embed the figures by relative path. Also print the contents of items 5, 6 and 9 to the terminal at the end.

## If you have to choose

Priorities in order: (1) the crossover — widths 32, 64, 256 with the planned seeds; (2) the LR-robustness sweep at 12 and 128 with 10 seeds; (3) the 128 top-up to 20 seeds; (4) layout at 128; (5) width 512; (6) MSE at 128. Cut from the bottom.
