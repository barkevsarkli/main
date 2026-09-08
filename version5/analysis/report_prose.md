@@machine
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

@@inventory_note
Seeds 1–N are shared by every configuration at every width, so all comparisons are paired.
Calibration seeds are 101–103 and are never used for evaluation.

`main_12_v2.csv` and `main_128_v2.csv` supersede `main_12.csv` and `main_128.csv`: the older files were
produced on a different machine whose `std::shuffle` yields a different train/val/test split from the
same `SPLIT_SEED`, so they are not row-comparable with anything here (see section 10). The analysis
prefers the `_v2` file wherever both exist and ignores the older one; both remain on disk untouched.

@@lr_note
Two width-12 selections moved against the first-look sweep, both on a grid that is now four
points rather than five and on the new split: `tanh+relu` 0.001 → 0.003 and `sigmoid+relu` 0.003 → 0.01.
No width-128 selection changed — the first-look sweep had only tried {0.003, 0.01} with a single seed
there, and the full four-point grid with three seeds confirms the same choice for all five configs it
had calibrated. The two sigmoid configurations at width 128 are calibrated here for the first time.

The learning rate rises with width for the hybrids (0.003 at 12–32, 0.01 from 64 up) and stays at
0.003 for `relu` and `leaky_relu` throughout, which matters when reading the accuracy tables: at the
larger widths the hybrids are running at a learning rate their ReLU parents cannot use.

@@reading
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

@@deviations
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

@@sanity
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
