# version6 experiment plan

Nothing below has been run. Stages 0–3 start only after Barkev approves them.

Common to every stage: `--rng-mode split`, 40 epochs, cross-entropy (linear outputs +
softmax), batch-size-1 SGD at a fixed LR, the version5 split (SPLIT_SEED 12345),
diagnostics every epoch (`stats_every = 1`), `--abort-on-collapse` off, evaluation seeds
1–30 (Stage 1 screening seeds 101–103, Stage 3 seeds 31–45; see *Seed sets* below), and
validation-only selection. **Every stage runs on AWS** on one instance type, one AMI and
one build (DECISIONS 42). Same seed ⇒ same initial dense weights, same conv
random weights and same data order across every model, kernel mode and layout (tested,
WORKLOG/test 7), so every model comparison is paired by seed.

Job files: `python3 sweep/make_jobs.py <stage>` → `results/<stage>/jobs.jsonl`.
Run: `python3 sweep/run_jobs.py results/<stage>/jobs.jsonl --workers N`.

## Stage 0 — MLP mechanism check (240 runs)

`mlp2`, MNIST, H ∈ {64, 128}, LR ∈ {0.02, 0.03, 0.05}, models {relu, leaky_relu, tanh,
tanh+relu} (tanh+relu = 50:50 interleave in both hidden layers), seeds 1–10.

*Can show:* does ReLU's late collapse coincide with ReLU units dying (dead-unit fraction
over epochs per hidden layer), and with an input-independent output (between-input logit
std → 0, one predicted class, training loss → ln 10)? Does the tanh half of the hybrid
stay alive, and what happens to weights, gradients and clip hits during leaky-ReLU's
one-epoch loss spike? Does training loss rise before validation collapses?
*Cannot show:* seed-for-seed replicas of version5. `split` RNG gives different initial
weights and orders; the comparison with version5 is distributional.

## Stage 1 — kernel screen, H = 128, both datasets (4,554 runs)

- 48 kernel configurations: `random` × K ∈ {3,5,7,9}; {gauss, sharpen, sobel, laplacian,
  gabor} × K × {frozen, bank_init}; `random_frozen` × K × frozen. All have C = 8 channels
  and pool 2, so 1,568 (MNIST) or 2,048 (CIFAR-10) features feed the hidden layer.
- Models: relu, tanh, tanh+relu (both layers mixed, ratio 0.5).
- LR grids: MNIST {0.001, 0.003, 0.01, 0.03, 0.1}; CIFAR-10 {0.0003, 0.001, 0.003,
  0.01, 0.03}. Every model runs on the full grid. Screening seeds 101–103.
- Mask control: every frozen bank, tanh+relu with `--conv-mask-offset 1` (the channels
  that were tanh become ReLU and vice versa), at MNIST 0.01 / CIFAR-10 0.003, seeds 101–103.
- No-conv reference: `mlp1` with the same models, LRs and seeds.

**Pre-registered selection for Stage 2** (`sweep/select_stage2.py`, validation only, per
dataset; tested in `tests/test_selection.py`):
1. *Configuration 1, standard CNN:* the `random`-mode K with the highest mean best-
   validation accuracy for ReLU at its best grid LR. Ties → smaller K.
2. *Configuration 2, most informative bank:* among frozen and bank_init configurations
   whose ReLU mean best-validation accuracy (at its best grid LR) is within 2 pp of
   configuration 1's, the one with the largest mean (tanh+relu − relu) difference in
   **final** validation accuracy at its own Stage 1 stress LR. Ties → smaller K, then family
   order (gauss, sharpen, sobel, laplacian, gabor, random_frozen), then frozen before
   bank_init.
   *Stage 1 stress LR* of a configuration: the smallest grid LR at which ReLU's mean final
   validation accuracy (seeds 101–103) is below 50% of its best grid mean final validation
   accuracy; if none, the largest grid LR.
3. Datasets select independently.

*Can show:* which fixed filters let a frozen or initialised conv layer match a learned
one, and where the hybrid's protection is largest. *Cannot show:* precise effects. Three
seeds is a screen, and the selection uses the same runs it ranks. Stage 2 re-tests the
winners on seeds 1–30, none of which appears in Stage 1, on a finer grid.

## Stage 2 — main CNN study (6,860 sweep runs + 5,880 extension runs)

- Architectures per dataset: configuration 1, configuration 2, `mlp1`.
- Widths H ∈ {64, 128, 256} for all three. H = 512 is deliberately excluded to save
  compute; the article's H = 512 evidence comes from version5's MLP.
- Models: relu, leaky_relu, tanh, sigmoid, tanh+relu; plus the controls tanh+relu@conv
  (conv mixed, hidden all ReLU) and tanh+relu@hidden (conv all ReLU, hidden mixed) for
  configuration 2 at H ∈ {128, 256}.
- LR grids: MNIST {0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1}; CIFAR-10 {0.0003,
  0.001, 0.003, 0.01, 0.02, 0.03, 0.05}. Sweep: seeds 1–10 on the full grid.
- **LR selection** (pre-registered, `sweep/select_lr.py`): per (arch, dataset, H, model),
  the LR with the highest mean **best-validation** accuracy over seeds 1–10. This matches
  the checkpoint used for the test score, fixing version5's final-vs-checkpoint mismatch.
  Ties → lower LR.
- **Stress LR** (pre-registered): per (arch, dataset, H), the smallest grid LR at which
  ReLU's mean final validation accuracy over seeds 1–10 is below 50% of its best grid
  mean; if none, the largest grid LR.
- **Extension**: seeds 11–30 for every model at three LRs: the selected LR, the stress
  LR, and the grid LR just below the stress LR (duplicates removed).

*Can show:* whether the MLP's collapse resistance transfers to a CNN, per dataset and
width; paired tests at the three LRs that matter, **confirmatory on seeds 11–30 only**
(those LRs were chosen from seeds 1–10; `analysis/summarise.py results/stage2 --seeds
11-30`), with the 1–30 tables as secondary; whether the effect needs the
mix in the conv layer, the hidden layer or both (controls). *Cannot show:* behaviour at
untested LRs or widths, or with other optimisers or batch sizes.

## Stage 3 — activation ratio × learning rate (450 runs)

Configuration 2 at H = 128, both datasets; models tanh+relu@ρ for ρ ∈ {0.25, 0.5, 0.75}
(ρ = share of ReLU: 2/4/6 of the 8 channels, 32/64/96 of the 128 neurons), relu and tanh;
LRs = Stage 2's stress LR for that setting and the grid LRs just below and above it;
seeds 31–45 (fresh: the stress LR was chosen from Stage 2's seeds 1–10, and no Stage 3 run
repeats a Stage 2 run).

*Can show:* how much tanh is needed to keep stability and how much ReLU to keep accuracy,
around the transition. *Cannot show:* the optimal ratio in general. Three ratios at three
LRs is a slice, not a surface.

## Seed sets (decision of 2026-10-18)

Under `--rng-mode split` the same seed gives the same initial weights and data order, so a
run with a reused seed is an exact repeat. Seed sets are therefore kept disjoint wherever a
later stage tests something an earlier stage selected:

| stage | seeds | used for |
|---|---|---|
| 0 | 1–10 | mechanism check (`mlp2`; no selection) |
| 1 | 101–103 | kernel screen → Stage 2 configurations |
| 2 sweep | 1–10 | LR selection and stress LR |
| 2 extension | 11–30 | confirmatory tests at the selected / stress / below-stress LRs |
| 3 | 31–45 | ratio × LR at Stage 2's stress LR |

## Outcome measures (what the analysis supports)

Per run (`analysis/load.py`): final and best validation accuracy; peak-to-final drop;
`final < 15%`; `drop > 10 pp`; `ever below 15%` (any epoch); first collapse epoch; final
training loss vs ln 10; between-input logit std; dead-unit fractions over time (per layer
and activation group); checkpoint and final test accuracy and macro-F1.

Summaries (`analysis/summarise.py`): per cell (dataset, configuration, H, model, LR), the
means, SDs and counts above, and the LR relative to each model's own best LR (version5
showed the hybrid's LR curve is shifted as well as widened). Paired comparisons by seed:
mean difference, 95% paired bootstrap CI (10,000 resamples, seed 20261018), wins, paired
t p, and Holm-adjusted p within each family (dataset × metric × comparison).

Figures (`analysis/figures.py`): LR curves (final and best validation); failure maps
under both the 15% and the 10 pp rules; dead-unit fraction and logit std over epochs next
to validation accuracy; kernel-family × K heatmaps (Stage 1).

## Budget

Measured on the Apple M1 laptop (`results/bench/bench.md`; provisional: Low Power Mode
on, 8 workers). Formula: runs × (40 × seconds/epoch + per-run overhead) ÷ workers × 1.05
+ the longest single run. AWS column: the same per-run times divided by 2.25, the measured
ratio between this machine and a c8g.48xlarge on version5's own code path, spread over
192 workers. The price is version5's spot price (≈ $2.6/h). Stages 2 and 3 assume
configuration 1 = random K = 3, configuration 2 = bank_init K = 5, and three distinct
extension LRs (an upper bound).

Workers here: 8. AWS: 192 workers, per-run time / 2.25 (this machine vs c8g.48xlarge on the version5 code path).

| stage | variant | runs | epochs | here (h) | target (h) | AWS (h) | AWS cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| stage0 | full | 240 | 9600 | 1.5 | 4 | 0.05 | 0 |
| stage1 | full | 4554 | 182160 | 198.2 | 72 | 4.00 | 10 |
| stage1 | seeds 3 -> 2 | 3036 | 121440 | 132.4 | 72 | 2.77 | 7 |
| stage1 | + drop K = 7 | 2292 | 91680 | 97.1 | 72 | 2.12 | 6 |
| stage1 | + drop lowest LR | 1848 | 73920 | 78.4 | 72 | 1.77 | 5 |
| stage2 | full | 12740 | 509600 | 494.9 | 120 | 9.51 | 25 |
| stage2 | sweep seeds 10 -> 6 | 11172 | 446880 | 434.1 | 120 | 8.38 | 22 |
| stage2 | + extension 30 -> 20 seeds | 8232 | 329280 | 320.0 | 120 | 6.27 | 16 |
| stage2 | + drop @conv/@hidden at H=256 | 7896 | 315840 | 298.1 | 120 | 5.86 | 15 |
| stage2 | + drop sigmoid CIFAR H=256 | 7644 | 305760 | 274.6 | 120 | 5.42 | 14 |
| stage3 | full | 450 | 18000 | 17.9 | 24 | 0.49 | 1 |

**Reading the table.**
- On this laptop only Stage 0 (1.5 h ≤ 4 h) and Stage 3 (17.9 h ≤ 24 h) fit their targets.
  Stage 1 still needs 78 h after every prescribed reduction (target 72 h); Stage 2 needs
  275 h after every prescribed reduction (target 120 h). The prescribed reduction order
  cannot bring either stage within budget here.
- On AWS (192 Graviton4 workers) the **full** plan takes about 0.05 + 4.0 + 9.5 + 0.5 ≈
  14 instance-hours, about $37 at spot prices. No reduction is needed there.
  Decision (2026-10-18): Stages 0–3 all run on AWS in full. Before
  launching, re-run `tools/bench.py --workers-list 96,144,192` on the instance (≈ 20 min):
  the 2.25 ratio was measured on the MLP, and the CIFAR-10 H = 256 cells were memory-bound
  on the laptop (see bench.md).
- Diagnostics cost ≈ 180 KB per run: Stage 1 ≈ 0.8 GB, Stage 2 ≈ 2.3 GB of stats files.
  Decision (2026-10-18): kept out of git as one compressed archive per stage
  (`stats_<stage>.tar.gz`, stored outside the repository, e.g. S3); git holds the merged
  row CSVs and the summaries (DECISIONS 39).

Reduced variants, if the work must stay on one machine (PROMPT.md 11.5 order; never
dropping relu, tanh, tanh+relu, the stress LR or the random CNN baseline):
- Stage 1: seeds 3 → 2, then drop K = 7, then drop the lowest grid LR (1,848 runs).
- Stage 2: sweep seeds 10 → 6, extension 30 → 20 seeds, drop the @conv/@hidden controls
  at H = 256, drop sigmoid at CIFAR-10 H = 256 (7,644 runs).
- Stage 3 (not in PROMPT; proposed only if needed): seeds 15 → 10.
