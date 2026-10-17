# version6 experiment plan

Nothing below has been run. Stages 0–3 start only after Barkev approves them.

Common to every stage: `--rng-mode split`, 40 epochs, cross-entropy (linear outputs +
softmax), batch-size-1 SGD at a fixed LR, the version5 split (SPLIT_SEED 12345),
diagnostics every epoch (`stats_every = 1`), `--abort-on-collapse` off, evaluation seeds
1–30, and validation-only selection. Same seed ⇒ same initial dense weights, same conv
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
  0.01, 0.03}. Every model runs on the full grid. Seeds 1–3.
- Mask control: every frozen bank, tanh+relu with `--conv-mask-offset 1` (the channels
  that were tanh become ReLU and vice versa), at MNIST 0.01 / CIFAR-10 0.003, seeds 1–3.
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
   validation accuracy (seeds 1–3) is below 50% of its best grid mean final validation
   accuracy; if none, the largest grid LR.
3. Datasets select independently.

*Can show:* which fixed filters let a frozen or initialised conv layer match a learned
one, and where the hybrid's protection is largest. *Cannot show:* precise effects. Three
seeds is a screen, and the selection uses the same runs it ranks (Stage 2 re-tests the
winners on fresh seeds and a finer grid).

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
width; paired tests at 30 seeds at the three LRs that matter; whether the effect needs the
mix in the conv layer, the hidden layer or both (controls). *Cannot show:* behaviour at
untested LRs or widths, or with other optimisers or batch sizes.

## Stage 3 — activation ratio × learning rate (450 runs)

Configuration 2 at H = 128, both datasets; models tanh+relu@ρ for ρ ∈ {0.25, 0.5, 0.75}
(ρ = share of ReLU: 2/4/6 of the 8 channels, 32/64/96 of the 128 neurons), relu and tanh;
LRs = Stage 2's stress LR for that setting and the grid LRs just below and above it;
seeds 1–15.

*Can show:* how much tanh is needed to keep stability and how much ReLU to keep accuracy,
around the transition. *Cannot show:* the optimal ratio in general. Three ratios at three
LRs is a slice, not a surface.

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

ESTIMATES_PLACEHOLDER
