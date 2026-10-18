# Benchmark (Apple M1, provisional — the runs will happen on AWS)

- Machine: Apple M1 (4 performance + 4 efficiency cores, 8 GB), macOS-26.6.2-arm64-arm-64bit
- Binary: AppleClang 21.0.0.21000101, `-O3 -march=native -ffp-contract=off -fno-fast-math`, commit 26e4f0d (dirty=1)
- Conditions: AC power, **macOS Low Power Mode on** (not changed by the agent), no thermal or performance warning recorded by `pmset -g therm`; room conditions not measured.
- Drift (CIFAR-10 H=128 cnn K=5, 3 epochs, 8 workers): epoch 1 → 3 mean 33.1 → 32.8 s. No throttling.
- Diagnostics size: 4378 B per measurement point, about 179 KB per 40-epoch run (CIFAR-10 cnn, mixed masks).
- Benchmark wall time: 42.7 min.

## Worker scaling (CIFAR-10, H=128, cnn K=5 random, 1 epoch, diagnostics on)

| workers | jobs/hour | epoch s (min–max) |
|---:|---:|---|
| 4 | 380 | 29.4–29.5 |
| 6 | 547 | 31.0–31.0 |
| 8 | 673 | 33.3–33.6 |

Chosen: **8 workers**. On the M1, the 4 efficiency cores still add throughput.

## Seconds per epoch with 8 workers busy (training 45k + evaluation 15k + diagnostics), and per-run overhead

| dataset | H | mlp1 | mlp2 | cnn K3 rand | K3 frozen | K5 rand | K5 frozen | K7 rand | K7 frozen | K9 rand | K9 frozen | overhead (cnn K5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mnist | 64 | 2.5 | 2.7 | 11.5 | 8.9 | 13.5 | 10.0 | 16.2 | 11.9 | 19.6 | 14.1 | 3.9 |
| mnist | 128 | 4.5 | 5.5 | 16.9 | 14.0 | 18.8 | 14.4 | 21.9 | 17.3 | 25.2 | 18.5 | 5.1 |
| mnist | 256 | 9.0 | 13.1 | 29.6 | 24.2 | 31.5 | 25.8 | 34.3 | 26.9 | 37.4 | 29.2 | 8.3 |
| cifar10 | 64 | 8.5 | 8.9 | 18.5 | 13.6 | 26.5 | 18.6 | 37.2 | 25.4 | 48.5 | 34.3 | 7.5 |
| cifar10 | 128 | 19.1 | 20.5 | 25.4 | 25.5 | 34.2 | 39.2 | 46.6 | 43.5 | 66.8 | 50.7 | 9.4 |
| cifar10 | 256 | 69.3 | 78.4 | 64.4 | 39.5 | 57.3 | 44.1 | 68.3 | 45.4 | 72.0 | 53.6 | 14.4 |

CIFAR-10 H=256 is memory-bound on this 8 GB laptop: mlp1 goes from 19.1 to 69.3 s/epoch when the width doubles.

## AWS reference

version5's code path (`mlp2 --rng-mode legacy`, CIFAR-10 H=128) takes 43.5 s/epoch here under the same load, against 19.4 s/epoch measured by version5 on a c8g.48xlarge (Graviton4) at 190 concurrent workers. Ratio 2.25; `tools/estimate.py` divides by it.
(The same job with version6's blocked dot product, `mlp2 --rng-mode split`, takes 20.5 s/epoch here: the new summation order alone halves the MLP's cost.)
