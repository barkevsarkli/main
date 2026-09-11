# Neural Network from Scratch in C++

A progressive exploration of neural network architectures built entirely from scratch in C++, evolving from a basic feedforward network to a biologically-inspired neural simulation. No external ML libraries are used -- all math, backpropagation, and data loading are implemented manually.

## Project Overview

| Version | Dataset | Architecture | Training Method | Key Innovation |
|---------|---------|-------------|----------------|----------------|
| **v1** | MNIST | 2-layer (1 hidden) | Backprop + SGD | Baseline implementation |
| **v2** | MNIST | 3-layer (2 hidden) | Backprop + SGD | Multi-activation experimentation (even/odd neuron split) |
| **v3** | MNIST | 3-layer (2 hidden) | Backprop + SGD | Scaled hidden layers (128 neurons), configurable epochs |
| **v4** | CIFAR-10 | 4-layer (3 hidden) | Backprop + SGD | Color image support (32x32 RGB) |
| **v5** | MNIST | 3-layer (2 hidden), width on CLI | Backprop + SGD | Corrected protocol: per-neuron activation masks, uniform output layer, tuned LR, fixed val/test split, macro-F1 |

## Datasets

- **MNIST**: 60,000 handwritten digit images (28x28 grayscale), stored in `train-images-idx3-ubyte/` and `train-labels-idx1-ubyte/`
- **CIFAR-10**: 60,000 color images (32x32 RGB) across 10 classes, stored in `CIFAR10/`
- **BMP Samples**: Exported sample images in `bmp_images/`

---

## Version Details

### Version 1 -- Baseline Feedforward Network

The simplest implementation: a 2-layer fully connected network for MNIST digit classification.

- **Architecture**: `784 -> 10 (hidden) -> 10 (output)`
- **Activation**: ReLU (hardcoded)
- **Weight Init**: He initialization (normal distribution)
- **Learning Rate**: 0.0001
- **Epochs**: 25 (hardcoded)
- **Loss**: Mean Squared Error (MSE)
- **Files**: `main.cpp`, `net.cpp`, `net.h`, `data.cpp`

The `Neuron` class represents a dense layer. Forward pass applies ReLU inline. Backward pass uses vanilla gradient descent with no clipping or stability protections.

**Limitations**: Hardcoded paths, no CLI arguments, no CSV output, no gradient clipping, small hidden layer.

---

### Version 2 -- Multi-Activation Function Experimentation

Introduces a systematic framework for comparing activation function combinations across neurons.

- **Architecture**: `784 -> 12 (hidden1) -> 12 (hidden2) -> 10 (output)`
- **Activations**: ReLU, Leaky ReLU, Tanh, Sigmoid (selectable via CLI)
- **Weight Init**: Kaiming uniform initialization
- **Learning Rate**: 0.01 (with adjustments for sigmoid/tanh)
- **Epochs**: 10 (hardcoded)
- **Loss**: MSE with NaN detection
- **CLI**: `./main <func1> <func2> <seed>`
- **Files**: `main.cpp`, `net.cpp`, `net.h`, `data.cpp`, `data.h`, `runtanh.sh`

**Key changes from v1**:
- **Even/Odd neuron activation split**: Even-indexed neurons use `func1`, odd-indexed neurons use `func2`. This enables testing heterogeneous activation within the same layer.
- **Added a second hidden layer** (3 layers total).
- **Gradient clipping** at +/-10.0 to prevent exploding gradients.
- **Output value clamping** at +/-20.0 before activation.
- **Numerically stable sigmoid** implementation.
- **CSV output** for batch experiment results (seed, activations, per-epoch loss, final test accuracy).
- **Seeded randomness** via CLI for reproducible runs.
- **Separated `data.h`/`data.cpp`** from a single-file design.
- **Shell script** (`runtanh.sh`) for automated batch experiments with incrementing seeds.
- **Training accuracy** tracked during each epoch.

**Experiment results** include 28 CSV files covering combinations like `relu_relu`, `tanh_relu`, `tanh_leaky_relu`, `sigmoid_sigmoid`, etc. with hidden sizes of 5, 10, 12, and 16.

---

### Version 3 -- Scaled Hidden Layers

Scales up the network capacity and makes epoch count configurable.

- **Architecture**: `784 -> 128 (hidden1) -> 128 (hidden2) -> 10 (output)`
- **CLI**: `./main <func1> <func2> <seed> <epochs>`
- **Files**: `main.cpp`, `net.cpp`, `net.h`, `run.sh`

**Key changes from v2**:
- **Hidden layer size increased** from 12 to 128 neurons (10x).
- **Epochs are now a CLI argument** instead of hardcoded.
- **Network code** (`net.cpp`, `net.h`) is identical to v2.
- **Shell script** (`run.sh`) tests 4 activation combos: `relu_relu`, `tanh_relu`, `leaky_relu_leaky_relu`, `tanh_leaky_relu`.

**Experiment results** include runs with hidden sizes of 32 and 128.

---

### Version 4 -- CIFAR-10 Support

Adapts the network for color image classification with a deeper architecture.

- **Architecture**: `3072 -> 258 (hidden1) -> 258 (hidden2) -> 258 (hidden3) -> 10 (output)`
- **Input Size**: 3072 (32 x 32 x 3 channels)
- **Learning Rate**: 0.001 (lower than MNIST versions)
- **CLI**: `./main <func1> <func2> <seed> <epochs>`
- **Files**: `main.cpp`, `net.cpp`, `net.h`, `cifar_data.cpp`, `cifar_data.h`, `run.sh`

**Key changes from v3**:
- **New dataset**: CIFAR-10 instead of MNIST.
- **New data loader** (`CIFAR_DATA` class) handling:
  - RGB channels stored separately (R, G, B byte arrays per image).
  - Loading 5 training batches (50,000 images) + 1 test batch (10,000 images).
  - Flat float arrays for normalized pixel data.
  - One-hot encoding using float arrays (instead of uint8).
- **Third hidden layer** added (4 layers total).
- **Larger hidden size** (258 neurons) to handle more complex features.
- **Lower learning rate** (0.001 vs 0.01) for the more complex dataset.
- **Network code** (`net.cpp`, `net.h`) is identical to v2/v3.

---

### Version 5 -- Corrected Hybrid-Activation Protocol

Re-tests the v2 hypothesis ("mixing activation functions within a layer improves accuracy") with the confounds removed. Results and interpretation: `version5/FINDINGS.md`; full tables: `version5/results/summary.md`.

- **Architecture**: `784 -> H -> H -> 10`, `H` on the CLI (12 and 128 tested)
- **Activations**: per-neuron mask built from `--act1 --act2 --ratio --layout` (`interleave` at ratio 0.5 reproduces the v2 even/odd split; `block` and `random` also available)
- **Output layer**: uniform -- linear + softmax with cross-entropy (`--loss ce`, default) or sigmoid with MSE (`--loss mse`)
- **Split**: fixed 45k / 5k / 10k train / val / test, identical for every run; per-epoch shuffling seeded by `--seed`
- **Metrics**: accuracy, macro-F1 and confusion matrix on validation (every epoch) and test (final)
- **CLI**: `./main --act1 tanh --act2 relu --ratio 0.5 --layout interleave --hidden 12 --seed 1 --epochs 10 --lr 0.003 --loss ce --data .. --out results/x.csv`
- **Files**: `main.cpp`, `net.cpp`, `net.h`, `data.cpp`, `data.h`, `sweep/` (job generation, LR selection, resumable chunked runner), `analysis/analyze.py`, `results/`

**Fixes relative to v2/v3**: output layer no longer hybridised; `tanh/tanh`, `leaky/leaky`, `sigmoid/sigmoid` baselines added; learning rate tuned per configuration on validation with calibration seeds disjoint from evaluation seeds; single leaky-ReLU slope in forward and backward; `std::mt19937` instead of `rand()`; data path on the CLI.

**Headline** (one-hour first-look sweep, 20 seeds at H=12, 6 seeds at H=128): the 6.5-point v2 advantage of `tanh/relu` over `relu/relu` does not survive the corrections -- it came from ReLU instability under the v2 setup. What survives: hybrids are far more robust to the learning rate (ReLU collapses at lr 0.03--0.1, the hybrid does not), and at H=128 `tanh+leaky_relu` beats `leaky_relu` on 6/6 seeds (+0.30 pts, p = 0.007) while at H=12 every hybrid is slightly below its better parent.

---

## Building and Running

### Version 1
```bash
cd version1
g++ -Wall -Wextra -O2 -o main main.cpp
./main
```

### Version 2
```bash
cd version2
g++ -Wall -Wextra -O2 -o main main.cpp net.cpp data.cpp
./main relu tanh 42
```

### Version 3
```bash
cd version3
g++ -Wall -Wextra -O2 -o main main.cpp net.cpp data.cpp
./main relu relu 42 10
```

### Version 4
```bash
cd version4
g++ -Wall -Wextra -O2 -o main main.cpp net.cpp cifar_data.cpp
./main tanh leaky_relu 42 10
```

### Version 5
```bash
cd version5
g++ -std=c++17 -O3 -march=native -o main main.cpp net.cpp data.cpp
./main --act1 tanh --act2 relu --ratio 0.5 --hidden 12 --seed 1 --epochs 10 --lr 0.003 --data .. --out results/test.csv
python3 analysis/analyze.py      # -> results/summary.md + results/figures/
```

## Evolution Summary

```
v1: Basic 2-layer MLP (ReLU only, MNIST)
 |
 v--- Added multi-activation, deeper network, experiment framework
 |
v2: 3-layer MLP with configurable activation pairs (MNIST)
 |
 v--- Scaled hidden layer size (12 -> 128), configurable epochs
 |
v3: Wider 3-layer MLP (MNIST)
 |
 v--- New dataset, deeper network, CIFAR-10 data loader
 |
v4: 4-layer MLP (CIFAR-10)
 |
 v--- Confounds removed, per-neuron activation masks, F1, paired statistics
 |
v5: Corrected hybrid-activation protocol (MNIST, H = 12 / 128)
```

## File Structure

```
main-main/
├── README.md
├── bmp_images/                  # Sample exported MNIST images (.bmp)
├── CIFAR10/                     # CIFAR-10 dataset (binary batches)
│   ├── train_images/            # 5 training batch files
│   └── test_images/             # 1 test batch file
├── train-images-idx3-ubyte/     # MNIST training images
├── train-labels-idx1-ubyte/     # MNIST training labels
├── results_sorted*.csv          # Aggregated experiment results
├── *.png                        # Analysis visualizations
│
├── version1/                    # Baseline MLP
│   ├── main.cpp
│   ├── net.cpp / net.h
│   └── data.cpp
│
├── version2/                    # Multi-activation experiments
│   ├── main.cpp
│   ├── net.cpp / net.h
│   ├── data.cpp / data.h
│   ├── runtanh.sh
│   └── results_*.csv            # 28 experiment result files
│
├── version3/                    # Scaled hidden layers
│   ├── main.cpp
│   ├── net.cpp / net.h
│   ├── run.sh
│   └── results_*.csv
│
├── version4/                    # CIFAR-10 support
│   ├── main.cpp
│   ├── net.cpp / net.h
│   ├── cifar_data.cpp / cifar_data.h
│   ├── run.sh
│   └── cifar_results_*.csv
│
└── version5/                    # Corrected hybrid-activation protocol
    ├── FINDINGS.md              # results + interpretation
    ├── main.cpp, net.cpp / net.h, data.cpp / data.h
    ├── sweep/                   # make_jobs.py, select_lr.py, pending.py, run_chunk.sh
    ├── analysis/analyze.py      # paired stats, bootstrap CIs, figures
    └── results/                 # *.csv, best_lr.json, summary.md, figures/
