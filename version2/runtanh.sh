#!/bin/bash

set -e

g++ -Wall -Wextra -O2 -o main main.cpp net.cpp data.cpp

echo "Compilation successful. Running program..."

activations=("relu" "tanh" "sigmoid")

seed=90   # starting seed (matches your current srand(50))

while true; do
    echo "=== Starting 9-combo cycle with seed: $seed ==="

    ./main tanh relu $seed

    seed=$((seed + 8))   # increment seed after every 9 runs
done
