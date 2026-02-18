#!/bin/bash

set -e

g++ -Wall -Wextra -O2 -o main main.cpp net.cpp data.cpp

echo "Compilation successful. Running program..."

seed=50

while true; do
    echo "=== Starting 9-combo cycle with seed: $seed ==="

    ./main relu relu "$seed" 10
    ./main tanh relu "$seed" 10
    ./main leaky_relu leaky_relu "$seed" 10
    ./main tanh leaky_relu "$seed" 10

    seed=$((seed + 8))
done
