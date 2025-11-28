#!/bin/bash

set -e

seed=66

while true; do
    ./main leaky_relu leaky_relu "$seed" 10
    ./main tanh leaky_relu "$seed" 10

    seed=$((seed + 8))
done
