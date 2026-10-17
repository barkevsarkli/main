#!/usr/bin/env bash
# Release build on Linux/macOS (GCC or Clang). Binaries land in version6/bin/.
#   ./build.sh            configure + build
#   ./build.sh test       ... then run the C++ tests (ctest)
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$( (nproc || sysctl -n hw.ncpu) 2>/dev/null )"
if [ "${1:-}" = "test" ]; then
  (cd build && ctest --output-on-failure)
fi
