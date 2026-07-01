#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"

run_example() {
    local dir="$1"
    local output="$2"
    local flags="$3"

    echo "== $(basename "$dir") =="
    pushd "$dir" >/dev/null
    g++ -O2 $flags -std=c++17 main.cpp -o "$output"
    "./$output"
    popd >/dev/null
    echo
}

run_example "$root_dir/compare_blend" compare_blend "-msse2"
run_example "$root_dir/movemask" movemask "-msse2"
run_example "$root_dir/compress_store" compress_store "-mssse3"