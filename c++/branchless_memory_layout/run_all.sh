#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"

run_example() {
    local dir="$1"
    local output="$2"

    echo "== $(basename "$dir") =="
    pushd "$dir" >/dev/null
    g++ -O2 -std=c++17 main.cpp -o "$output"
    "./$output"
    popd >/dev/null
    echo
}

run_example "$root_dir/halo_padding" halo_padding
run_example "$root_dir/double_buffer" double_buffer
run_example "$root_dir/two_phase_compact" two_phase_compact
run_example "$root_dir/table_driven_parser" table_driven_parser