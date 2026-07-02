#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"
cpu_core="${CPU_CORE:-1}"
opt_level="${OPT_LEVEL:--O3}"

declare -a experiments=(
    "compare_blend"
    "movemask"
    "compress_store"
)

rows_file="$(mktemp)"
cleanup() {
    rm -f "$rows_file"
}
trap cleanup EXIT

parse_pmu_rows() {
    local experiment="$1"
    local output_file="$2"

    awk -v experiment_name="$experiment" '
        /\[pmu\]/ {
            if ($2 == "unavailable:") {
                next
            }

            name = $2
            ns = "-"
            cycles = "-"
            instr = "-"
            ipc = "-"
            cpi = "-"
            br = "-"
            cache = "-"
            fe = "-"
            be = "-"

            for (i = 3; i <= NF; i++) {
                split($i, kv, "=")
                if (length(kv) < 2) {
                    continue
                }
                key = kv[1]
                val = kv[2]
                gsub(/%/, "", val)
                if (key == "ns") ns = val
                else if (key == "cycles") cycles = val
                else if (key == "instr") instr = val
                else if (key == "ipc") ipc = val
                else if (key == "cpi") cpi = val
                else if (key == "br_miss") br = val
                else if (key == "cache_miss/kI") cache = val
                else if (key == "fe_stall") fe = val
                else if (key == "be_stall") be = val
            }

            printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n", experiment_name, name, ns, cycles, instr, ipc, cpi, br, cache, fe, be
        }
    ' "$output_file" >>"$rows_file"
}

run_example() {
    local example="$1"
    local dir="$root_dir/$example"
    local output_bin="${example}_bench"
    local output_log

    output_log="$(mktemp)"

    echo "== $example =="
    pushd "$dir" >/dev/null
    g++ "$opt_level" -mavx2 -std=c++17 main.cpp -o "$output_bin"
    taskset -c "$cpu_core" "./$output_bin" | tee "$output_log"
    popd >/dev/null
    echo

    parse_pmu_rows "$example" "$output_log"
    rm -f "$output_log"
}

for exp in "${experiments[@]}"; do
    run_example "$exp"
done

echo "== PMU Summary (core=$cpu_core, opt=$opt_level) =="
printf "%-16s %-20s %12s %12s %12s %8s %8s %10s %14s %10s %10s\n" \
    "experiment" "kernel" "ns" "cycles" "instr" "ipc" "cpi" "br_miss%" "cache_miss/kI" "fe_stall%" "be_stall%"
printf "%-16s %-20s %12s %12s %12s %8s %8s %10s %14s %10s %10s\n" \
    "----------------" "--------------------" "------------" "------------" "------------" "--------" "--------" "----------" "--------------" "----------" "----------"

while IFS='|' read -r exp kernel ns cycles instr ipc cpi br cache fe be; do
    printf "%-16s %-20s %12s %12s %12s %8s %8s %10s %14s %10s %10s\n" \
        "$exp" "$kernel" "$ns" "$cycles" "$instr" "$ipc" "$cpi" "$br" "$cache" "$fe" "$be"
done <"$rows_file"