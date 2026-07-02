#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_with_fixed_freq.sh --freq-khz <kHz> -- <command> [args...]

Example:
  ./run_with_fixed_freq.sh --freq-khz 3000000 -- taskset -c 1 ./compare_blend_bench

Notes:
  - Requires sudo privileges for writing cpufreq/turbo sysfs settings.
  - Automatically restores original settings on exit (success or failure).
EOF
}

log() {
    echo "[fixed-freq] $*"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing required command: $cmd" >&2
        exit 1
    fi
}

if [[ $# -lt 4 ]]; then
    usage
    exit 1
fi

FREQ_KHZ=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --freq-khz)
            shift
            FREQ_KHZ="${1:-}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$FREQ_KHZ" ]]; then
    echo "--freq-khz is required" >&2
    usage
    exit 1
fi

if ! [[ "$FREQ_KHZ" =~ ^[0-9]+$ ]]; then
    echo "--freq-khz must be an integer in kHz" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "missing command after --" >&2
    usage
    exit 1
fi

require_cmd sudo

POLICIES=(/sys/devices/system/cpu/cpufreq/policy*)
if [[ ! -e "${POLICIES[0]}" ]]; then
    echo "cpufreq policy paths not found under /sys/devices/system/cpu/cpufreq" >&2
    exit 1
fi

BACKUP_DIR="$(mktemp -d /tmp/fixed-freq-backup.XXXXXX)"
RESTORED=0

TURBO_PATH=""
if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
elif [[ -e /sys/devices/system/cpu/cpufreq/boost ]]; then
    TURBO_PATH="/sys/devices/system/cpu/cpufreq/boost"
fi

restore_settings() {
    if [[ "$RESTORED" -eq 1 ]]; then
        return
    fi
    RESTORED=1

    log "restoring CPU frequency settings"
    for p in "${POLICIES[@]}"; do
        local policy
        policy="$(basename "$p")"
        local gov_file="$BACKUP_DIR/${policy}.governor"
        local min_file="$BACKUP_DIR/${policy}.min"
        local max_file="$BACKUP_DIR/${policy}.max"

        if [[ -f "$gov_file" && -w "$p/scaling_governor" ]]; then
            sudo tee "$p/scaling_governor" >/dev/null <"$gov_file" || true
        fi
        if [[ -f "$min_file" && -w "$p/scaling_min_freq" ]]; then
            sudo tee "$p/scaling_min_freq" >/dev/null <"$min_file" || true
        fi
        if [[ -f "$max_file" && -w "$p/scaling_max_freq" ]]; then
            sudo tee "$p/scaling_max_freq" >/dev/null <"$max_file" || true
        fi
    done

    if [[ -n "$TURBO_PATH" && -f "$BACKUP_DIR/turbo" && -w "$TURBO_PATH" ]]; then
        sudo tee "$TURBO_PATH" >/dev/null <"$BACKUP_DIR/turbo" || true
    fi

    rm -rf "$BACKUP_DIR"
}

trap restore_settings EXIT INT TERM

log "saving current CPU frequency settings to $BACKUP_DIR"
for p in "${POLICIES[@]}"; do
    policy="$(basename "$p")"
    cat "$p/scaling_governor" >"$BACKUP_DIR/${policy}.governor"
    cat "$p/scaling_min_freq" >"$BACKUP_DIR/${policy}.min"
    cat "$p/scaling_max_freq" >"$BACKUP_DIR/${policy}.max"
done

if [[ -n "$TURBO_PATH" ]]; then
    cat "$TURBO_PATH" >"$BACKUP_DIR/turbo"
fi

log "applying fixed-frequency settings: governor=performance freq=${FREQ_KHZ}kHz"
for p in "${POLICIES[@]}"; do
    policy="$(basename "$p")"
    hw_min="$(cat "$p/cpuinfo_min_freq")"
    hw_max="$(cat "$p/cpuinfo_max_freq")"
    if (( FREQ_KHZ < hw_min || FREQ_KHZ > hw_max )); then
        echo "requested freq ${FREQ_KHZ} out of range for ${policy}: [${hw_min}, ${hw_max}]" >&2
        exit 1
    fi

    if [[ -w "$p/scaling_governor" ]]; then
        echo performance | sudo tee "$p/scaling_governor" >/dev/null
    fi
    if [[ -w "$p/scaling_min_freq" ]]; then
        echo "$FREQ_KHZ" | sudo tee "$p/scaling_min_freq" >/dev/null
    fi
    if [[ -w "$p/scaling_max_freq" ]]; then
        echo "$FREQ_KHZ" | sudo tee "$p/scaling_max_freq" >/dev/null
    fi
done

if [[ -n "$TURBO_PATH" && -w "$TURBO_PATH" ]]; then
    if [[ "$TURBO_PATH" == *"no_turbo" ]]; then
        echo 1 | sudo tee "$TURBO_PATH" >/dev/null
        log "turbo disabled via $TURBO_PATH=1"
    else
        echo 0 | sudo tee "$TURBO_PATH" >/dev/null
        log "boost disabled via $TURBO_PATH=0"
    fi
fi

log "running command: $*"
"$@"
