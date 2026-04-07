// example.cpp — Demonstrates the three main ways to use SimplePerf.
//
// Build & run:
//   g++ -O2 -std=c++11 -o example example.cpp
//   ./example
//
// If perf_event_open fails, relax the paranoid setting:
//   echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

#include "simple_perf.h"
#include <cmath>
#include <numeric>
#include <vector>

// A small workload: dot-product of two float vectors.
static float dot(const float* a, const float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

// -------------------------------------------------------------------
// 1) Basic start/stop with default counters (cycles + instructions)
// -------------------------------------------------------------------
static void demo_basic() {
    printf("=== Demo 1: basic start/stop ===\n");

    const int N = 1024 * 1024;
    std::vector<float> a(N, 1.0f), b(N, 2.0f);

    SimplePerf perf;          // default: CPU_CYCLES + INSTRUCTIONS
    perf.start();
    volatile float result = dot(a.data(), b.data(), N);
    auto& evs = perf.stop();
    (void)result;

    printf("  duration   : %lu ns\n",  (unsigned long)evs.at("ns"));
    printf("  cycles     : %lu\n",     (unsigned long)evs.at("C"));
    printf("  instructions: %lu\n",    (unsigned long)evs.at("I"));
    printf("  CPI        : %.2f\n\n",  (double)evs.at("C") / evs.at("I"));
}

// -------------------------------------------------------------------
// 2) Custom event set (add cache-misses and branch-misses)
// -------------------------------------------------------------------
static void demo_custom_events() {
    printf("=== Demo 2: custom events ===\n");

    const int N = 4 * 1024 * 1024;
    std::vector<float> a(N), b(N);
    // Initialize with some values so the branch predictor has work
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(N - i);
    }

    SimplePerf perf({
        {"CPU_CYCLES",    0},
        {"INSTRUCTIONS",  0},
        {"CACHE_MISSES",  0},
        {"BRANCH_MISSES", 0},
    });

    perf.start();
    volatile float result = dot(a.data(), b.data(), N);
    auto& evs = perf.stop();
    (void)result;

    for (auto& kv : evs)
        printf("  %-22s : %lu\n", kv.first.c_str(), (unsigned long)kv.second);
    printf("\n");
}

// -------------------------------------------------------------------
// 3) profile() helper — auto-prints per-iteration stats
// -------------------------------------------------------------------
static void demo_profile_helper() {
    printf("=== Demo 3: profile() helper ===\n");

    const int N = 1024 * 1024;
    std::vector<float> a(N, 1.0f), b(N, 2.0f);

    // 2 FLOPs per element (multiply + add), N elements
    double total_ops  = 2.0 * N;           // in FLOPs
    double total_bytes = 2.0 * N * sizeof(float); // bytes read

    SimplePerf perf;
    perf.profile(
        [&]() { volatile float r = dot(a.data(), b.data(), N); (void)r; },
        "dot",
        /* n_repeats  = */ 5,
        /* loop_count = */ 1,
        /* ops        = */ total_ops,
        /* bytes      = */ total_bytes
    );
}

int main() {
    demo_basic();
    demo_custom_events();
    demo_profile_helper();
    return 0;
}
