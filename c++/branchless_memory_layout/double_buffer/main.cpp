#include <stdio.h>
#include <string.h>

#include "../common/benchmark.hpp"

#define N 100000
#define ROUNDS 3000

static void fill_random_bits(int *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = rand() & 1;
    }
}

static void step_branchy_commit(int *state, const int *snapshot, size_t length) {
    state[0] = snapshot[0];
    for (size_t i = 1; i + 1 < length; i++) {
        int next_value = (snapshot[i - 1] + snapshot[i] + snapshot[i + 1] >= 2) ? 1 : 0;
        if (next_value != state[i]) {
            state[i] = next_value;
        }
    }
    state[length - 1] = snapshot[length - 1];
}

static void step_double_buffer(const int *current, int *next, size_t length) {
    next[0] = current[0];
    for (size_t i = 1; i + 1 < length; i++) {
        next[i] = (current[i - 1] + current[i] + current[i + 1] >= 2) ? 1 : 0;
    }
    next[length - 1] = current[length - 1];
}

int main() {
    srand(42);

    int *initial = branchless_memory_layout::checked_malloc<int>(N);
    int *branchy_state = branchless_memory_layout::checked_malloc<int>(N);
    int *snapshot = branchless_memory_layout::checked_malloc<int>(N);
    int *current = branchless_memory_layout::checked_malloc<int>(N);
    int *next = branchless_memory_layout::checked_malloc<int>(N);

    fill_random_bits(initial, N);
    memcpy(branchy_state, initial, N * sizeof(int));
    memcpy(current, initial, N * sizeof(int));

    for (int round = 0; round < ROUNDS; round++) {
        memcpy(snapshot, branchy_state, N * sizeof(int));
        step_branchy_commit(branchy_state, snapshot, N);
    }

    for (int round = 0; round < ROUNDS; round++) {
        step_double_buffer(current, next, N);
        int *tmp = current;
        current = next;
        next = tmp;
    }

    branchless_memory_layout::validate_equal_buffers(branchy_state, current, N);

    memcpy(branchy_state, initial, N * sizeof(int));
    memcpy(current, initial, N * sizeof(int));

    double branchy_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() {
            memcpy(snapshot, branchy_state, N * sizeof(int));
            step_branchy_commit(branchy_state, snapshot, N);
        });

    double double_buffer_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() {
            step_double_buffer(current, next, N);
            int *tmp = current;
            current = next;
            next = tmp;
        });

    printf("double-buffer example\n");
    printf("  branchy-commit: %.3f s\n", branchy_seconds);
    printf("  double-buffer:  %.3f s\n", double_buffer_seconds);
    printf("  speedup:        %.2fx\n", branchy_seconds / double_buffer_seconds);
    printf("  final center:   %d\n", current[N / 2]);

    free(initial);
    free(branchy_state);
    free(snapshot);
    free(current);
    free(next);
    return 0;
}