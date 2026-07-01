#include <stdio.h>

#include "../common/benchmark.hpp"

#define N 100000
#define ROUNDS 5000

static void fill_random_ints(int *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = rand() - RAND_MAX / 2;
    }
}

static size_t compact_branchy(const int *input, int *output, size_t length) {
    size_t count = 0;
    for (size_t i = 0; i < length; i++) {
        if (input[i] >= 0) {
            output[count++] = input[i];
        }
    }
    return count;
}

static size_t compact_two_phase(const int *input, int *output, size_t length) {
    unsigned char *flags = branchless_memory_layout::checked_malloc<unsigned char>(length);
    size_t *positions = branchless_memory_layout::checked_malloc<size_t>(length);

    size_t count = 0;
    for (size_t i = 0; i < length; i++) {
        flags[i] = (unsigned char)(input[i] >= 0);
        positions[i] = count;
        count += (size_t)flags[i];
    }

    for (size_t i = 0; i < length; i++) {
        output[positions[i]] = input[i];
    }

    free(flags);
    free(positions);
    return count;
}
int main() {
    srand(42);

    int *input = branchless_memory_layout::checked_malloc<int>(N);
    int *branchy = branchless_memory_layout::checked_malloc<int>(N);
    int *two_phase = branchless_memory_layout::checked_malloc<int>(N);

    fill_random_ints(input, N);

    size_t branchy_count = compact_branchy(input, branchy, N);
    size_t two_phase_count = compact_two_phase(input, two_phase, N);
    branchless_memory_layout::validate_equal_size(branchy_count, two_phase_count);
    branchless_memory_layout::validate_equal_buffers(branchy, two_phase, branchy_count);

    double branchy_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() { branchy_count = compact_branchy(input, branchy, N); });

    double two_phase_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() { two_phase_count = compact_two_phase(input, two_phase, N); });

    printf("two-phase compact example\n");
    printf("  branchy:   %.3f s\n", branchy_seconds);
    printf("  two-phase: %.3f s\n", two_phase_seconds);
    printf("  speedup:   %.2fx\n", branchy_seconds / two_phase_seconds);
    printf("  kept:      %zu\n", two_phase_count);

    free(input);
    free(branchy);
    free(two_phase);
    return 0;
}