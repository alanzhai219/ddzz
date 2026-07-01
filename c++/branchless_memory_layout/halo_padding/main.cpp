#include <stdio.h>
#include <string.h>

#include "../common/benchmark.hpp"

#define N 100000
#define ROUNDS 5000

static void fill_random_ints(int *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = rand() % 100;
    }
}

static void stencil_branchy(const int *input, int *output, size_t length) {
    for (size_t i = 0; i < length; i++) {
        int left = (i == 0) ? 0 : input[i - 1];
        int center = input[i];
        int right = (i + 1 == length) ? 0 : input[i + 1];
        output[i] = left + center + right;
    }
}

static void stencil_halo_padded(const int *input, int *output, size_t length) {
    int *padded = branchless_memory_layout::checked_malloc<int>(length + 2);

    padded[0] = 0;
    memcpy(padded + 1, input, length * sizeof(int));
    padded[length + 1] = 0;

    for (size_t i = 0; i < length; i++) {
        output[i] = padded[i] + padded[i + 1] + padded[i + 2];
    }

    free(padded);
}
int main() {
    srand(42);

    int *input = branchless_memory_layout::checked_malloc<int>(N);
    int *branchy = branchless_memory_layout::checked_malloc<int>(N);
    int *halo = branchless_memory_layout::checked_malloc<int>(N);

    fill_random_ints(input, N);

    stencil_branchy(input, branchy, N);
    stencil_halo_padded(input, halo, N);
    branchless_memory_layout::validate_equal_buffers(branchy, halo, N);

    double branchy_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() { stencil_branchy(input, branchy, N); });

    double halo_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() { stencil_halo_padded(input, halo, N); });

    printf("halo padding example\n");
    printf("  branchy: %.3f s\n", branchy_seconds);
    printf("  halo:    %.3f s\n", halo_seconds);
    printf("  speedup: %.2fx\n", branchy_seconds / halo_seconds);

    free(input);
    free(branchy);
    free(halo);
    return 0;
}