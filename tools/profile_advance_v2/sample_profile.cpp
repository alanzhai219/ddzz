#include "linux_perf_advance.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

double run_workload(std::vector<float>& values, int rounds) {
    double sum = 0.0;
    for (int round = 0; round < rounds; ++round) {
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = std::sin(values[index] + round * 0.001f)
                          + values[index] * 1.0001f;
            sum += values[index];
        }
    }
    return sum;
}

}  // namespace

int main() {
    std::vector<float> values(1U << 18U);
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index % 251U) * 0.125f;
    }

    double result = 0.0;
    {
        LINUX_PERF_PROFILE("vector_sin_update");
        result = run_workload(values, 32);
    }

    std::cout << "result: " << result << '\n';
    return result == 0.0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
