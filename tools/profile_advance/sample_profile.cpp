#include "linux_perf_advance.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

namespace {

double run_workload(std::vector<float>& data, int rounds) {
    double total = 0.0;
    for (int round = 0; round < rounds; ++round) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::sin(data[i] + static_cast<float>(round) * 0.001f) + data[i] * 1.0001f;
            total += data[i];
        }
    }
    return total;
}

void print_selected_counters(const std::map<std::string, uint64_t>& counters) {
    const char* keys[] = {
        "HW_CPU_CYCLES",
        "HW_INSTRUCTIONS",
        "HW_CACHE_MISSES",
        "SW_CONTEXT_SWITCHES",
        "SW_TASK_CLOCK",
        "SW_PAGE_FAULTS"
    };

    for (const char* key : keys) {
        std::map<std::string, uint64_t>::const_iterator it = counters.find(key);
        if (it != counters.end()) {
            std::cout << key << ": " << it->second << std::endl;
        }
    }
}

}  // namespace

int main() {
    LinuxPerf::Init();

    const size_t element_count = 1 << 18;
    const int rounds = 32;
    std::vector<float> data(element_count);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 251) * 0.125f;
    }

    std::map<std::string, uint64_t> counters;
    double result = 0.0;

    {
        LinuxPerf::ProfileScope scope = LinuxPerf::Profile(
                "vector_sin_update",
                std::string("teaching"),
                0,
                static_cast<int64_t>(element_count),
                rounds,
                3.1415926);
        result = run_workload(data, rounds);
        scope.finish(&counters);
    }

    std::cout << "result = " << result << std::endl;
    print_selected_counters(counters);

    return (result == 0.0) ? EXIT_FAILURE : EXIT_SUCCESS;
}