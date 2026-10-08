#include "gemm.hpp"

#include <array>
#include <tuple>

int main() {
    // Include full tiles, fringes, all three blocking boundaries, and empty K.
    const std::array<std::tuple<int, int, int>, 16> cases{{
        {6, 16, 1}, {5, 15, 3}, {7, 17, 255},
        {6, 16, 256}, {6, 16, 257}, {6, 17, 513},
        {95, 127, 257}, {96, 128, 256}, {97, 129, 257},
        {100, 130, 512}, {512, 512, 512}, {1, 1, 0}, {7, 17, 0},
        {1, 1, 257}, {0, 7, 1}, {7, 0, 1},
    }};

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int failed = 0;

    for (const auto& [M, N, K] : cases) {
        // Padded strides also check that neither the inputs nor C are
        // assumed to have tightly packed rows.
        const int lda = K + 3;
        const int ldb = N + 5;
        const int ldc = N + 7;
        std::vector<float> A(static_cast<size_t>(std::max(M, 1)) * lda);
        std::vector<float> B(static_cast<size_t>(std::max(K, 1)) * ldb);
        std::vector<float> C(static_cast<size_t>(std::max(M, 1)) * ldc, 12345.0f);
        std::vector<float> expected = C;

        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        gemm_reference(M, N, K, A.data(), lda, B.data(), ldb,
                       expected.data(), ldc);
        gemm(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);

        float max_error = 0.0f;
        int mismatches = 0;
        for (size_t p = 0; p < C.size(); ++p) {
            const float error = std::fabs(C[p] - expected[p]);
            max_error = std::max(max_error, error);
            // Unwritten padding must match exactly; arithmetic allows
            // small differences from FMA versus separate operations.
            const bool padding = static_cast<int>(p % ldc) >= N ||
                                 static_cast<int>(p / ldc) >= M;
            if (!(padding ? C[p] == expected[p] :
                  error <= 2e-4f + 2e-4f * std::fabs(expected[p]))) {
                ++mismatches;
            }
        }

        std::cout << "M=" << M << " N=" << N << " K=" << K
                  << " max_error=" << max_error
                  << " mismatches=" << mismatches << '\n';
        failed += mismatches != 0;
    }

    std::cout << (failed ? "FAIL" : "PASS") << ": " << failed
              << "/" << cases.size() << " cases failed\n";
    return failed ? 1 : 0;
}