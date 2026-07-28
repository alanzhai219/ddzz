#include <iostream>
#include <vector>

#include <immintrin.h>

#include "utils.hpp"

template <typename T, size_t M, size_t N, size_t K>
void mat_mul_outer_product(const T* A, const T* B, T* C) {

    for (int k = 0; k < K; k++) {           // 遍历内部维度 K
        for (int i = 0; i < M; i++) {       // 遍历 A 的行
            float a_ik = A[i * K + k];      // 取出一个标量
            for (int j = 0; j < N; j++) {  // 遍历 B 的行 (连续访存！)
                // 优势：B 和 C 都是按行连续访问，Cache 命中率极高
                C[i * N + j] += a_ik * B[k * N + j]; 
            }
        }
    }
}

template <typename T, size_t M, size_t N, size_t K>
void mat_mul_outer_product_avx2(const T* A, const T* B, T* C) {
   for (size_t k = 0; k < K; ++k) {
      for (size_t m = 0; m < M; ++m) {
         // auto a_value = A[m * K + k];
         auto ymm_a = _mm256_broadcast_ss(A + m * K + k);
         for (size_t n = 0; n < N; n+=256) {
            auto ymm_b = _mm256_loadu_ps(B + k * N + n);
            // TODO
         }
      }
   }
}

int main() {
   const size_t M = 4;
   const size_t N = 16;
   const size_t K = 8;

   std::vector<float> mat_A(M*K, 0.0F);
   std::vector<float> mat_B(K*N, 0.0F);
   std::vector<float> mat_C(M*N, 0.0F);

   fill_rand<float, M*K>(mat_A.data());
   fill_rand<float, K*N>(mat_B.data());

   mat_mul_outer_product<float, M, N, K>(mat_A.data(), mat_B.data(), mat_C.data());

   std::cout << mat_C[0] << ", " << mat_C[1] << "\n";

   std::vector<float> mat_C2(M*N, 0.0F);
   mat_mul_outer_product_avx2<float, M, N, K>(mat_A.data(), mat_B.data(), mat_C2.data());
   std::cout << mat_C2[0] << ", " << mat_C2[1] << "\n";

   return 0;
}
