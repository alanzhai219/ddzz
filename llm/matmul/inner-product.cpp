#include <iostream>
#include <vector>

#include <immintrin.h>

#include "utils.hpp"

// #include "xbyak/xbyak.h"

template <typename T, size_t M, size_t N, size_t K>
void mat_mul_dot_product(const T* A, const T* B, T* C) {
    for (int i = 0; i < M; i++) {           // 遍历 C 的行
        for (int j = 0; j < N; j++) {      // 遍历 C 的列
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {   // 内积的累加过程
                // 痛点：A 按行访问(连续)，但 B 按列访问(stride=16，极度不连续！)
                sum += A[i * K + k] * B[k * N+ j]; 
            }
            C[i * N + j] = sum;
        }
    }
}

template <typename T, size_t M, size_t N, size_t K>
void mat_mul_dot_product_avx2(const T* A, const T* B, T* C) {
   std::vector<T> B_trans(N*K, static_cast<T>(0));
   T* B_trans_ptr = B_trans.data();
   transpose<T, N, K>(B, B_trans_ptr);
   for (size_t m = 0; m < M; ++m) {
      auto ymm_a = _mm256_loadu_ps(A + m * 256);
      for (size_t n = 0; n < N; ++n) {
         auto ymm_b = _mm256_loadu_ps(B_trans_ptr + n * 256);
         // TODO
      }
   }
}

// TODO: how to make the pipeline full
/*
template <typename T, size_t M, size_t N, size_t K>
void mat_mul_dot_product_avx2(const T* A, const T* B, T* C) {
   std::vector<T> B_trans(N*K, static_cast<T>(0));
   T* B_trans_ptr = B_trans.data();
   transpose<T, N, K>(B, B_trans_ptr);
   for (size_t m = 0; m < M; ++m) {
      auto ymm_a = _mm256_loadu_ps(A + m * 256);
      for (size_t n = 0; n < N; ++n) {
         auto ymm_b = _mm256_loadu_ps(B_trans_ptr + n * 256);
         C[m * N + n] = _mm256_fmadd_ps(ymm_a, ymm_b);
      }
   }
}
*/

// jit version

int main() {
   const size_t M = 4;
   const size_t N = 16;
   const size_t K = 8;

   std::vector<float> mat_A(M*K, 0.0F);
   std::vector<float> mat_B(K*N, 0.0F);
   std::vector<float> mat_C(M*N, 0.0F);

   fill_rand<float, M*K>(mat_A.data());
   fill_rand<float, K*N>(mat_B.data());

   mat_mul_dot_product<float, M, N, K>(mat_A.data(), mat_B.data(), mat_C.data());

   std::cout << mat_C[0] << ", " << mat_C[1] << "\n";

   std::vector<float> mat_C2(M*N, 0.0F);
   mat_mul_dot_product<float, M, N, K>(mat_A.data(), mat_B.data(), mat_C2.data());
   std::cout << mat_C2[0] << ", " << mat_C2[1] << "\n";

   return 0;
}
