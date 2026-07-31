template <typename T, size_t N>
void fill_rand(T* v) {
  for (size_t i =0; i < N; ++i) {
    v[i] = static_cast<T>(i);
  }
}

template <typename T, size_t X, size_t Y>
void transpose(const T* src, T* dst) {
  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {
      dst[x * Y + y] = src[y * X + x];
    }
  }
}

inline float hsum256_ps(__m256 v) {
    // 合并低/高 128-bit lane：
    // [v0+v4, v1+v5, v2+v6, v3+v7, ...]
    v = _mm256_add_ps(v, _mm256_permute2f128_ps(v, v, 0x01));

    // 在 128-bit lane 内继续水平相加
    v = _mm256_hadd_ps(v, v);
    v = _mm256_hadd_ps(v, v);

    return _mm_cvtss_f32(_mm256_castps256_ps128(v));
}
