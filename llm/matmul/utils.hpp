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
