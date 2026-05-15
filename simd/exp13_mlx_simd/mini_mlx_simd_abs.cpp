#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <iostream>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*
 * 这个文件把 MLX 的 SIMD 设计压缩成一个最小可读版本，只保留 5 个核心概念：
 *
 * 1. Simd<T, N>
 *    对应 MLX 里的“向量寄存器抽象”。上层只知道一次处理 N 个元素，
 *    不需要知道底层是标量、AVX2 还是 Accelerate。
 *
 * 2. max_size<T>
 *    对应 MLX 的“当前类型一次最多并行多少个元素”。
 *    标量后端返回 1，AVX2 的 float32 返回 8。
 *
 * 3. load / store / abs
 *    这些是最底层 primitive。这里故意不用 intrinsic，
 *    只保留“批量处理 N 个元素”的抽象形状。
 *
 * 4. AbsOp functor
 *    对应 MLX 里的 unary_ops.h。算子并不直接操心寄存器细节，
 *    它只把“abs”映射到 simd::abs。
 *
 * 5. unary_contiguous
 *    对应 MLX 里的 unary.h。连续内存主循环会：
 *      load N 个元素 -> 调算子 -> store N 个结果
 *    尾巴不足 N 的部分退回标量。
 *
 * 和 MLX 代码的对照关系：
 *   - type.h              -> backend 选择
 *   - base_simd.h         -> 标量兜底实现
 *   - avx2_simd.h         -> 真正项目里会在这里接具体 CPU 指令
 *   - unary_ops.h         -> AbsOp 这种 functor
 *   - unary.h             -> contiguous 主循环
 */

namespace mini_mlx::simd {

template <typename T>
inline constexpr int max_size = 1;

// Simd is a data structure for packing multi values together.
// Then instructions like avx2, neon, accelerate can operate on the whole pack at once.
template <typename T, int N>
struct Simd {
  static constexpr int size = N;
  T lane[N];

  Simd() = default;

  template <typename U>
  Simd(Simd<U, N> other) {
    for (int i = 0; i < N; ++i) {
      lane[i] = static_cast<T>(other[i]);
    }
  }

  template <typename U>
  explicit Simd(U v) {
    for (T& x : lane) {
      x = static_cast<T>(v);
    }
  }

  T operator[](int idx) const {
    return lane[idx];
  }

  T& operator[](int idx) {
    return lane[idx];
  }
};

template <typename T, int N>
inline Simd<T, N> load(const T* src) {
  Simd<T, N> out;
  std::memcpy(&out, src, sizeof(out));
  return out;
}

template <typename T, int N>
inline void store(T* dst, Simd<T, N> x) {
  std::memcpy(dst, &x, sizeof(x));
}

template <typename T, int N>
inline Simd<T, N> abs(Simd<T, N> x) {
  Simd<T, N> out;
  using std::abs;
  for (int i = 0; i < N; ++i) {
    out[i] = abs(x[i]);
  }
  return out;
}

template <>
inline constexpr int max_size<float> = 8;

#if defined(__AVX2__)
template <>
inline Simd<float, 8> load(const float* src) {
  Simd<float, 8> out;
  _mm256_storeu_ps(out.lane, _mm256_loadu_ps(src));
  return out;
}

template <>
inline void store(float* dst, Simd<float, 8> x) {
  _mm256_storeu_ps(dst, _mm256_loadu_ps(x.lane));
}

template <>
inline Simd<float, 8> abs(Simd<float, 8> x) {
  Simd<float, 8> out;
  auto sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  auto value = _mm256_loadu_ps(x.lane);
  _mm256_storeu_ps(out.lane, _mm256_and_ps(value, sign_mask));
  return out;
}
#endif

} // namespace mini_mlx::simd

namespace mini_mlx {

#if defined(__GNUC__) && !defined(__clang__)
#define MINI_MLX_NOINLINE_SCALAR __attribute__((noinline, optimize("no-tree-vectorize")))
#elif defined(__clang__)
#define MINI_MLX_NOINLINE_SCALAR __attribute__((noinline))
#else
#define MINI_MLX_NOINLINE_SCALAR
#endif

struct BenchmarkResult {
  double best_ms;
  double avg_ms;
  double gib_per_s;
};

struct AbsOp {
  template <typename T, int N>
  simd::Simd<T, N> operator()(simd::Simd<T, N> x) const {
    return simd::abs(x);
  }

  template <typename T>
  T operator()(T x) const {
    using std::abs;
    return abs(x);
  }
};

template <typename T, typename Op>
void unary_contiguous(const T* src, T* dst, std::size_t size, Op op) {
  constexpr int N = simd::max_size<T>;

  while (size >= static_cast<std::size_t>(N)) {
    auto x = simd::load<T, N>(src);
    auto y = op(x);
    simd::store(dst, y);
    src += N;
    dst += N;
    size -= N;
  }

  while (size > 0) {
    *dst++ = op(*src++);
    --size;
  }
}

template <typename T>
MINI_MLX_NOINLINE_SCALAR
void scalar_abs_reference(const T* src, T* dst, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    dst[i] = std::abs(src[i]);
  }
}

template <typename Fn>
BenchmarkResult benchmark(Fn&& fn, std::size_t bytes_per_run, int warmup, int runs) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }

  double best_ms = std::numeric_limits<double>::max();
  double total_ms = 0.0;
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best_ms = std::min(best_ms, ms);
    total_ms += ms;
  }

  auto best_seconds = best_ms / 1000.0;
  auto gib_per_s = static_cast<double>(bytes_per_run) /
      (best_seconds * 1024.0 * 1024.0 * 1024.0);
  return {best_ms, total_ms / runs, gib_per_s};
}

BenchmarkResult benchmark_scalar(const std::vector<float>& input, std::vector<float>& out) {
  return benchmark(
      [&]() { scalar_abs_reference(input.data(), out.data(), input.size()); },
      input.size() * sizeof(float) * 2,
      5,
      30);
}

BenchmarkResult benchmark_simd(const std::vector<float>& input, std::vector<float>& out) {
  return benchmark(
      [&]() { unary_contiguous(input.data(), out.data(), input.size(), AbsOp{}); },
      input.size() * sizeof(float) * 2,
      5,
      30);
}

const char* backend_name() {
#if defined(__AVX2__)
  return "AVX2 backend float32 x8";
#else
  return "teaching backend float32 x8 (no intrinsic)";
#endif
}

void explain_design() {
  std::cout << "Design summary\n";
  std::cout << "- top abstraction: Simd<T, N>\n";
  std::cout << "- backend width for float: " << simd::max_size<float> << "\n";
  std::cout << "- current backend: " << backend_name() << "\n";
  std::cout << "- operator mapping: AbsOp -> simd::abs\n";
  std::cout << "- execution path: load -> abs -> store -> tail loop\n\n";
  std::cout << "- note: this file explains the design only; real MLX speedup comes from replacing\n";
  std::cout << "        this backend with AVX2/Accelerate/NEON while keeping the same upper layers\n\n";
}

} // namespace mini_mlx

int main() {
  using namespace mini_mlx;

  explain_design();

  constexpr std::size_t size = 1 << 20;
  std::vector<float> input(size);
  std::vector<float> scalar_out(size);
  std::vector<float> simd_out(size);

  for (std::size_t i = 0; i < size; ++i) {
    float x = static_cast<float>((static_cast<int>(i % 97) - 48)) * 0.25f;
    input[i] = (i % 3 == 0) ? -x : x;
  }

  auto scalar_stats = benchmark_scalar(input, scalar_out);
  auto simd_stats = benchmark_simd(input, simd_out);

  bool ok = std::equal(scalar_out.begin(), scalar_out.end(), simd_out.begin());

  std::cout << "First 8 outputs\n";
  for (int i = 0; i < 8; ++i) {
    std::cout << "  in=" << std::setw(7) << input[i]
              << "  abs=" << std::setw(7) << simd_out[i] << "\n";
  }

  std::cout << "\nBenchmark\n";
  std::cout << "- scalar reference best: " << scalar_stats.best_ms
            << " ms, avg: " << scalar_stats.avg_ms
            << " ms, throughput: " << scalar_stats.gib_per_s << " GiB/s\n";
  std::cout << "- current backend best: " << simd_stats.best_ms
            << " ms, avg: " << simd_stats.avg_ms
            << " ms, throughput: " << simd_stats.gib_per_s << " GiB/s\n";
  std::cout << "- speedup (best): " << (scalar_stats.best_ms / simd_stats.best_ms) << "x\n";
  std::cout << "- outputs match: " << (ok ? "yes" : "no") << "\n\n";

  return ok ? 0 : 1;
}