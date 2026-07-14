#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <immintrin.h>
#include <string>

// ==========================================
// 1. 自定义 HFT 自旋锁 (TTAS + Pause)
// ==========================================
class alignas(64) HFTSpinlock {
private:
    std::atomic<bool> locked_{false};
    inline void cpu_pause() const noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _mm_pause(); 
#elif defined(__aarch64__) || defined(__arm__)
        asm volatile("yield" ::: "memory");
#endif
    }
public:
    void lock() noexcept {
        if (locked_.exchange(true, std::memory_order_acquire)) {
            while (true) {
                while (locked_.load(std::memory_order_relaxed)) { cpu_pause(); }
                if (!locked_.exchange(true, std::memory_order_acquire)) return;
            }
        }
    }
    void unlock() noexcept { locked_.store(false, std::memory_order_release); }
};

// ==========================================
// 2. 系统自旋锁 (pthread_spinlock)
// ==========================================
class alignas(64) PthreadSpinlock {
private:
    pthread_spinlock_t spin_;
public:
    PthreadSpinlock() { pthread_spin_init(&spin_, PTHREAD_PROCESS_PRIVATE); }
    ~PthreadSpinlock() { pthread_spin_destroy(&spin_); }
    void lock() { pthread_spin_lock(&spin_); }
    void unlock() { pthread_spin_unlock(&spin_); }
};

// ==========================================
// 3. 标准互斥锁 (std::mutex)
// ==========================================
class alignas(64) StdMutexWrapper {
private:
    std::mutex mtx_;
public:
    void lock() { mtx_.lock(); }
    void unlock() { mtx_.unlock(); }
};

// ==========================================
// 辅助组件：RAII Guard 与 共享状态
// ==========================================
template <typename LockType>
class Guard {
    LockType& lock_;
public:
    explicit Guard(LockType& lock) : lock_(lock) { lock_.lock(); }
    ~Guard() { lock_.unlock(); }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
};

struct alignas(64) SharedState {
    int counter = 0;
};

// ==========================================
// 核心测试函数
// ==========================================
template <typename LockType>
void run_benchmark(const std::string& name, int num_threads, int iterations) {
    LockType lock;
    SharedState state;
    std::vector<std::thread> threads;

    auto worker = [&]() {
        for (int i = 0; i < iterations; ++i) {
            Guard<LockType> guard(lock);
            state.counter++; // 极小临界区
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    double avg_ns = (double)duration_ns / (num_threads * iterations);

    std::cout << "[" << name << "]\n";
    std::cout << "  Total Ops: " << (num_threads * iterations) << "\n";
    std::cout << "  Total Time: " << (duration_ns / 1000000.0) << " ms\n";
    std::cout << "  Avg Time / Op: " << avg_ns << " ns\n";
    std::cout << "  Final Counter: " << state.counter << "\n\n";
}

int main() {
    const int NUM_THREADS = 4;         // 模拟 4 个交易线程激烈竞争
    const int ITERATIONS = 2000000;    // 每个线程执行 200 万次

    std::cout << "=== Lock Performance Benchmark ===\n";
    std::cout << "Threads: " << NUM_THREADS << ", Iterations/Thread: " << ITERATIONS << "\n\n";

    // 预热 CPU
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    run_benchmark<HFTSpinlock>("1. Custom HFT Spinlock (TTAS + Pause)", NUM_THREADS, ITERATIONS);
    run_benchmark<PthreadSpinlock>("2. OS Pthread Spinlock (TAS)", NUM_THREADS, ITERATIONS);
    run_benchmark<StdMutexWrapper>("3. std::mutex (Futex / Context Switch)", NUM_THREADS, ITERATIONS);

    return 0;
}
