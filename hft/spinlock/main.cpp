#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <numeric>

#ifdef __linux__
#include <sched.h>
#endif

// === 锁的实现 (同前) ===
class alignas(64) HFTSpinlock {
    std::atomic<bool> locked_{false};
    inline void cpu_pause() const noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause(); 
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
    std::mutex mtx_;
public:
    void lock() { mtx_.lock(); }
    void unlock() { mtx_.unlock(); }
};

// === 绑核函数 ===
void pin_thread(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// === 核心测试逻辑 ===
template <typename LockType>
std::vector<uint64_t> run_latency_test(int thread_id, int num_threads, int iterations) {
    // 注意：这里为了演示，我们将锁和状态设为全局或外部传入，这里简化为局部静态
    // 实际测试中应确保多线程共享同一个 Lock 和 State
    static LockType shared_lock;
    static alignas(64) uint64_t shared_state = 0;

    // 强制绑核：假设前两个核留给 OS，工作线程从 Core 2 开始绑定
    pin_thread(thread_id + 2); 

    std::vector<uint64_t> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        shared_lock.lock();
        // 模拟极小的临界区业务逻辑
        shared_state += 1; 
        shared_lock.unlock();

        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    return latencies;
}

void print_stats(const std::string& name, std::vector<uint64_t>& lats) {
    std::sort(lats.begin(), lats.end());
    uint64_t sum = std::accumulate(lats.begin(), lats.end(), 0ULL);
    double avg = (double)sum / lats.size();
    
    std::cout << "[" << name << "]\n";
    std::cout << "  Avg Latency : " << avg << " ns\n";
    std::cout << "  P50 Latency : " << lats[lats.size() * 0.50] << " ns\n";
    std::cout << "  P99 Latency : " << lats[lats.size() * 0.99] << " ns\n"; // HFT 最看重的指标
    std::cout << "  P99.9 Latency: " << lats[lats.size() * 0.999] << " ns\n";
    std::cout << "  Max Latency : " << lats.back() << " ns\n\n";
}

int main() {
    const int THREADS = 4;
    const int ITERS = 100000; // 每个线程 10 万次

    std::cout << "=== HFT vs Mutex Latency Test (Pinned) ===\n\n";

    // 测试 HFT Spinlock
    std::vector<std::vector<uint64_t>> hft_results(THREADS);
    std::vector<std::thread> hft_threads;
    for(int i=0; i<THREADS; ++i) {
        hft_threads.emplace_back([&](int id){ hft_results[id] = run_latency_test<HFTSpinlock>(id, THREADS, ITERS); }, i);
    } 
    for(auto& t : hft_threads) {
        t.join();
    }
    
    std::vector<uint64_t> all_hft_lats;
    for(auto& v : hft_results) {
        all_hft_lats.insert(all_hft_lats.end(), v.begin(), v.end());
    }
    print_stats("HFT Spinlock", all_hft_lats);

    // 测试 OS Spinlock
    std::vector<std::vector<uint64_t>> os_results(THREADS);
    std::vector<std::thread> os_threads;
    for(int i=0; i<THREADS; ++i) {
        os_threads.emplace_back([&](int id){ os_results[id] = run_latency_test<PthreadSpinlock>(id, THREADS, ITERS); }, i);
    }
    for(auto& t : os_threads) {
        t.join();
    }
    std::vector<uint64_t> all_os_lats;
    for(auto& v : os_results) {
        all_os_lats.insert(all_os_lats.end(), v.begin(), v.end());
    }
    print_stats("OS Spinlock", all_os_lats);

    // 测试 Mutex
    std::vector<std::vector<uint64_t>> mtx_results(THREADS);
    std::vector<std::thread> mtx_threads;
    for(int i=0; i<THREADS; ++i) {
        mtx_threads.emplace_back([&](int id){ mtx_results[id] = run_latency_test<StdMutexWrapper>(id, THREADS, ITERS); }, i);
    }
    for(auto& t : mtx_threads) {
        t.join();
    }
    
    std::vector<uint64_t> all_mtx_lats;
    for(auto& v : mtx_results) {
        all_mtx_lats.insert(all_mtx_lats.end(), v.begin(), v.end());
    }
    print_stats("std::mutex", all_mtx_lats);

    return 0;
}
