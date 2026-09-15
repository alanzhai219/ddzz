#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename Key, typename Value>
class MemCache {
public:
    template <typename Factory>
    std::shared_ptr<Value> get_or_create(const Key& key, Factory&& factory) {
        {
            std::shared_lock lock(mutex);
            if (const auto iterator = cache.find(key); iterator != cache.end())
                return iterator->second;
        }

        // Do expensive construction without blocking cache readers.
        auto value = std::invoke(std::forward<Factory>(factory));

        // Another thread may have inserted the same key in the meantime.
        std::unique_lock lock(mutex);
        const auto iterator = cache.try_emplace(key, std::move(value)).first;
        return iterator->second;
    }

private:
    mutable std::shared_mutex mutex;
    std::unordered_map<Key, std::shared_ptr<Value>> cache;
};

struct User {
    int id;
    std::string name;
};

int main() {
    MemCache<std::string, User> cache;
    std::atomic<int> factory_calls{0};

    auto load_user = [&] {
        ++factory_calls;
        std::cout << "factory: loading user\n";
        return std::make_shared<User>(User{1, "Alice"});
    };

    const auto first = cache.get_or_create("user:1", load_user);  // Miss.
    const auto second = cache.get_or_create("user:1", load_user); // Hit.

    std::cout << "user=" << second->name << '\n';
    std::cout << "same object=" << std::boolalpha << (first == second) << '\n';
    std::cout << "factory calls=" << factory_calls << "\n\n";

    // Concurrent misses are safe. Factories may run more than once, but all
    // callers receive the single value that wins insertion into the cache.
    std::vector<std::shared_ptr<User>> results(4);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i] {
            results[i] = cache.get_or_create("user:2", [&, i] {
                ++factory_calls;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                return std::make_shared<User>(User{2, "candidate-" + std::to_string(i)});
            });
        });
    }

    for (auto& thread : threads)
        thread.join();

    const auto cached = results.front();
    for (const auto& result : results) {
        if (result != cached)
            return 1;
    }

    std::cout << "concurrent cached user=" << cached->name << '\n';
    std::cout << "all threads received the same object=true\n";
    std::cout << "total factory calls=" << factory_calls << '\n';
}
