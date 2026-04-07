#include <xbyak/xbyak.h>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

// 教学版：只保留最核心的三个模板参数
// KT        : JIT 生成器类型
// KernelFunc: 导出的函数指针类型
// KeyTs...  : 缓存键类型，可以是一个或多个参数
template <typename KernelType, typename KernelFunc, typename... KeyTs>
class KernelEngine {
private:
    static_assert(std::is_base_of<Xbyak::CodeGenerator, KernelType>::value,
            "KernelType must derive from Xbyak::CodeGenerator");
    static_assert(sizeof...(KeyTs) > 0,
            "KernelEngine needs at least one key type");

    using key_type = std::tuple<std::decay_t<KeyTs>...>;
    using jit_ptr_t = std::unique_ptr<KernelType>;

    std::map<key_type, jit_ptr_t> cache_;
    size_t max_cache_size_;

public:
    explicit KernelEngine(size_t max_cache_size = 100)
        : max_cache_size_(max_cache_size) {}

    // 唯一接口：keys 同时用于缓存键和 KT(keys...) 构造参数
    KernelFunc getOrCompile(const KeyTs &...keys) {
        const key_type key(keys...);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            std::cout << "[Cache Hit]" << std::endl;
            return it->second->template getCode<KernelFunc>();
        }

        std::cout << "[Cache Miss] Compiling..." << std::endl;

        auto jit = std::make_unique<KernelType>(keys...);
        KernelFunc func = jit->template getCode<KernelFunc>();

        if (cache_.size() >= max_cache_size_) {
            cache_.clear(); 
        }
        cache_.emplace(key, std::move(jit));
        return func;
    }

    size_t size() const { return cache_.size(); }
    void clear() { cache_.clear(); }
};