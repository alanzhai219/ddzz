#include <cstdint>
#include <cstddef>

// 1. 【被测函数】：这里故意留了一个越界读取漏洞
void check_password(const uint8_t* data, size_t size) {
    // 如果输入的前4个字节是 "PASS"
    if (size >= 4 && data[0] == 'P' && data[1] == 'A' && data[2] == 'S' && data[3] == 'S') {
        
        // 🚨 漏洞：开发者忘记检查 size 是否 >= 11，就直接访问了 data[10]
        // 如果输入的长度只有 5 (例如 "PASS1")，访问 data[10] 就会发生越界读取！
        if (data[10] == 'X') {
            // 模拟触发某个深层逻辑
        }
    }
}

// 2. 【Fuzzer 入口】：libFuzzer 唯一认识的函数签名
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // 直接把 libFuzzer 生成的随机字节流喂给被测函数
    check_password(data, size);
    
    // 必须返回 0
    return 0;
}
