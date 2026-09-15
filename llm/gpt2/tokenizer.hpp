#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tk {

// GPT-2 byte-level BPE tokenizer. It matches the reference tokenizer for ASCII
// text. Unicode letter/number classification is a best-effort approximation
// because this project deliberately has no Unicode-property dependency.
class Tokenizer {
public:
    // Loads encoder.json (token -> id) and vocab.bpe (merge ranks).
    Tokenizer(const std::string& encoder_json_path,
              const std::string& vocab_bpe_path);

    std::vector<int> encode(const std::string& text) const;
    // Throws std::out_of_range when an ID does not belong to this vocabulary.
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }

private:
    // Split raw text into GPT-2 pre-tokens (the regex stage).
    std::vector<std::string> pretokenize(const std::string& text) const;
    // Apply BPE merges to one byte-encoded word, returning the final symbols.
    std::vector<std::string> bpe(const std::string& token) const;

    std::unordered_map<std::string, int> token_to_id_;
    // Token IDs in GPT-2 are dense. An indexed table makes decoding faster and
    // validates malformed encoder files during construction.
    std::vector<std::string> id_to_token_;

    // (first, second) pair -> merge rank (lower = merged earlier).
    std::unordered_map<std::string, int> bpe_ranks_;

    // Pre-tokens repeat constantly in real text, so caching their merge result
    // avoids redoing the same BPE work. This makes encode() non-thread-safe.
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    // byte (0-255) -> byte-level unicode char (UTF-8 encoded), and the reverse.
    // Byte-level codepoints are all below 512, so a dense table beats a map.
    std::array<std::string, 256> byte_encoder_;
    std::array<int16_t, 512> byte_decoder_;
};

}  // namespace tk
