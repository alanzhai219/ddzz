#include "weights.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpt2 {
namespace {

class JsonCursor {
public:
    explicit JsonCursor(const std::string& text) : text_(text) {}

    void whitespace() { while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' || text_[pos_] == '\t')) ++pos_; }
    char peek() { whitespace(); return pos_ < text_.size() ? text_[pos_] : '\0'; }
    void expect(char ch) { whitespace(); if (pos_ >= text_.size() || text_[pos_] != ch) throw std::runtime_error("safetensors: malformed JSON header"); ++pos_; }

    std::string string() {
        expect('"');
        std::string result;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            if (text_[pos_] == '\\') {
                if (++pos_ >= text_.size()) throw std::runtime_error("safetensors: bad JSON escape");
            }
            result.push_back(text_[pos_++]);
        }
        if (pos_ >= text_.size()) throw std::runtime_error("safetensors: unterminated JSON string");
        ++pos_;
        return result;
    }

    uint64_t integer() {
        whitespace();
        const size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        if (start == pos_) throw std::runtime_error("safetensors: expected integer");
        return std::stoull(text_.substr(start, pos_ - start));
    }

    std::vector<size_t> array() {
        std::vector<size_t> result;
        expect('[');
        while (peek() != ']') {
            result.push_back(static_cast<size_t>(integer()));
            if (peek() == ',') { expect(','); continue; }
            if (peek() != ']') throw std::runtime_error("safetensors: malformed array");
        }
        expect(']');
        return result;
    }

    void skip_value() {
        whitespace();
        const char first = peek();
        if (first == '"') { (void)string(); return; }
        if (first != '{' && first != '[') { (void)integer(); return; }
        const char open = first, close = first == '{' ? '}' : ']';
        int depth = 0;
        bool in_string = false, escaped = false;
        do {
            const char ch = text_[pos_++];
            if (in_string) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') in_string = false;
            } else if (ch == '"') in_string = true;
            else if (ch == open) ++depth;
            else if (ch == close) --depth;
        } while (pos_ < text_.size() && depth > 0);
        if (depth != 0) throw std::runtime_error("safetensors: unbalanced JSON value");
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
};

struct TensorMeta {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

}  // namespace

std::unordered_map<std::string, Tensor> load_safetensors(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + filename);

    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) throw std::runtime_error("safetensors: failed to read header size");
    std::string header(static_cast<size_t>(header_size), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) throw std::runtime_error("safetensors: failed to read header");

    JsonCursor json(header);
    json.expect('{');
    std::vector<TensorMeta> metas;
    while (json.peek() != '}') {
        TensorMeta meta;
        meta.name = json.string();
        json.expect(':');
        if (meta.name == "__metadata__") {
            json.skip_value();
        } else {
            json.expect('{');
            while (json.peek() != '}') {
                const std::string key = json.string();
                json.expect(':');
                if (key == "dtype") meta.dtype = json.string();
                else if (key == "shape") meta.shape = json.array();
                else if (key == "data_offsets") {
                    const auto offsets = json.array();
                    if (offsets.size() != 2) throw std::runtime_error("safetensors: invalid offsets");
                    meta.begin = offsets[0]; meta.end = offsets[1];
                } else json.skip_value();
                if (json.peek() == ',') json.expect(',');
            }
            json.expect('}');
            metas.push_back(std::move(meta));
        }
        if (json.peek() == ',') json.expect(',');
    }
    json.expect('}');

    std::string blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(metas.size());
    for (const TensorMeta& meta : metas) {
        if (meta.dtype != "F32") throw std::runtime_error("safetensors: unsupported dtype " + meta.dtype);
        if (meta.end < meta.begin || meta.end > blob.size() || (meta.end - meta.begin) % sizeof(float) != 0) {
            throw std::runtime_error("safetensors: invalid data range for " + meta.name);
        }
        std::vector<float> values(static_cast<size_t>((meta.end - meta.begin) / sizeof(float)));
        std::memcpy(values.data(), blob.data() + meta.begin, values.size() * sizeof(float));
        tensors.emplace(meta.name, Tensor(meta.shape, values));
    }
    return tensors;
}

GPT2Weights build_gpt2_weights(std::unordered_map<std::string, Tensor> tensors,
                               const GPT2Config& config) {
    GPT2Weights weights;
    weights.config = config;
    weights.wte = std::move(tensors.at("wte.weight"));
    weights.wpe = std::move(tensors.at("wpe.weight"));
    weights.ln_f_w = std::move(tensors.at("ln_f.weight"));
    weights.ln_f_b = std::move(tensors.at("ln_f.bias"));
    weights.layers.resize(config.n_layer);
    for (size_t i = 0; i < config.n_layer; ++i) {
        const std::string prefix = "h." + std::to_string(i) + ".";
        LayerWeights& layer = weights.layers[i];
        layer.ln_1_w = std::move(tensors.at(prefix + "ln_1.weight"));
        layer.ln_1_b = std::move(tensors.at(prefix + "ln_1.bias"));
        layer.attn_c_attn_w = std::move(tensors.at(prefix + "attn.c_attn.weight"));
        layer.attn_c_attn_b = std::move(tensors.at(prefix + "attn.c_attn.bias"));
        layer.attn_c_proj_w = std::move(tensors.at(prefix + "attn.c_proj.weight"));
        layer.attn_c_proj_b = std::move(tensors.at(prefix + "attn.c_proj.bias"));
        layer.ln_2_w = std::move(tensors.at(prefix + "ln_2.weight"));
        layer.ln_2_b = std::move(tensors.at(prefix + "ln_2.bias"));
        layer.mlp_c_fc_w = std::move(tensors.at(prefix + "mlp.c_fc.weight"));
        layer.mlp_c_fc_b = std::move(tensors.at(prefix + "mlp.c_fc.bias"));
        layer.mlp_c_proj_w = std::move(tensors.at(prefix + "mlp.c_proj.weight"));
        layer.mlp_c_proj_b = std::move(tensors.at(prefix + "mlp.c_proj.bias"));
    }
    return weights;
}

}  // namespace gpt2
