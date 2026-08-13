#include "weights.hpp"
#include "utils.hpp"

#include <fstream>
#include <cassert>
#include <stdexcept>

namespace gpt2 {

struct TensorMeta {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    std::size_t beg;
    std::size_t end;
};
// load safetensors
std::unordered_map<std::string, Tensor> load_safetensors(const std::string& filename) {
    // 0: open file to a stream
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file"); 
    }

    // 1: read and skip head
    uint64_t head_size = 0;
    file.read(reinterpret_cast<char*>(&head_size), 8);

    // 2. read and skip header
    std::string header(head_size, '\0');
    file.read(header.data(), reinterpret_cast<std::streamsize>(head_size));

    // 3. start parsing json
    JsonParser parser(header);
    parser.skip_whitespace();

    if (parser.buffer()[parser.cursor()] != '{') {
        throw std::runtime_error("not expected safetensors");
    }
    parser.advance(); // skip '{'
    
    // 4. loop the tensor meta
    std::vector<TensorMeta> metas;
    while(true) {
        parser.skip_whitespace();
        if (parser.buffer()[parser.cursor()] == '}') {
            break;
        }

        TensorMeta tm;

        // parse top name
        std::string name = parser.parse_string();
        if (parser.buffer()[parser.cursor()] != ':') {
            throw std::runtime_error("[safetensors] expeced :");
        }

        if (name == "__metadata__") {
            parser.skip_value();
        } else {
            tm.name = name;
            parser.advance(); // skip ':'
            parser.skip_whitespace();

            // parse sub name
            while(true) {
                if (parser.buffer()[parser.cursor()] == '}') {
                    break;
                }
                std::string sub_name = parser.parse_string();
                switch (sub_name):
                    case std::string("dtype"):
                        {
                            parser.advance();
                            parser.skip_whitespace();
                            tm.dtype = parser.parse_string();
                            parser.advance();
                            break;
                        }
                    case std::string("shape"):
                        {
                            parser.advance();
                            parser.skip_whitespace();
                            tm.shape = parser.parse_array();
                            parser.advance();
                            break;
                        }
                    case std::string("offsets"):
                        {
                            parser.advance();
                            parser.skip_whitespace();
                            auto offsets = parser.parse_array();
                            tm.beg = offsets[0];
                            tm.end = offsets[1];
                            parser.advance();
                            break;
                        }
                    case std::string("__metadata__"):
                        {
                            parser.skip_value();
                            break;
                        }
                    default:
                        {
                            break;
                        }
                }
            }
        }
        metas.push_back(tm);
    }

    // read all safetensors
    std::string st_buffer;
    st_buffer.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    
    // read weights buffer
    std::unordered_map<std::string, Tensor> wts;
    for (const auto m : metas) {
        std::string name = m.name;
        // shape
        Tensor t(m.shape);
        // stride
        t.compute_stride();
        // data
        size_t t_size = m.end - m.beg;
        memcpy(t.data(), st_buffer[m.beg], m.end - m.beg);
        wts[name] = t;
    }
    return wts;
}

GPT2Weights build_gpt2_weights(const std::unordered_map<std::string, Tensor>& tensors, const GPT2Config& config) {
    GPT2Weights gpt2_wei;
    gpt2_wei.config = config;
    gpt2_wei.wpe = m["wpe.weight"];
    gpt2_wei.wte = m["wte.weight"];

    gpt2_wei.layers.resize(config.n_layer);
    for (size_t i = 0; i < config.n_layer; ++i) {
        const std::string pre = "h." + std::to_string(i) + "."; 
        LayerWeights L;
        L.ln_1_w = m[pre + "ln_1.weight"]
        L.ln_1_b = m[pre + "ln_1.bias"]
        L.attn_c_attn_w = m[pre + "attn_c_attn.weight"]
        L.attn_c_attn_b = m[pre + "attn_c_attn.bias"]
        L.attn_c_proj_w = m[pre + "attn_c_proj.weight"]
        L.attn_c_proj_b = m[pre + "attn_c_proj.bias"]
        L.ln_2_w = m[pre + "ln_2.weights"]
        L.ln_2_b = m[pre + "ln_2.bias"]
        L.mlp_c_fc_w = m[pre + "mlp.c_fc.weight"];
        L.mlp_c_fc_b = m[pre + "mlp.c_fc.bias"];
        L.mlp_c_proj_w = m[pre + "mlp.c_proj.weight"];
        L.mlp_c_proj_b = m[pre + "mlp.c_proj.bias"];
        gpt2_wei.layers[i] = L;
    }

    gpt2_wei.ln_f_w = m["ln_f.weight"];
    gpt2_wei.ln_f_b = m["ln_f.bias"];
}
} // namespace gpt2
