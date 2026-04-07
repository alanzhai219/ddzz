#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

struct SequenceState {
    int seq_id = -1;
    std::vector<int> logical_blocks;
    int past_len = 0;
};

struct BatchMetadata {
    std::vector<int> past_lens;
    std::vector<int> subsequence_begins;
    std::vector<int> block_indices;
    std::vector<int> block_indices_begins;
};

struct BlockCopyPlan {
    int src_block = -1;
    int dst_block = -1;
};

class KVBlockManager {
public:
    KVBlockManager(int num_blocks, int block_size) : m_num_blocks(num_blocks), m_block_size(block_size) {
        for (int i = 0; i < num_blocks; ++i) {
            m_free_blocks.push_back(i);
        }
        m_block_ref_counts.assign(num_blocks, 0);
    }

    void add_sequence(int seq_id) {
        if (m_sequences.count(seq_id)) {
            throw std::runtime_error("sequence already exists");
        }
        m_sequences.emplace(seq_id, SequenceState{seq_id, {}, 0});
    }

    void fork_sequence(int parent_seq_id, int child_seq_id) {
        if (m_sequences.count(child_seq_id)) {
            throw std::runtime_error("child sequence already exists");
        }
        const auto& parent = m_sequences.at(parent_seq_id);
        m_sequences.emplace(child_seq_id, SequenceState{child_seq_id, parent.logical_blocks, parent.past_len});
        for (int block : parent.logical_blocks) {
            m_block_ref_counts[block] += 1;
        }
    }

    void beam_merge(int dst_seq_id, int src_seq_id) {
        if (dst_seq_id == src_seq_id) {
            return;
        }
        auto& dst = m_sequences.at(dst_seq_id);
        const auto& src = m_sequences.at(src_seq_id);
        release_sequence_blocks(dst);
        dst.logical_blocks = src.logical_blocks;
        dst.past_len = src.past_len;
        for (int block : dst.logical_blocks) {
            m_block_ref_counts[block] += 1;
        }
    }

    void finish_sequence(int seq_id) {
        auto it = m_sequences.find(seq_id);
        if (it == m_sequences.end()) {
            throw std::runtime_error("sequence does not exist");
        }
        SequenceState seq = it->second;
        m_sequences.erase(it);
        release_sequence_blocks(seq);
    }

    std::vector<BlockCopyPlan> reserve_for_prefill(int seq_id, int q_len) {
        auto& seq = m_sequences.at(seq_id);
        auto copy_plans = q_len > 0 ? ensure_writable_tail(seq) : std::vector<BlockCopyPlan>{};
        ensure_capacity_for_append(seq, q_len);
        return copy_plans;
    }

    std::vector<BlockCopyPlan> reserve_for_decode(int seq_id) {
        return reserve_for_prefill(seq_id, 1);
    }

    void commit_tokens(int seq_id, int num_tokens) {
        m_sequences.at(seq_id).past_len += num_tokens;
    }

    BatchMetadata build_batch_metadata(const std::vector<int>& seq_ids, const std::vector<int>& q_lens) const {
        if (seq_ids.size() != q_lens.size()) {
            throw std::runtime_error("seq_ids size mismatch with q_lens");
        }

        BatchMetadata meta;
        meta.subsequence_begins.push_back(0);
        meta.block_indices_begins.push_back(0);

        int token_acc = 0;
        int block_acc = 0;

        for (size_t i = 0; i < seq_ids.size(); ++i) {
            const auto& seq = m_sequences.at(seq_ids[i]);
            int q_len = q_lens[i];

            meta.past_lens.push_back(seq.past_len);

            token_acc += q_len;
            meta.subsequence_begins.push_back(token_acc);

            int total_tokens = seq.past_len + q_len;
            int total_blocks = div_up(total_tokens, m_block_size);
            for (int j = 0; j < total_blocks; ++j) {
                meta.block_indices.push_back(seq.logical_blocks[j]);
            }

            block_acc += total_blocks;
            meta.block_indices_begins.push_back(block_acc);
        }

        return meta;
    }

    void dump_state(const std::vector<int>& seq_ids) const {
        std::cout << "scheduler state:\n";
        for (int seq_id : seq_ids) {
            const auto& seq = m_sequences.at(seq_id);
            std::cout << "  seq=" << seq_id << " past_len=" << seq.past_len << " blocks=[";
            for (size_t i = 0; i < seq.logical_blocks.size(); ++i) {
                if (i) {
                    std::cout << ", ";
                }
                std::cout << seq.logical_blocks[i];
            }
            std::cout << "]\n";
        }
        std::cout << "  free_blocks_head=[";
        for (size_t i = 0; i < std::min<size_t>(8, m_free_blocks.size()); ++i) {
            if (i) {
                std::cout << ", ";
            }
            std::cout << m_free_blocks[i];
        }
        std::cout << "]\n";
        std::cout << "  block_ref_counts=[";
        bool first = true;
        for (size_t block = 0; block < m_block_ref_counts.size(); ++block) {
            if (m_block_ref_counts[block] == 0) {
                continue;
            }
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << block << ":" << m_block_ref_counts[block];
        }
        std::cout << "]\n";
    }

    const std::vector<int>& free_blocks() const {
        return m_free_blocks;
    }

    const std::vector<int>& block_ref_counts() const {
        return m_block_ref_counts;
    }

private:
    int m_num_blocks;
    int m_block_size;
    std::vector<int> m_free_blocks;
    std::vector<int> m_block_ref_counts;
    std::unordered_map<int, SequenceState> m_sequences;

    static int div_up(int x, int y) {
        return (x + y - 1) / y;
    }

    int allocate_block() {
        if (m_free_blocks.empty()) {
            throw std::runtime_error("out of KV blocks");
        }
        int block = m_free_blocks.front();
        m_free_blocks.erase(m_free_blocks.begin());
        m_block_ref_counts[block] = 1;
        return block;
    }

    void release_block(int block) {
        if (m_block_ref_counts[block] <= 0) {
            throw std::runtime_error("block already free");
        }
        m_block_ref_counts[block] -= 1;
        if (m_block_ref_counts[block] == 0) {
            m_free_blocks.push_back(block);
            std::sort(m_free_blocks.begin(), m_free_blocks.end());
        }
    }

    void release_sequence_blocks(SequenceState& seq) {
        for (int block : seq.logical_blocks) {
            release_block(block);
        }
        seq.logical_blocks.clear();
        seq.past_len = 0;
    }

    std::vector<BlockCopyPlan> ensure_writable_tail(SequenceState& seq) {
        if (seq.past_len == 0) {
            return {};
        }
        if (seq.past_len % m_block_size == 0) {
            return {};
        }

        int tail_index = (seq.past_len - 1) / m_block_size;
        int tail_block = seq.logical_blocks[tail_index];
        if (m_block_ref_counts[tail_block] <= 1) {
            return {};
        }

        int new_block = allocate_block();
        seq.logical_blocks[tail_index] = new_block;
        release_block(tail_block);
        return {{tail_block, new_block}};
    }

    void ensure_capacity_for_append(SequenceState& seq, int append_tokens) {
        int needed_tokens = seq.past_len + append_tokens;
        int needed_blocks = div_up(needed_tokens, m_block_size);
        while (static_cast<int>(seq.logical_blocks.size()) < needed_blocks) {
            seq.logical_blocks.push_back(allocate_block());
        }
    }
};

class ExecutorPACommon {
public:
    explicit ExecutorPACommon(int block_size) : m_block_size(block_size) {}

    std::vector<int> build_slot_mapping(const BatchMetadata& meta, const std::vector<int>& q_lens) const {
        std::vector<int> slots;
        for (size_t seq_idx = 0; seq_idx < q_lens.size(); ++seq_idx) {
            int past_len = meta.past_lens[seq_idx];
            int block_begin = meta.block_indices_begins[seq_idx];
            int q_len = q_lens[seq_idx];

            for (int j = 0; j < q_len; ++j) {
                int logical_pos = past_len + j;
                int logical_block = logical_pos / m_block_size;
                int offset = logical_pos % m_block_size;
                int physical_block = meta.block_indices[block_begin + logical_block];
                slots.push_back(physical_block * m_block_size + offset);
            }
        }
        return slots;
    }

    std::vector<std::pair<int, int>> collect_context_positions(
        const BatchMetadata& meta,
        int seq_idx,
        int total_kv_len) const {
        std::vector<std::pair<int, int>> result;
        int block_begin = meta.block_indices_begins[seq_idx];
        for (int pos = 0; pos < total_kv_len; ++pos) {
            int logical_block = pos / m_block_size;
            int offset = pos % m_block_size;
            int physical_block = meta.block_indices[block_begin + logical_block];
            result.push_back({physical_block, offset});
        }
        return result;
    }

private:
    int m_block_size;
};

class Int8Quantizer {
public:
    static std::pair<std::vector<int8_t>, float> quantize_row(const std::vector<float>& x) {
        float max_abs = 0.0f;
        for (float v : x) {
            max_abs = std::max(max_abs, std::fabs(v));
        }

        float scale = max_abs < 1e-12f ? 1.0f : max_abs / 127.0f;
        std::vector<int8_t> q(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            float v = x[i] / scale;
            v = std::max(-127.0f, std::min(127.0f, std::round(v)));
            q[i] = static_cast<int8_t>(v);
        }
        return {q, scale};
    }

    static std::vector<float> dequantize_row(const std::vector<int8_t>& q, float scale) {
        std::vector<float> x(q.size());
        for (size_t i = 0; i < q.size(); ++i) {
            x[i] = static_cast<float>(q[i]) * scale;
        }
        return x;
    }
};

static std::vector<float> softmax(const std::vector<float>& x) {
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : x) {
        max_v = std::max(max_v, v);
    }

    std::vector<float> e(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        e[i] = std::exp(x[i] - max_v);
        sum += e[i];
    }
    for (float& v : e) {
        v /= sum;
    }
    return e;
}

static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

class PagedAttentionExecutor {
public:
    using Tensor2 = std::vector<std::vector<float>>;
    using Tensor3 = std::vector<std::vector<std::vector<float>>>;

    PagedAttentionExecutor(
        int layer_id,
        int num_blocks,
        int num_heads,
        int head_size,
        int block_size,
        bool use_int8_cache)
        : m_layer_id(layer_id),
          m_num_blocks(num_blocks),
          m_num_heads(num_heads),
          m_head_size(head_size),
          m_block_size(block_size),
          m_use_int8_cache(use_int8_cache),
          m_common(block_size) {
        int total_slots = num_blocks * num_heads * block_size;
        if (m_use_int8_cache) {
            m_k_cache_q.resize(total_slots * head_size, 0);
            m_v_cache_q.resize(total_slots * head_size, 0);
            m_k_scales.resize(total_slots, 1.0f);
            m_v_scales.resize(total_slots, 1.0f);
        } else {
            m_k_cache_f.resize(total_slots * head_size, 0.0f);
            m_v_cache_f.resize(total_slots * head_size, 0.0f);
        }
    }

    void write_kv(
        const BatchMetadata& meta,
        const std::vector<int>& q_lens,
        const Tensor3& k_new,
        const Tensor3& v_new) {
        auto slots = m_common.build_slot_mapping(meta, q_lens);
        for (size_t token_idx = 0; token_idx < slots.size(); ++token_idx) {
            write_one_token(slots[token_idx], k_new[token_idx], v_new[token_idx]);
        }
    }

    void copy_block(int src_block, int dst_block) {
        for (int h = 0; h < m_num_heads; ++h) {
            for (int offset = 0; offset < m_block_size; ++offset) {
                if (m_use_int8_cache) {
                    int src_base = slot_head_base(src_block, h, offset);
                    int dst_base = slot_head_base(dst_block, h, offset);
                    std::copy(
                        m_k_cache_q.begin() + src_base,
                        m_k_cache_q.begin() + src_base + m_head_size,
                        m_k_cache_q.begin() + dst_base);
                    std::copy(
                        m_v_cache_q.begin() + src_base,
                        m_v_cache_q.begin() + src_base + m_head_size,
                        m_v_cache_q.begin() + dst_base);
                    m_k_scales[slot_scale_index(dst_block, h, offset)] = m_k_scales[slot_scale_index(src_block, h, offset)];
                    m_v_scales[slot_scale_index(dst_block, h, offset)] = m_v_scales[slot_scale_index(src_block, h, offset)];
                } else {
                    int src_base = slot_head_base(src_block, h, offset);
                    int dst_base = slot_head_base(dst_block, h, offset);
                    std::copy(
                        m_k_cache_f.begin() + src_base,
                        m_k_cache_f.begin() + src_base + m_head_size,
                        m_k_cache_f.begin() + dst_base);
                    std::copy(
                        m_v_cache_f.begin() + src_base,
                        m_v_cache_f.begin() + src_base + m_head_size,
                        m_v_cache_f.begin() + dst_base);
                }
            }
        }
    }

    Tensor3 prefill(
        const BatchMetadata& meta,
        const std::vector<int>& q_lens,
        const Tensor3& q,
        const Tensor3& k,
        const Tensor3& v) {
        write_kv(meta, q_lens, k, v);

        Tensor3 outputs;
        int token_start = 0;
        for (size_t seq_idx = 0; seq_idx < q_lens.size(); ++seq_idx) {
            int q_len = q_lens[seq_idx];
            int total_kv_len = meta.past_lens[seq_idx] + q_len;

            Tensor3 q_seq(q.begin() + token_start, q.begin() + token_start + q_len);
            auto out_seq = attention_one_sequence(q_seq, meta, static_cast<int>(seq_idx), total_kv_len);
            outputs.insert(outputs.end(), out_seq.begin(), out_seq.end());
            token_start += q_len;
        }
        return outputs;
    }

    Tensor3 decode(
        const BatchMetadata& meta,
        const Tensor3& q,
        const Tensor3& k,
        const Tensor3& v) {
        std::vector<int> q_lens(meta.past_lens.size(), 1);
        write_kv(meta, q_lens, k, v);

        Tensor3 outputs;
        for (size_t seq_idx = 0; seq_idx < q_lens.size(); ++seq_idx) {
            int total_kv_len = meta.past_lens[seq_idx] + 1;
            Tensor3 q_seq = {q[seq_idx]};
            auto out_seq = attention_one_sequence(q_seq, meta, static_cast<int>(seq_idx), total_kv_len);
            outputs.insert(outputs.end(), out_seq.begin(), out_seq.end());
        }
        return outputs;
    }

private:
    int m_layer_id;
    int m_num_blocks;
    int m_num_heads;
    int m_head_size;
    int m_block_size;
    bool m_use_int8_cache;
    ExecutorPACommon m_common;

    std::vector<int8_t> m_k_cache_q;
    std::vector<int8_t> m_v_cache_q;
    std::vector<float> m_k_scales;
    std::vector<float> m_v_scales;

    std::vector<float> m_k_cache_f;
    std::vector<float> m_v_cache_f;

    int slot_head_base(int block, int head, int offset) const {
        int slot = ((block * m_num_heads + head) * m_block_size + offset);
        return slot * m_head_size;
    }

    int slot_scale_index(int block, int head, int offset) const {
        return ((block * m_num_heads + head) * m_block_size + offset);
    }

    void write_one_token(int slot, const Tensor2& k, const Tensor2& v) {
        int block = slot / m_block_size;
        int offset = slot % m_block_size;

        for (int h = 0; h < m_num_heads; ++h) {
            if (m_use_int8_cache) {
                auto qk = Int8Quantizer::quantize_row(k[h]);
                auto qv = Int8Quantizer::quantize_row(v[h]);

                int base_k = slot_head_base(block, h, offset);
                int base_v = slot_head_base(block, h, offset);

                std::copy(qk.first.begin(), qk.first.end(), m_k_cache_q.begin() + base_k);
                std::copy(qv.first.begin(), qv.first.end(), m_v_cache_q.begin() + base_v);

                m_k_scales[slot_scale_index(block, h, offset)] = qk.second;
                m_v_scales[slot_scale_index(block, h, offset)] = qv.second;
            } else {
                int base_k = slot_head_base(block, h, offset);
                int base_v = slot_head_base(block, h, offset);

                std::copy(k[h].begin(), k[h].end(), m_k_cache_f.begin() + base_k);
                std::copy(v[h].begin(), v[h].end(), m_v_cache_f.begin() + base_v);
            }
        }
    }

    std::pair<Tensor2, Tensor2> read_token_kv(int block, int offset) const {
        Tensor2 k(m_num_heads, std::vector<float>(m_head_size, 0.0f));
        Tensor2 v(m_num_heads, std::vector<float>(m_head_size, 0.0f));

        for (int h = 0; h < m_num_heads; ++h) {
            if (m_use_int8_cache) {
                int base_k = slot_head_base(block, h, offset);
                int base_v = slot_head_base(block, h, offset);

                std::vector<int8_t> qk(m_k_cache_q.begin() + base_k, m_k_cache_q.begin() + base_k + m_head_size);
                std::vector<int8_t> qv(m_v_cache_q.begin() + base_v, m_v_cache_q.begin() + base_v + m_head_size);

                k[h] = Int8Quantizer::dequantize_row(qk, m_k_scales[slot_scale_index(block, h, offset)]);
                v[h] = Int8Quantizer::dequantize_row(qv, m_v_scales[slot_scale_index(block, h, offset)]);
            } else {
                int base_k = slot_head_base(block, h, offset);
                int base_v = slot_head_base(block, h, offset);

                std::copy(m_k_cache_f.begin() + base_k, m_k_cache_f.begin() + base_k + m_head_size, k[h].begin());
                std::copy(m_v_cache_f.begin() + base_v, m_v_cache_f.begin() + base_v + m_head_size, v[h].begin());
            }
        }

        return {k, v};
    }

    Tensor3 attention_one_sequence(
        const Tensor3& q_seq,
        const BatchMetadata& meta,
        int seq_idx,
        int total_kv_len) const {
        auto ctx_positions = m_common.collect_context_positions(meta, seq_idx, total_kv_len);

        Tensor3 k_ctx;
        Tensor3 v_ctx;
        for (const auto& pos : ctx_positions) {
            auto kv = read_token_kv(pos.first, pos.second);
            k_ctx.push_back(kv.first);
            v_ctx.push_back(kv.second);
        }

        Tensor3 out(
            q_seq.size(),
            std::vector<std::vector<float>>(m_num_heads, std::vector<float>(m_head_size, 0.0f)));

        float scale = 1.0f / std::sqrt(static_cast<float>(m_head_size));

        for (size_t t = 0; t < q_seq.size(); ++t) {
            int causal_kv_len = total_kv_len - static_cast<int>(q_seq.size() - 1 - t);
            for (int h = 0; h < m_num_heads; ++h) {
                std::vector<float> scores(causal_kv_len, 0.0f);
                for (int kv_i = 0; kv_i < causal_kv_len; ++kv_i) {
                    scores[kv_i] = dot(q_seq[t][h], k_ctx[kv_i][h]) * scale;
                }

                auto probs = softmax(scores);
                for (int kv_i = 0; kv_i < causal_kv_len; ++kv_i) {
                    for (int d = 0; d < m_head_size; ++d) {
                        out[t][h][d] += probs[kv_i] * v_ctx[kv_i][h][d];
                    }
                }
            }
        }
        return out;
    }
};

class ToyLayer {
public:
    using Tensor2 = std::vector<std::vector<float>>;
    using Tensor3 = std::vector<std::vector<std::vector<float>>>;

    ToyLayer(
        int layer_id,
        int hidden_size,
        int num_heads,
        int head_size,
        int num_blocks,
        int block_size,
        bool use_int8_cache,
        uint32_t seed)
        : m_layer_id(layer_id),
          m_hidden_size(hidden_size),
          m_num_heads(num_heads),
          m_head_size(head_size),
          m_pa(layer_id, num_blocks, num_heads, head_size, block_size, use_int8_cache) {
        init_weights(seed);
    }

    Tensor2 forward_prefill(const Tensor2& x, const BatchMetadata& meta, const std::vector<int>& q_lens) {
        auto qkv = project_qkv(x);
        auto attn_out = m_pa.prefill(meta, q_lens, qkv.q, qkv.k, qkv.v);
        auto merged = merge_heads(attn_out);
        return linear(merged, m_wo);
    }

    Tensor2 forward_decode(const Tensor2& x, const BatchMetadata& meta) {
        auto qkv = project_qkv(x);
        auto attn_out = m_pa.decode(meta, qkv.q, qkv.k, qkv.v);
        auto merged = merge_heads(attn_out);
        return linear(merged, m_wo);
    }

    void copy_block(int src_block, int dst_block) {
        m_pa.copy_block(src_block, dst_block);
    }

private:
    struct QKV {
        Tensor3 q;
        Tensor3 k;
        Tensor3 v;
    };

    int m_layer_id;
    int m_hidden_size;
    int m_num_heads;
    int m_head_size;

    std::vector<std::vector<float>> m_wq;
    std::vector<std::vector<float>> m_wk;
    std::vector<std::vector<float>> m_wv;
    std::vector<std::vector<float>> m_wo;

    PagedAttentionExecutor m_pa;

    void init_weights(uint32_t seed) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(m_hidden_size)));

        int proj = m_num_heads * m_head_size;
        m_wq.assign(m_hidden_size, std::vector<float>(proj));
        m_wk.assign(m_hidden_size, std::vector<float>(proj));
        m_wv.assign(m_hidden_size, std::vector<float>(proj));
        m_wo.assign(proj, std::vector<float>(m_hidden_size));

        for (int i = 0; i < m_hidden_size; ++i) {
            for (int j = 0; j < proj; ++j) {
                m_wq[i][j] = dist(gen);
                m_wk[i][j] = dist(gen);
                m_wv[i][j] = dist(gen);
            }
        }
        for (int i = 0; i < proj; ++i) {
            for (int j = 0; j < m_hidden_size; ++j) {
                m_wo[i][j] = dist(gen);
            }
        }
    }

    static Tensor2 linear(const Tensor2& x, const std::vector<std::vector<float>>& w) {
        int rows = static_cast<int>(x.size());
        int in_dim = static_cast<int>(w.size());
        int out_dim = static_cast<int>(w[0].size());

        Tensor2 y(rows, std::vector<float>(out_dim, 0.0f));
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < in_dim; ++i) {
                float xv = x[r][i];
                for (int j = 0; j < out_dim; ++j) {
                    y[r][j] += xv * w[i][j];
                }
            }
        }
        return y;
    }

    QKV project_qkv(const Tensor2& x) {
        auto q2 = linear(x, m_wq);
        auto k2 = linear(x, m_wk);
        auto v2 = linear(x, m_wv);

        QKV out;
        out.q = split_heads(q2);
        out.k = split_heads(k2);
        out.v = split_heads(v2);
        return out;
    }

    Tensor3 split_heads(const Tensor2& x) const {
        Tensor3 y(
            x.size(),
            std::vector<std::vector<float>>(m_num_heads, std::vector<float>(m_head_size, 0.0f)));

        for (size_t t = 0; t < x.size(); ++t) {
            for (int h = 0; h < m_num_heads; ++h) {
                for (int d = 0; d < m_head_size; ++d) {
                    y[t][h][d] = x[t][h * m_head_size + d];
                }
            }
        }
        return y;
    }

    Tensor2 merge_heads(const Tensor3& x) const {
        Tensor2 y(x.size(), std::vector<float>(m_num_heads * m_head_size, 0.0f));
        for (size_t t = 0; t < x.size(); ++t) {
            for (int h = 0; h < m_num_heads; ++h) {
                for (int d = 0; d < m_head_size; ++d) {
                    y[t][h * m_head_size + d] = x[t][h][d];
                }
            }
        }
        return y;
    }
};

class ToyLLMRuntime {
public:
    using Tensor2 = std::vector<std::vector<float>>;

    ToyLLMRuntime(
        int num_layers,
        int hidden_size,
        int num_heads,
        int head_size,
        int num_blocks,
        int block_size,
        bool use_int8_cache)
        : m_num_layers(num_layers),
          m_hidden_size(hidden_size),
          m_manager(num_blocks, block_size) {
        for (int i = 0; i < num_layers; ++i) {
            m_layers.emplace_back(
                i,
                hidden_size,
                num_heads,
                head_size,
                num_blocks,
                block_size,
                use_int8_cache,
                1234u + static_cast<uint32_t>(i));
        }
    }

    void add_sequence(int seq_id) {
        m_manager.add_sequence(seq_id);
    }

    void fork_sequence(int parent_seq_id, int child_seq_id) {
        m_manager.fork_sequence(parent_seq_id, child_seq_id);
    }

    void beam_merge(int dst_seq_id, int src_seq_id) {
        m_manager.beam_merge(dst_seq_id, src_seq_id);
    }

    void finish_sequence(int seq_id) {
        m_manager.finish_sequence(seq_id);
    }

    Tensor2 prefill(const std::vector<int>& seq_ids, const Tensor2& x, const std::vector<int>& q_lens) {
        if (m_prefill_done) {
            throw std::runtime_error("prefill already executed");
        }

        std::vector<BlockCopyPlan> copy_plans;
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            auto seq_copy_plans = m_manager.reserve_for_prefill(seq_ids[i], q_lens[i]);
            copy_plans.insert(copy_plans.end(), seq_copy_plans.begin(), seq_copy_plans.end());
        }
        apply_copy_plans(copy_plans);

        auto meta = m_manager.build_batch_metadata(seq_ids, q_lens);

        Tensor2 hidden = x;
        for (auto& layer : m_layers) {
            hidden = layer.forward_prefill(hidden, meta, q_lens);
        }

        for (size_t i = 0; i < seq_ids.size(); ++i) {
            m_manager.commit_tokens(seq_ids[i], q_lens[i]);
        }

        m_prefill_done = true;
        return hidden;
    }

    Tensor2 decode(const std::vector<int>& seq_ids, const Tensor2& x) {
        if (!m_prefill_done) {
            throw std::runtime_error("decode called before prefill");
        }

        std::vector<int> q_lens(seq_ids.size(), 1);
        std::vector<BlockCopyPlan> copy_plans;
        for (int seq_id : seq_ids) {
            auto seq_copy_plans = m_manager.reserve_for_decode(seq_id);
            copy_plans.insert(copy_plans.end(), seq_copy_plans.begin(), seq_copy_plans.end());
        }

        apply_copy_plans(copy_plans);

        auto meta = m_manager.build_batch_metadata(seq_ids, q_lens);

        Tensor2 hidden = x;
        for (auto& layer : m_layers) {
            hidden = layer.forward_decode(hidden, meta);
        }

        for (int seq_id : seq_ids) {
            m_manager.commit_tokens(seq_id, 1);
        }

        return hidden;
    }

    const KVBlockManager& manager() const {
        return m_manager;
    }

private:
    int m_num_layers;
    int m_hidden_size;
    bool m_prefill_done = false;

    KVBlockManager m_manager;
    std::vector<ToyLayer> m_layers;

    void apply_copy_plans(const std::vector<BlockCopyPlan>& copy_plans) {
        for (const auto& plan : copy_plans) {
            for (auto& layer : m_layers) {
                layer.copy_block(plan.src_block, plan.dst_block);
            }
        }
    }
};

static std::vector<std::vector<float>> make_random_tensor2(int rows, int cols, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> x(rows, std::vector<float>(cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            x[i][j] = dist(gen);
        }
    }
    return x;
}

int main() {
    ToyLLMRuntime runtime(
        4,     // num_layers
        32,    // hidden_size
        4,     // num_heads
        8,     // head_size
        64,    // num_blocks
        4,     // block_size
        true   // use_int8_cache
    );

    std::vector<int> seq_ids = {100, 200};
    for (int seq_id : seq_ids) {
        runtime.add_sequence(seq_id);
    }

    std::cout << "=== initial state ===\n";
    runtime.manager().dump_state(seq_ids);

    std::cout << "\n=== prefill ===\n";
    std::vector<int> prefill_q_lens = {3, 2};
    auto x_prefill = make_random_tensor2(5, 32, 1);
    auto out_prefill = runtime.prefill(seq_ids, x_prefill, prefill_q_lens);
    std::cout << "prefill output shape = [" << out_prefill.size() << ", " << out_prefill[0].size() << "]\n";
    runtime.manager().dump_state(seq_ids);

    std::cout << "\n=== decode step 1 ===\n";
    auto x_decode_1 = make_random_tensor2(2, 32, 2);
    auto out_decode_1 = runtime.decode(seq_ids, x_decode_1);
    std::cout << "decode step 1 output shape = [" << out_decode_1.size() << ", " << out_decode_1[0].size() << "]\n";
    runtime.manager().dump_state(seq_ids);

    std::cout << "\n=== decode step 2 ===\n";
    auto x_decode_2 = make_random_tensor2(2, 32, 3);
    auto out_decode_2 = runtime.decode(seq_ids, x_decode_2);
    std::cout << "decode step 2 output shape = [" << out_decode_2.size() << ", " << out_decode_2[0].size() << "]\n";
    runtime.manager().dump_state(seq_ids);

    std::cout << "\n=== fork beam: 100 -> 300 ===\n";
    runtime.fork_sequence(100, 300);
    runtime.manager().dump_state({100, 200, 300});

    std::cout << "\n=== branch decode after fork ===\n";
    auto x_decode_3 = make_random_tensor2(2, 32, 4);
    auto out_decode_3 = runtime.decode({100, 300}, x_decode_3);
    std::cout << "branch decode output shape = [" << out_decode_3.size() << ", " << out_decode_3[0].size() << "]\n";
    runtime.manager().dump_state({100, 200, 300});

    std::cout << "\n=== beam merge: 200 <- 300 ===\n";
    runtime.beam_merge(200, 300);
    runtime.manager().dump_state({100, 200, 300});

    std::cout << "\n=== finish sequence 300 ===\n";
    runtime.finish_sequence(300);
    runtime.manager().dump_state({100, 200});

    std::cout << "\n=== finish remaining sequences ===\n";
    runtime.finish_sequence(100);
    runtime.finish_sequence(200);
    std::cout << "scheduler state:\n";
    std::cout << "  free_blocks_head=[";
    for (size_t i = 0; i < std::min<size_t>(8, runtime.manager().free_blocks().size()); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << runtime.manager().free_blocks()[i];
    }
    std::cout << "]\n";
    std::cout << "  block_ref_counts=[";
    bool first = true;
    for (size_t block = 0; block < runtime.manager().block_ref_counts().size(); ++block) {
        int ref_count = runtime.manager().block_ref_counts()[block];
        if (ref_count == 0) {
            continue;
        }
        if (!first) {
            std::cout << ", ";
        }
        first = false;
        std::cout << block << ":" << ref_count;
    }
    std::cout << "]\n";

    return 0;
}