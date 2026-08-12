#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace gpt2 {
namespace utils {

size_t numel(const std::vector<size_t>& shape) {
    size_t num = 1;
    for (auto v : shape) {
        num *= v;
    }
    return num;
}
struct JsonParser {
    JsonParser(const std::string& json_file) : m_json(json_file), m_cursor(0) {}

    // parse string
    std::string parse_string() {
        skip_whitespace();
        if (!ensure_at('\"')) {
            throw std::runtime_error("no string");
        }
        advance();

        std::string out_str;
        while(!ensure_at('\"') && !at_end()) {
            if (ensure_at('\\')) {
                ++m_cursor;
            }
            out_str.push_back(m_json[m_cursor]);
            ++m_cursor;
        }
        ++m_cursor;
        return out_str;
    }

    // parse number
    double parse_number() {
        skip_whitespace();

        size_t start = m_cursor;
        while(is_number_char(m_json[m_cursor]) && !at_end()) {
            ++m_cursor;
        }
        double ret_val = std::stod(m_json.substr(start, m_cursor - start));
        return ret_val;
    }

    // parse list
    std::vector<size_t> parse_array() {
        std::vector<size_t> ret_vec;

        skip_whitespace();
        if (!ensure_at('[')) {
            throw std::runtime_error("no array");
        }
        ++m_cursor;
        skip_whitespace();

        while(is_number_char(m_json[m_cursor]) && !at_end()) {
            ret_vec.push_back(static_cast<size_t>(parser_number()));

            skip_whitespace();
            ++m_cursor;

            if (m_json[m_cursor] == ',') {
                ++m_cursor;
                skip_whitespace();
            }
        }
        ++m_cursor;
    }

    // skip whitespace char
    void skip_whitespace() {
        while (is_whitespace(m_json[m_cursor]) && m_cursor < m_json.size()) {
            ++m_cursor;
        }
    }

    // skip arbitrary value
    // only parse specified values.
    // use skip_value to skip unused values.
    void skip_value() {
        skip_whitespace();
        if (at_end()) {
            return;
        }

        switch (text_[cursor_]) {
            case '{': skip_balanced('{', '}'); break;
            case '[': skip_balanced('[', ']'); break;
            case '"': parse_string();          break;
            default:  parse_number();          break;
        }
    }

    // go ahead
    void advance() {
        ++m_cursor;
    }

    std::string buffer() {
        return m_json;
    }

    size_t cursor() {
        return m_cursor;
    }

private:
    bool is_whitespace(char ch) {
        return ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r';
    }

    bool is_number_char(char ch) {
        return ch == '.' || ch == '+' || ch == '-' || ch == 'e' || ch == 'E' || (ch >= '0' && ch <= '9'); 
    }

    bool at_end() {
        return m_cursor == m_json.size();
    }

    bool ensure_at(const char* expected) {
        return (m_json[m_cursor] == expected[0] && !at_end();
    }

    // -------------------------------------------------------------------------
    // Skip a balanced pair of brackets ({} or []) including nested content.
    // Caller must ensure cursor_ points to the opening bracket.
    // -------------------------------------------------------------------------
    void skip_balanced(char open, char close) {
        int depth = 1;
        advance();                           // consume opening bracket
        while (cursor_ < text_.size() && depth > 0) {
            if (text_[cursor_] == open)  ++depth;
            if (text_[cursor_] == close) --depth;
            advance();
        }
    }

    const std::string& m_json;
    size_t m_cursor;

};

} // utils
} // gp2_cpp
