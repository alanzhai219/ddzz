#pragma once

#include <string>

namespace gpt2 {
namespace utils {

struct JsonParser {
    JsonParser(const std::string& json_file) : m_json(json_file), m_cursor(0) {}

    // parse string
    std::string parse_string() {
        skip_whitespace();
        ensure_at('\"');
        advance();

        while() {
            
        }
    }

    // skip whitespace char
    void skip_whitespace() {
        while (is_whitespace(m_json[m_cursor]) && m_cursor < m_json.size()) {
            ++m_cursor;
        }
    }

    // go ahead
    void advance() {
        ++m_cursor;
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

    void ensure_at(const char* expected) {
        if (at_end() || m_json[m_cursor] != expected[0]) {
            std::string msg = "safetensors header: expected '";
            msg += expected;
            msg += "';";
            throw std::runtime_error(msg);
        }
    }

    const std::string& m_json;
    size_t m_cursor;

};

} // utils
} // gp2_cpp
