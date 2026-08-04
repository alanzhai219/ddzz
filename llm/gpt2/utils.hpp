#pragma once

#include <string>

namespace gpt2 {
namespace utils {

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
        while(ensure_at('\"') && !at_end()) {
            out_str.push_back(m_json[m_cursor]);
            ++m_cursor;
        }
        ++m_cursor;
        return out_str;
    }

    // parse number
    double parse_number() {
        skip_whitespace();

        size_t begin = m_cursor;
        while(is_number_char(m_json[m_cursor]) && !at_end()) {
            ++m_cursor;
        }
        double ret_val = std::stod(m_json.substr(begin, m_cursor - begin));
        return ret_val;
    }
    // parse list

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

    bool ensure_at(const char* expected) {
        return (m_json[m_cursor] == expected[0] && !at_end();
    }

    const std::string& m_json;
    size_t m_cursor;

};

} // utils
} // gp2_cpp
