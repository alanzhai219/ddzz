#include <stdio.h>
#include <string.h>

#include "../common/benchmark.hpp"

#define N 200000
#define ROUNDS 2000

enum CharClass {
    CLASS_OTHER = 0,
    CLASS_ALPHA = 1,
    CLASS_DIGIT = 2,
    CLASS_SPACE = 3,
};

enum ParserState {
    STATE_GROUND = 0,
    STATE_IDENT = 1,
    STATE_NUMBER = 2,
};

struct ParseStats {
    int identifiers;
    int numbers;
};

static unsigned char char_class_table[256];
static unsigned char next_state_table[3][4];

static void init_tables() {
    memset(char_class_table, 0, sizeof(char_class_table));
    for (int c = 'a'; c <= 'z'; c++) char_class_table[c] = CLASS_ALPHA;
    for (int c = 'A'; c <= 'Z'; c++) char_class_table[c] = CLASS_ALPHA;
    char_class_table[(unsigned char)'_'] = CLASS_ALPHA;
    for (int c = '0'; c <= '9'; c++) char_class_table[c] = CLASS_DIGIT;
    char_class_table[(unsigned char)' '] = CLASS_SPACE;
    char_class_table[(unsigned char)'\n'] = CLASS_SPACE;
    char_class_table[(unsigned char)'\t'] = CLASS_SPACE;

    next_state_table[STATE_GROUND][CLASS_OTHER] = STATE_GROUND;
    next_state_table[STATE_GROUND][CLASS_ALPHA] = STATE_IDENT;
    next_state_table[STATE_GROUND][CLASS_DIGIT] = STATE_NUMBER;
    next_state_table[STATE_GROUND][CLASS_SPACE] = STATE_GROUND;

    next_state_table[STATE_IDENT][CLASS_OTHER] = STATE_GROUND;
    next_state_table[STATE_IDENT][CLASS_ALPHA] = STATE_IDENT;
    next_state_table[STATE_IDENT][CLASS_DIGIT] = STATE_IDENT;
    next_state_table[STATE_IDENT][CLASS_SPACE] = STATE_GROUND;

    next_state_table[STATE_NUMBER][CLASS_OTHER] = STATE_GROUND;
    next_state_table[STATE_NUMBER][CLASS_ALPHA] = STATE_IDENT;
    next_state_table[STATE_NUMBER][CLASS_DIGIT] = STATE_NUMBER;
    next_state_table[STATE_NUMBER][CLASS_SPACE] = STATE_GROUND;
}

static void fill_program(char *buffer, size_t length) {
    const char *idents[] = {"alpha", "beta", "value", "tmp", "node_1"};
    const char *numbers[] = {"1", "42", "1024", "77", "9000"};
    size_t pos = 0;

    while (pos + 16 < length) {
        const char *ident = idents[rand() % 5];
        const char *number = numbers[rand() % 5];
        int written = snprintf(buffer + pos,
                               length - pos,
                               "%s = %s ; ",
                               ident,
                               number);
        if (written <= 0) {
            break;
        }
        pos += (size_t)written;
    }

    while (pos < length) {
        buffer[pos++] = ' ';
    }
}

static struct ParseStats parse_branchy(const char *text, size_t length) {
    struct ParseStats stats = {0, 0};
    int state = STATE_GROUND;

    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];

        if (state == STATE_GROUND) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
                stats.identifiers++;
                state = STATE_IDENT;
            } else if (c >= '0' && c <= '9') {
                stats.numbers++;
                state = STATE_NUMBER;
            }
        } else if (state == STATE_IDENT) {
            if (!((c >= 'a' && c <= 'z') ||
                  (c >= 'A' && c <= 'Z') ||
                  c == '_' ||
                  (c >= '0' && c <= '9'))) {
                state = STATE_GROUND;
            }
        } else {
            if (c >= 'a' && c <= 'z') {
                stats.identifiers++;
                state = STATE_IDENT;
            } else if (!(c >= '0' && c <= '9')) {
                state = STATE_GROUND;
            }
        }
    }

    return stats;
}

static struct ParseStats parse_table_driven(const char *text, size_t length) {
    struct ParseStats stats = {0, 0};
    unsigned char state = STATE_GROUND;

    for (size_t i = 0; i < length; i++) {
        unsigned char klass = char_class_table[(unsigned char)text[i]];
        unsigned char next_state = next_state_table[state][klass];

        if (state == STATE_GROUND) {
            stats.identifiers += (next_state == STATE_IDENT);
            stats.numbers += (next_state == STATE_NUMBER);
        } else if (state == STATE_NUMBER) {
            stats.identifiers += (next_state == STATE_IDENT);
        }

        state = next_state;
    }

    return stats;
}

int main() {
    srand(42);
    init_tables();

    char *program = branchless_memory_layout::checked_malloc<char>(N + 1);

    fill_program(program, N);
    program[N] = '\0';

    struct ParseStats branchy = parse_branchy(program, N);
    struct ParseStats table = parse_table_driven(program, N);
    branchless_memory_layout::validate_equal_int_pair(
        branchy.identifiers,
        branchy.numbers,
        table.identifiers,
        table.numbers);

    volatile int sink = 0;

    double branchy_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() {
            branchy = parse_branchy(program, N);
            sink += branchy.identifiers + branchy.numbers;
        });

    double table_seconds = branchless_memory_layout::benchmark_seconds(
        ROUNDS,
        [&]() {
            table = parse_table_driven(program, N);
            sink += table.identifiers + table.numbers;
        });

    printf("table-driven parser example\n");
    printf("  branchy: %.3f s\n", branchy_seconds);
    printf("  table:   %.3f s\n", table_seconds);
    printf("  speedup: %.2fx\n", branchy_seconds / table_seconds);
    printf("  ids:     %d\n", branchy.identifiers);
    printf("  nums:    %d\n", branchy.numbers);
    printf("  sink:    %d\n", sink);

    free(program);
    return 0;
}