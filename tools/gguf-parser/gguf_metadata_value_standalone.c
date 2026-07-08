#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Minimal standalone definitions for learning gguf_metadata_value_t.
 * This file intentionally keeps only the type declarations.
 */

typedef enum {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
} gguf_metadata_value_type;

typedef struct {
    uint64_t len;
    char *string;
} gguf_string_t;

typedef union gguf_metadata_value_t gguf_metadata_value_t;

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        /* Any value type is valid, including arrays. */
        gguf_metadata_value_type type;
        /* Number of elements, not bytes. */
        uint64_t len;
        /* Variable-length payload in serialized GGUF representation. */
        gguf_metadata_value_t *array;
    } array;
};

static char *dup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    memcpy(p, s, n);
    return p;
}

static gguf_metadata_value_t make_int32(int32_t x) {
    gguf_metadata_value_t v;
    v.int32 = x;
    return v;
}

static gguf_metadata_value_t make_string(const char *s) {
    gguf_metadata_value_t v;
    v.string.len = (uint64_t)strlen(s);
    v.string.string = dup_cstr(s);
    return v;
}

static gguf_metadata_value_t make_array(gguf_metadata_value_type elem_type, uint64_t len) {
    gguf_metadata_value_t v;
    v.array.type = elem_type;
    v.array.len = len;
    v.array.array = (gguf_metadata_value_t *)calloc((size_t)len, sizeof(gguf_metadata_value_t));
    if (!v.array.array) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return v;
}

static int32_t read_int32(gguf_metadata_value_type type, const gguf_metadata_value_t *v) {
    assert(type == GGUF_METADATA_VALUE_TYPE_INT32);
    return v->int32;
}

static const char *read_string(gguf_metadata_value_type type, const gguf_metadata_value_t *v) {
    assert(type == GGUF_METADATA_VALUE_TYPE_STRING);
    return v->string.string;
}

static const gguf_metadata_value_t *read_array_item(
    gguf_metadata_value_type type,
    const gguf_metadata_value_t *v,
    uint64_t index) {
    assert(type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(index < v->array.len);
    return &v->array.array[index];
}

static void free_value(gguf_metadata_value_type type, gguf_metadata_value_t *v) {
    uint64_t i;

    if (type == GGUF_METADATA_VALUE_TYPE_STRING) {
        free(v->string.string);
        v->string.string = NULL;
        v->string.len = 0;
        return;
    }

    if (type == GGUF_METADATA_VALUE_TYPE_ARRAY) {
        for (i = 0; i < v->array.len; ++i) {
            free_value(v->array.type, &v->array.array[i]);
        }
        free(v->array.array);
        v->array.array = NULL;
        v->array.len = 0;
    }
}

int main(void) {
    gguf_metadata_value_t scalar_i32;
    gguf_metadata_value_t scalar_str;
    gguf_metadata_value_t nested;

    /* Test 1: scalar int32 branch selected by external type tag. */
    scalar_i32 = make_int32(42);
    assert(scalar_i32.int32 == 42);

    /* Test 2: scalar string branch selected by external type tag. */
    scalar_str = make_string("hello");
    assert(scalar_str.string.len == 5);
    assert(strcmp(scalar_str.string.string, "hello") == 0);

    /* Test 3: recursive array branch: array[array[int32]]. */
    nested = make_array(GGUF_METADATA_VALUE_TYPE_ARRAY, 2);

    nested.array.array[0] = make_array(GGUF_METADATA_VALUE_TYPE_INT32, 2);
    nested.array.array[0].array.array[0] = make_int32(1);
    nested.array.array[0].array.array[1] = make_int32(2);

    nested.array.array[1] = make_array(GGUF_METADATA_VALUE_TYPE_INT32, 3);
    nested.array.array[1].array.array[0] = make_int32(10);
    nested.array.array[1].array.array[1] = make_int32(20);
    nested.array.array[1].array.array[2] = make_int32(30);

    assert(nested.array.len == 2);
    assert(nested.array.array[0].array.array[1].int32 == 2);
    assert(nested.array.array[1].array.array[2].int32 == 30);

    /* Test 4: read path with external type tag (decode-like usage). */
    assert(read_int32(GGUF_METADATA_VALUE_TYPE_INT32, &scalar_i32) == 42);
    assert(strcmp(read_string(GGUF_METADATA_VALUE_TYPE_STRING, &scalar_str), "hello") == 0);
    assert(read_array_item(GGUF_METADATA_VALUE_TYPE_ARRAY, &nested, 0)->array.len == 2);
    assert(read_int32(
               GGUF_METADATA_VALUE_TYPE_INT32,
               read_array_item(GGUF_METADATA_VALUE_TYPE_ARRAY,
                               read_array_item(GGUF_METADATA_VALUE_TYPE_ARRAY, &nested, 1),
                               2)) == 30);

    free_value(GGUF_METADATA_VALUE_TYPE_STRING, &scalar_str);
    free_value(GGUF_METADATA_VALUE_TYPE_ARRAY, &nested);

    puts("All minimal tests passed.");
    return 0;
}
