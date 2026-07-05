/*****************************************************************************
 * papi_func_profiler.h — Function-Level Timeline Profiler
 *
 * Usage:
 *   1. Call profiler_init() once.
 *   2. Wrap functions with PROFILER_TRACE(name) { ... }
 *   3. Call profiler_write_json("trace.json") at end.
 *
 * Output: JSON file with function traces including:
 *   - function name
 *   - start/end timestamp (ns)
 *   - performance counters delta (cycles, instructions, cache misses, etc.)
 *
 * Then visualize with: python3 visualize_timeline.py trace.json
 *****************************************************************************/

#ifndef PAPI_FUNC_PROFILER_H
#define PAPI_FUNC_PROFILER_H

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "papi.h"

#define PROFILER_MAX_TRACES 512
#define PROFILER_MAX_EVENTS 6

/* --- Data structures --- */

typedef struct {
    const char *name;           /* function name */
    long long t_start;          /* start timestamp (ns) */
    long long t_end;            /* end timestamp (ns) */
    long long counters[PROFILER_MAX_EVENTS];  /* delta values */
} profiler_trace_t;

typedef struct {
    int         num_events;
    int         event_codes[PROFILER_MAX_EVENTS];
    const char *event_names[PROFILER_MAX_EVENTS];
    int         EventSet;
    profiler_trace_t traces[PROFILER_MAX_TRACES];
    int         trace_count;
    int         initialized;
} profiler_state_t;

static profiler_state_t _ps = {0};

/* --- Init / Shutdown --- */

static inline int profiler_init(const int *events, const char **names, int n)
{
    int retval;

    if (_ps.initialized) return PAPI_OK;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "profiler_init: PAPI_library_init failed: %d\n", retval);
        return retval;
    }

    _ps.num_events = n;
    for (int i = 0; i < n; i++) {
        _ps.event_codes[i]  = events[i];
        _ps.event_names[i]  = names[i];
    }

    retval = PAPI_create_eventset(&_ps.EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "profiler_init: PAPI_create_eventset: %d\n", retval);
        return retval;
    }

    /* Assign to CPU component 0 (required for multiplex) */
    retval = PAPI_assign_eventset_component(_ps.EventSet, 0);
    if (retval != PAPI_OK) {
        fprintf(stderr, "profiler_init: assign_component: %d\n", retval);
        return retval;
    }

    /* Try direct add; fall back to multiplex */
    retval = PAPI_add_events(_ps.EventSet, _ps.event_codes, n);
    if (retval != PAPI_OK) {
        /* Retry with multiplexing */
        retval = PAPI_set_multiplex(_ps.EventSet);
        if (retval != PAPI_OK) {
            fprintf(stderr, "profiler_init: set_multiplex: %d\n", retval);
            return retval;
        }
        retval = PAPI_add_events(_ps.EventSet, _ps.event_codes, n);
        if (retval != PAPI_OK) {
            fprintf(stderr, "profiler_init: add_events (mux): %s (%d)\n",
                    PAPI_strerror(retval), retval);
            return retval;
        }
    }

    retval = PAPI_start(_ps.EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "profiler_init: PAPI_start: %d\n", retval);
        return retval;
    }

    _ps.initialized = 1;
    return PAPI_OK;
}

static inline void profiler_shutdown(void)
{
    if (!_ps.initialized) return;
    PAPI_stop(_ps.EventSet, NULL);
    PAPI_cleanup_eventset(_ps.EventSet);
    PAPI_destroy_eventset(&_ps.EventSet);
    PAPI_shutdown();
    _ps.initialized = 0;
}

/* --- Trace begin / end --- */

static inline void profiler_trace_begin(const char *func_name)
{
    if (!_ps.initialized || _ps.trace_count >= PROFILER_MAX_TRACES)
        return;

    profiler_trace_t *t = &_ps.traces[_ps.trace_count++];
    t->name = func_name;

    /* Read starting values */
    PAPI_read(_ps.EventSet, t->counters);
    (void)t; /* counters now hold starting values, overwrite on end */

    /* Store start time */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t->t_start = ts.tv_sec * 1000000000LL + ts.tv_nsec;
    t->t_end = 0; /* filled later */
}

static inline void profiler_trace_end(void)
{
    if (!_ps.initialized || _ps.trace_count == 0)
        return;

    profiler_trace_t *t = &_ps.traces[_ps.trace_count - 1];

    /* Read end values */
    long long end_vals[PROFILER_MAX_EVENTS];
    PAPI_read(_ps.EventSet, end_vals);

    /* Compute deltas */
    for (int i = 0; i < _ps.num_events; i++) {
        t->counters[i] = end_vals[i] - t->counters[i];
    }

    /* Store end time */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t->t_end = ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* --- Write JSON output --- */

static inline void profiler_write_json(const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"event_names\": [");
    for (int i = 0; i < _ps.num_events; i++) {
        fprintf(f, "\"%s\"%s", _ps.event_names[i],
                i < _ps.num_events - 1 ? ", " : "");
    }
    fprintf(f, "],\n");

    /* Compute time offset (start from 0) */
    long long t0 = (_ps.trace_count > 0) ? _ps.traces[0].t_start : 0;

    fprintf(f, "  \"traces\": [\n");
    for (int i = 0; i < _ps.trace_count; i++) {
        profiler_trace_t *t = &_ps.traces[i];
        long long start_us = (t->t_start - t0) / 1000;  /* us */
        long long dur_us   = (t->t_end - t->t_start) / 1000;
        double ipc = (t->counters[0] > 0)
            ? (double)t->counters[1] / t->counters[0] : 0.0;

        fprintf(f, "    {\"name\": \"%s\", "
                   "\"start_us\": %lld, \"dur_us\": %lld, "
                   "\"cycles\": %lld, \"insns\": %lld, \"ipc\": %.4f",
                   t->name, start_us, dur_us,
                   t->counters[0], t->counters[1], ipc);

        /* Additional counters (index 2+) */
        for (int j = 2; j < _ps.num_events; j++) {
            fprintf(f, ", \"%s\": %lld",
                    _ps.event_names[j], t->counters[j]);
        }
        fprintf(f, "}%s\n", i < _ps.trace_count - 1 ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

/* --- Convenience macros --- */

/* PROFILER_TRACE marks a code block with begin/end profiling.
 * Usage:
 *   PROFILER_TRACE("matrix_multiply") {
 *       matmul(256);
 *   }
 */
static inline void _profiler_trace_begin(const char *nm) { profiler_trace_begin(nm); }
static inline void _profiler_trace_end(void)              { profiler_trace_end();    }

/* Trick: use for-loop to auto-cleanup */
#define PROFILER_TRACE(name) \
    for (int _pt_done = (_profiler_trace_begin(name), 0); \
         _pt_done < 1; \
         _profiler_trace_end(), _pt_done++)

#endif /* PAPI_FUNC_PROFILER_H */