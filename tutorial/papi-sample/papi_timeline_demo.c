/*****************************************************************************
 * papi_timeline_demo.c — Function-Level Timeline Profiling Demo
 *
 * Inline profiler — no separate header needed.
 * Wraps functions to produce a timeline trace with PAPI performance counters.
 *
 * Output: trace.json  →  visualize with: python3 visualize_timeline.py trace.json
 *
 * Build:
 *   gcc -O2 -Wall -o papi_timeline_demo papi_timeline_demo.c \
 *       -I ../src/install/include -L ../src/install/lib \
 *       -lpapi -lpfm -Wl,-rpath,../src/install/lib -lm
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "papi.h"

/* ================================================================== */
/*  Mini Profiler (inline, no separate header)                        */
/* ================================================================== */

#define MAX_TRACES 64
#define MAX_EVENTS 8

static struct {
    int num_events;
    int event_codes[MAX_EVENTS];
    const char *event_names[MAX_EVENTS];
    int EventSet;
    struct {
        const char *name;
        long long t_start;
        long long t_end;
        long long counters[MAX_EVENTS];
    } traces[MAX_TRACES];
    int trace_count;
    int ok;
} g_prof;

static int prof_init(const int *events, const char **names, int n) {
    if (g_prof.ok) return 1;
    if (n > MAX_EVENTS) return 0;

    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed\n");
        return 0;
    }

    g_prof.num_events = n;
    memcpy(g_prof.event_codes, events, n * sizeof(int));
    for (int i = 0; i < n; i++) g_prof.event_names[i] = names[i];

    g_prof.EventSet = PAPI_NULL;
    int r = PAPI_create_eventset(&g_prof.EventSet);
    if (r != PAPI_OK) {
        fprintf(stderr, "PAPI_create_eventset: %s (%d)\n", PAPI_strerror(r), r);
        return 0;
    }

    r = PAPI_assign_eventset_component(g_prof.EventSet, 0);
    if (r != PAPI_OK) {
        fprintf(stderr, "assign_component: %s (%d)\n", PAPI_strerror(r), r);
        return 0;
    }

    /* Enable multiplex first (avoid counter conflicts) */
    PAPI_set_multiplex(g_prof.EventSet);

    r = PAPI_add_events(g_prof.EventSet, g_prof.event_codes, n);
    if (r != PAPI_OK) {
        fprintf(stderr, "add_events: %s (%d)\n", PAPI_strerror(r), r);
        return 0;
    }

    r = PAPI_start(g_prof.EventSet);
    if (r != PAPI_OK) {
        fprintf(stderr, "PAPI_start: %s (%d)\n", PAPI_strerror(r), r);
        return 0;
    }

    g_prof.ok = 1;
    fprintf(stderr, "Profiler initialized with %d events.\n", n);
    return 1;
}

/* Begin a named trace region */
static void tr_begin(const char *name) {
    if (!g_prof.ok || g_prof.trace_count >= MAX_TRACES) return;
    int idx = g_prof.trace_count++;
    g_prof.traces[idx].name = name;
    PAPI_read(g_prof.EventSet, g_prof.traces[idx].counters);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    g_prof.traces[idx].t_start = ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* End current trace region */
static void tr_end(void) {
    if (!g_prof.ok || g_prof.trace_count == 0) return;
    int idx = g_prof.trace_count - 1;
    long long end_vals[MAX_EVENTS];
    PAPI_read(g_prof.EventSet, end_vals);
    for (int i = 0; i < g_prof.num_events; i++)
        g_prof.traces[idx].counters[i] = end_vals[i] - g_prof.traces[idx].counters[i];
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    g_prof.traces[idx].t_end = ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Convenience macro: PROF_TRACE(name) { code... } */
#define PROF_TRACE(name) \
    for (int _t = (tr_begin(name), 0); _t < 1; tr_end(), _t++)

static void prof_json(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    fprintf(f, "{\"event_names\":[");
    for (int i = 0; i < g_prof.num_events; i++)
        fprintf(f, "\"%s\"%c", g_prof.event_names[i], i < g_prof.num_events-1 ? ',' : ' ');
    fprintf(f, "],\"traces\":[\n");

    long long t0 = g_prof.trace_count ? g_prof.traces[0].t_start : 0;
    for (int i = 0; i < g_prof.trace_count; i++) {
        long long start_us = (g_prof.traces[i].t_start - t0) / 1000;
        long long dur_us   = (g_prof.traces[i].t_end - g_prof.traces[i].t_start) / 1000;
        double ipc = (g_prof.traces[i].counters[0] > 0)
            ? (double)g_prof.traces[i].counters[1] / g_prof.traces[i].counters[0] : 0.0;
        fprintf(f, "  {\"name\":\"%s\",\"start_us\":%lld,\"dur_us\":%lld",
                g_prof.traces[i].name, start_us, dur_us);
        fprintf(f, ",\"cycles\":%lld,\"insns\":%lld,\"ipc\":%.4f",
                g_prof.traces[i].counters[0], g_prof.traces[i].counters[1], ipc);
        for (int j = 2; j < g_prof.num_events; j++)
            fprintf(f, ",\"%s\":%lld", g_prof.event_names[j], g_prof.traces[i].counters[j]);
        fprintf(f, "}%s\n", i < g_prof.trace_count-1 ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

static void prof_done(void) {
    if (!g_prof.ok) return;
    PAPI_stop(g_prof.EventSet, NULL);
    PAPI_cleanup_eventset(g_prof.EventSet);
    PAPI_destroy_eventset(&g_prof.EventSet);
    PAPI_shutdown();
    g_prof.ok = 0;
}

/* ================================================================== */
/*  Simulated hotspot functions                                       */
/* ================================================================== */

double data[2048][2048];

/* A: matrix init — memory-heavy write */
void init_matrix(int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            data[i][j] = (double)(i * N + j) * 0.001;
}

/* B: naive matmul — cache-unfriendly (k inner → B[k][j] strides) */
void matmul_naive(int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += data[i][k] * data[k][j];
            data[i][j] = sum;
        }
}

/* C: row sums — serial accumulation, data dependent */
double compute_row_sums(int N) {
    double total = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total += data[i][j];
    return total;
}

/* D: branching kernel — high mispredict rate */
void branching_kernel(int N) {
    int pos = 0, neg = 0;
    for (int i = 0; i < N * N; i++) {
        double val = data[0][i % N];
        if (val > 0.5) {
            if (val > 0.7) { if (val > 0.9) pos++; else neg++; }
            else neg++;
        } else pos++;
    }
    (void)pos; (void)neg;
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main(void) {
    int events[] = {
        PAPI_TOT_CYC, PAPI_TOT_INS,
        PAPI_L1_DCM,  PAPI_L2_TCM,
        PAPI_TLB_DM,  PAPI_BR_MSP,
    };
    const char *names[] = {
        "PAPI_TOT_CYC","PAPI_TOT_INS",
        "PAPI_L1_DCM","PAPI_L2_TCM",
        "PAPI_TLB_DM","PAPI_BR_MSP",
    };
    int n_ev = sizeof(events) / sizeof(events[0]);

    if (!prof_init(events, names, n_ev)) {
        fprintf(stderr, "prof_init failed\n");
        return 1;
    }

    int N = 512;

    PROF_TRACE("init_matrix")        { init_matrix(N); }
    PROF_TRACE("matmul_naive")       { matmul_naive(N); }
    PROF_TRACE("compute_row_sums")   {
        double t = compute_row_sums(N);
        printf("row_sums=%.2f\n", t);
    }
    PROF_TRACE("branching_kernel")   { branching_kernel(N); }

    prof_json("trace.json");
    printf("trace.json written (%d traces)\n", g_prof.trace_count);

    prof_done();
    return 0;
}