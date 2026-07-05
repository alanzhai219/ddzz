/*****************************************************************************
 * PAPI Data Collector — outputs CSV for visualization
 *
 * Runs multiple workloads (matmul at different sizes) and collects
 * performance counter data across multiple PAPI events.
 *
 * Output format: CSV with header row
 *   N,Cycles,Instructions,IPC,L1_Miss,L2_Miss,TLB_Miss,Br_Mispredict
 *
 * Build:
 *   gcc -O2 -o papi_collector papi_collector.c \
 *       -I ../src/install/include -L ../src/install/lib \
 *       -lpapi -lpfm -Wl,-rpath,../src/install/lib
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"

/* ------------------------------------------------------------------ */
/*  Matrix multiplication (naive triple-loop)                         */
/* ------------------------------------------------------------------ */
static void matmul(int N) {
    /* Allocate dynamically to stress memory at any size */
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    /* Init */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }

    /* Multiply */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }

    /* Free */
    for (int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(C[i]);
    }
    free(A); free(B); free(C);
}

/* ------------------------------------------------------------------ */
/*  Events we collect                                                  */
/* ------------------------------------------------------------------ */
#define NUM_EVENTS 6
static const int events[NUM_EVENTS] = {
    PAPI_TOT_CYC,
    PAPI_TOT_INS,
    PAPI_L1_DCM,
    PAPI_L2_TCM,
    PAPI_TLB_DM,
    PAPI_BR_MSP,
};
static const char *event_names[NUM_EVENTS] = {
    "PAPI_TOT_CYC", "PAPI_TOT_INS",
    "PAPI_L1_DCM",  "PAPI_L2_TCM",
    "PAPI_TLB_DM",  "PAPI_BR_MSP",
};

/* ------------------------------------------------------------------ */
/*  Measure one workload at size N, output CSV row                     */
/* ------------------------------------------------------------------ */
static void measure_and_print(int N)
{
    int retval, EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];

    /* Create eventset and add events */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "create_eventset failed for N=%d: %d\n", N, retval);
        return;
    }

    retval = PAPI_add_events(EventSet, (int *)events, NUM_EVENTS);
    if (retval != PAPI_OK) {
        /* Try with multiplexing */
        PAPI_set_multiplex(EventSet);
        retval = PAPI_add_events(EventSet, (int *)events, NUM_EVENTS);
        if (retval != PAPI_OK) {
            fprintf(stderr, "add_events failed for N=%d: %s (%d)\n",
                    N, PAPI_strerror(retval), retval);
            PAPI_cleanup_eventset(EventSet);
            PAPI_destroy_eventset(&EventSet);
            return;
        }
    }

    /* Warm-up: run once to avoid cold-cache effects */
    matmul(N);

    /* Measure */
    PAPI_start(EventSet);
    matmul(N);
    PAPI_stop(EventSet, values);

    /* Compute IPC */
    double ipc = (values[0] > 0) ? (double)values[1] / values[0] : 0.0;

    /* Output CSV row */
    printf("%d,%lld,%lld,%.4f,%lld,%lld,%lld,%lld\n",
           N,
           values[0],                  /* cycles */
           values[1],                  /* instructions */
           ipc,
           values[2],                  /* L1 miss */
           values[3],                  /* L2 miss */
           values[4],                  /* TLB miss */
           values[5]);                 /* branch mispredict */

    /* Cleanup */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    int retval;

    /* Init PAPI */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed: %d\n", retval);
        return 1;
    }

    /* Print header */
    printf("N,Cycles,Instructions,IPC,L1_Miss,L2_Miss,TLB_Miss,Br_Mispredict\n");

    /* Measure at various sizes */
    int sizes[] = { 64, 128, 192, 256, 384, 512, 768, 1024 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        fprintf(stderr, "Measuring N=%d (%d/%d)...\n", sizes[i], i+1, num_sizes);
        measure_and_print(sizes[i]);
    }

    PAPI_shutdown();
    return 0;
}