/*****************************************************************************
 * PAPI Low-Level API Sample
 *
 * Demonstrates:
 *   - PAPI_library_init()
 *   - PAPI_create_eventset()
 *   - PAPI_add_event()  (PAPI_TOT_CYC, PAPI_TOT_INS,
 *                         PAPI_L1_DCM, PAPI_L2_TCM)
 *   - PAPI_start() / PAPI_stop() / PAPI_read()
 *   - IPC (Instructions Per Cycle) calculation
 *   - Cache miss rates (L1/L2/L3)
 *
 * Build:
 *   gcc -o papi_low_level papi_low_level.c \
 *       -I ../src/install/include \
 *       -L ../src/install/lib \
 *       -lpapi -lpfm -Wl,-rpath,../src/install/lib
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

/* ------------------------------------------------------------------ */
/*  Workload: simple dense matrix multiplication (N x N)               */
/* ------------------------------------------------------------------ */
#define N 256

static double A[N][N];
static double B[N][N];
static double C[N][N];

static void init_matrices(void)
{
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }
    }
}

static void matmul(void)
{
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    int retval;
    int EventSet = PAPI_NULL;
    long long values[4];

    /* --- 1. Initialize PAPI library --- */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed: %d\n", retval);
        exit(1);
    }

    /* --- 2. Create an EventSet --- */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_create_eventset failed: %d\n", retval);
        exit(1);
    }

    /* --- 3. Add events: cycles, instructions, cache misses --- */
    retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_event(PAPI_TOT_CYC) failed: %d\n", retval);
        exit(1);
    }

    retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_event(PAPI_TOT_INS) failed: %d\n", retval);
        exit(1);
    }

    retval = PAPI_add_event(EventSet, PAPI_L1_DCM);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_event(PAPI_L1_DCM) failed: %d\n", retval);
        exit(1);
    }

    retval = PAPI_add_event(EventSet, PAPI_L2_TCM);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_event(PAPI_L2_TCM) failed: %d\n", retval);
        exit(1);
    }

    /* --- 4. Initialize matrices and start counting --- */
    init_matrices();

    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_start failed: %d\n", retval);
        exit(1);
    }

    /* --- 5. Run the workload --- */
    matmul();

    /* --- 6. Stop counting and read values --- */
    retval = PAPI_stop(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_stop failed: %d\n", retval);
        exit(1);
    }

    /* --- 7. Print results --- */
    long long total_cycles = values[0];
    long long total_ins    = values[1];
    long long l1_misses    = values[2];
    long long l2_misses    = values[3];

    double ipc = (total_cycles > 0)
        ? (double)total_ins / (double)total_cycles
        : 0.0;

    printf("========================================\n");
    printf("  PAPI Low-Level API Sample Results\n");
    printf("========================================\n");
    printf("  Workload:       %dx%d matrix multiply\n", N, N);
    printf("  Total Cycles:   %lld\n", total_cycles);
    printf("  Total Instrs:   %lld\n", total_ins);
    printf("  IPC:            %.4f\n", ipc);
    printf("----------------------------------------\n");
    printf("  L1 Data Cache Misses:  %lld\n", l1_misses);
    printf("  L2 Total Cache Misses: %lld\n", l2_misses);
    printf("========================================\n");

    /* --- 8. Cleanup --- */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    return 0;
}