/*****************************************************************************
 * PAPI Multiplexing Sample
 *
 * Demonstrates how to monitor more events than available hardware counters
 * using PAPI's multiplexing feature.
 *
 * Key: PAPI_set_multiplex() must be called BEFORE PAPI_start().
 *      If multiplexing is not supported by the component, this sample
 *      falls back to demonstrating a basic multi-event measurement.
 *
 * Build: make papi_multiplex
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

int main(void)
{
    int retval, EventSet = PAPI_NULL;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init: %d\n", retval);
        exit(1);
    }

    /* Show component info */
    const PAPI_component_info_t *cmp = PAPI_get_component_info(0);
    printf("Component 0: %s\n", cmp->short_name);
    printf("Hardware counters: %d, max multiplex: %d\n",
           cmp->num_cntrs, cmp->num_mpx_cntrs);

    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_create_eventset: %d\n", retval);
        exit(1);
    }

    /* Assign eventset to component 0 (CPU component) */
    retval = PAPI_assign_eventset_component(EventSet, 0);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_assign_eventset_component: %d\n", retval);
        exit(1);
    }

    /* Try to enable multiplexing */
    retval = PAPI_set_multiplex(EventSet);
    int multiplex_enabled = (retval == PAPI_OK);
    if (!multiplex_enabled) {
        printf("Per-eventset multiplex not supported (%d), "
               "using direct counting.\n", retval);
    } else {
        printf("Multiplexing enabled.\n");
    }

    /* Add events — more than hardware counter limit to demo multiplex need */
    int events[] = {
        PAPI_TOT_CYC, PAPI_TOT_INS,
        PAPI_L1_DCM,  PAPI_L2_TCM,
        PAPI_TLB_DM,  PAPI_BR_MSP,
        PAPI_LD_INS,  PAPI_SR_INS,
    };
    int num_events = sizeof(events) / sizeof(events[0]);
    const char *names[] = {
        "PAPI_TOT_CYC", "PAPI_TOT_INS",
        "PAPI_L1_DCM",  "PAPI_L2_TCM",
        "PAPI_TLB_DM",  "PAPI_BR_MSP",
        "PAPI_LD_INS",  "PAPI_SR_INS",
    };

    retval = PAPI_add_events(EventSet, events, num_events);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_events: %s (%d)\n"
                "Hint: some events may conflict. "
                "Try enabling multiplexing.\n",
                PAPI_strerror(retval), retval);
        exit(1);
    }

    printf("Added %d events.\n\n", num_events);

    /* Start counting */
    PAPI_start(EventSet);

    /* Workload: heavy floating point loop */
    volatile double x = 0.0;
    for (int i = 0; i < 10000000; i++) {
        x += i * 0.001;
    }
    (void)x;

    /* Stop and read */
    long long values[num_events];
    PAPI_stop(EventSet, values);

    printf("Results (%s):\n",
           multiplex_enabled ? "multiplexed" : "direct");
    printf("----------------------------------------\n");
    for (int i = 0; i < num_events; i++) {
        printf("  %-18s  %lld\n", names[i], values[i]);
    }
    printf("----------------------------------------\n");

    /* Derived metrics */
    double ipc = (values[0] > 0) ? (double)values[1] / values[0] : 0.0;
    printf("  IPC:                  %.4f\n", ipc);

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();
    return 0;
}