# PAPI Sample Makefile
#
# Uses PAPI installed at 3rdparty/papi/src/install/

CC      = gcc
CFLAGS  = -Wall -O2
PAPI_DIR = ../src/install
INCLUDES = -I $(PAPI_DIR)/include
LDFLAGS  = -L $(PAPI_DIR)/lib -Wl,-rpath,$(PAPI_DIR)/lib
LDLIBS   = -lpapi -lpfm

TARGETS = papi_low_level papi_multiplex papi_timeline_demo

.PHONY: all clean run

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LDLIBS)

run: papi_low_level
	./papi_low_level

clean:
	rm -f $(TARGETS)