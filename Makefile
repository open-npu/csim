# Open-NPU C Functional Simulator — Makefile
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   make                          # Default config (16×16, 128KB, INT8+INT16)
#   make CFLAGS_HW="-DNPU_ARRAY_SIZE=4 -DNPU_SPAD_SIZE_KB=32 -DNPU_HAS_INT16=0"
#
# Any NPU_* define in npu_config.h can be overridden via CFLAGS_HW.

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -O2 -Iinclude $(CFLAGS_HW)
# -MMD -MP: track include/*.h deps so a header change (e.g. layer_config_t
# layout) rebuilds every .o instead of silently mixing incompatible ABIs.
DEPFLAGS = -MMD -MP
LDFLAGS = -lm

SRCDIR  = src
TESTDIR = test
SRCS    = $(wildcard $(SRCDIR)/*.c)
OBJS    = $(SRCS:.c=.o)
DEPS    = $(SRCS:.c=.d)
TARGET  = npu_sim

# Objects without main.o (for linking tests)
LIB_OBJS = $(filter-out $(SRCDIR)/main.o, $(OBJS))

.PHONY: all clean test test_conv test_postproc test_add_e2e test_resize_tiling

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

-include $(DEPS)

# ─── Tests ───

test: test_conv test_postproc test_add_e2e test_resize_tiling

test_conv: $(LIB_OBJS) $(TESTDIR)/test_conv2d.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Running operator tests ──"
	@./$@

test_postproc: $(LIB_OBJS) $(TESTDIR)/test_postproc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Running post-processing tests ──"
	@./$@

test_add_e2e: $(LIB_OBJS) $(TESTDIR)/test_add_e2e.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Generating Add test data ──"
	@python3 $(TESTDIR)/gen_add_testdata.py
	@echo "── Running Add E2E tests ──"
	@failed=0; \
	while IFS=' ' read -r name H W C M_A S_A M_B S_B relu; do \
		./test_add_e2e testdata/add_$${name}_a.bin testdata/add_$${name}_b.bin \
			testdata/add_$${name}_ref.bin $$H $$W $$C $$M_A $$S_A $$M_B $$S_B $$relu || failed=$$((failed+1)); \
	done < testdata/add_tests.txt; \
	if [ $$failed -eq 0 ]; then echo "All Add E2E tests PASSED"; else echo "$$failed test(s) FAILED"; exit 1; fi

# Tiled Resize == untiled Resize == textbook global reference.
# Includes src/main.c directly (to reach the static execute_layer_tiled), so it
# links LIB_OBJS *without* main.o.
test_resize_tiling: $(LIB_OBJS) $(TESTDIR)/test_resize_tiling.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Running tiled Resize tests ──"
	@./$@

clean:
	rm -f $(SRCDIR)/*.o $(SRCDIR)/*.d $(TARGET) \
	      test_conv test_postproc test_add_e2e test_resize_tiling
