# Open-NPU C Functional Simulator — Makefile
# SPDX-License-Identifier: Apache-2.0

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

SRCDIR  = src
TESTDIR = test
SRCS    = $(wildcard $(SRCDIR)/*.c)
OBJS    = $(SRCS:.c=.o)
TARGET  = npu_sim

# Objects without main.o (for linking tests)
LIB_OBJS = $(filter-out $(SRCDIR)/main.o, $(OBJS))

.PHONY: all clean test test_conv test_postproc test_add_e2e

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ─── Tests ───

test: test_conv test_postproc test_add_e2e

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

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) test_conv test_postproc test_add_e2e
