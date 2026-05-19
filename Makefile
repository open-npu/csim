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

.PHONY: all clean test test_conv test_postproc

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ─── Tests ───

test: test_conv test_postproc

test_conv: $(LIB_OBJS) $(TESTDIR)/test_conv2d.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Running operator tests ──"
	@./$@

test_postproc: $(LIB_OBJS) $(TESTDIR)/test_postproc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "── Running post-processing tests ──"
	@./$@

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) test_conv test_postproc
