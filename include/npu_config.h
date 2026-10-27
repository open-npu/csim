/*
 * Open-NPU C Functional Simulator
 * npu_config.h — Hardware configuration constants
 *
 * All parameters can be overridden at compile time via -D flags:
 *   make CFLAGS_HW="-DNPU_DW_CHANNELS=8 -DNPU_ACT_BANK_SIZE=24576 \
 *                   -DNPU_WEIGHT_BUF_SIZE=16384 -DNPU_HAS_INT16=0"
 *
 * Defaults match the 64-MAC row, 8-lane depthwise bank, and 164KB scratchpad.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_CONFIG_H
#define NPU_CONFIG_H

/* ─── Primary parameters (override via -D) ─── */

#ifndef NPU_MAC_LANES
#define NPU_MAC_LANES        64      /* Dense row MACs issued per cycle */
#endif

#ifndef NPU_DW_CHANNELS
#define NPU_DW_CHANNELS      8       /* Depthwise channels processed in parallel */
#endif

#ifndef NPU_SPAD_SIZE_KB
#define NPU_SPAD_SIZE_KB     164     /* HW_CONFIG-reported profile size */
#endif

#ifndef NPU_HAS_INT16
#define NPU_HAS_INT16        1
#endif

#ifndef NPU_HAS_LUT
#define NPU_HAS_LUT          1
#endif

#ifndef NPU_HAS_IPU
#define NPU_HAS_IPU          0       /* IPU is V1.1 */
#endif

/* ─── Derived constants ─── */

#define NPU_MACS_PER_CYCLE   NPU_MAC_LANES

/* Compiler-visible capacities; keep aligned with tools/hw_config.py. */
#ifndef NPU_ACT_BANK_SIZE
#define NPU_ACT_BANK_SIZE    (24 * 1024)
#endif

#ifndef NPU_WEIGHT_BUF_SIZE
#define NPU_WEIGHT_BUF_SIZE  (80 * 1024)
#endif

/* Simulator memory pool (external SRAM simulation) */
#define SIM_MEMORY_SIZE      (4 * 1024 * 1024)  /* 4MB simulated external memory */

#endif /* NPU_CONFIG_H */
