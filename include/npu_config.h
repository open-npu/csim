/*
 * Open-NPU C Functional Simulator
 * npu_config.h — Hardware configuration constants
 *
 * All parameters can be overridden at compile time via -D flags:
 *   make CFLAGS_HW="-DNPU_ARRAY_SIZE=4 -DNPU_SPAD_SIZE_KB=32 -DNPU_HAS_INT16=0"
 *
 * Defaults match the 0.2T reference design (16×16, 128KB, INT8+INT16).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_CONFIG_H
#define NPU_CONFIG_H

/* ─── Primary parameters (override via -D) ─── */

#ifndef NPU_ARRAY_SIZE
#define NPU_ARRAY_SIZE       16      /* Systolic array dimension (N×N) */
#endif

#ifndef NPU_NUM_ARRAYS
#define NPU_NUM_ARRAYS       1       /* Number of arrays */
#endif

#ifndef NPU_DW_CHANNELS
#define NPU_DW_CHANNELS      NPU_ARRAY_SIZE  /* DW conv parallel channels */
#endif

#ifndef NPU_SPAD_SIZE_KB
#define NPU_SPAD_SIZE_KB     128     /* Total scratchpad size in KB */
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

#define NPU_MACS_PER_CYCLE   (NPU_ARRAY_SIZE * NPU_ARRAY_SIZE * NPU_NUM_ARRAYS)

/* Activation buffer: ping-pong, each bank = SPAD/4 */
#define NPU_ACT_BANK_SIZE    (NPU_SPAD_SIZE_KB * 1024 / 4)

/* Weight buffer: half of SPAD */
#define NPU_WEIGHT_BUF_SIZE  (NPU_SPAD_SIZE_KB * 1024 / 2)

/* Simulator memory pool (external SRAM simulation) */
#define SIM_MEMORY_SIZE      (4 * 1024 * 1024)  /* 4MB simulated external memory */

#endif /* NPU_CONFIG_H */
