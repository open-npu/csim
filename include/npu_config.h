/*
 * Open-NPU C Functional Simulator
 * npu_config.h — Hardware configuration constants
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_CONFIG_H
#define NPU_CONFIG_H

/* Default hardware configuration (matches 0.2T variant) */
#define NPU_ARRAY_SIZE       16       /* Systolic array dimension (16×16) */
#define NPU_NUM_ARRAYS       1        /* Number of arrays */
#define NPU_DW_CHANNELS      16       /* DW conv parallel channels */
#define NPU_SPAD_SIZE_KB     128      /* Total scratchpad size in KB */
#define NPU_HAS_INT16        1
#define NPU_HAS_LUT          1
#define NPU_HAS_IPU          0        /* IPU is V1.1 */

/* Derived constants */
#define NPU_MACS_PER_CYCLE   (NPU_ARRAY_SIZE * NPU_ARRAY_SIZE * NPU_NUM_ARRAYS)

/* Activation buffer: ping-pong, each bank = SPAD/4 */
#define NPU_ACT_BANK_SIZE    (NPU_SPAD_SIZE_KB * 1024 / 4)

/* Weight buffer: half of SPAD */
#define NPU_WEIGHT_BUF_SIZE  (NPU_SPAD_SIZE_KB * 1024 / 2)

/* Simulator memory pool (external SRAM simulation) */
#define SIM_MEMORY_SIZE      (4 * 1024 * 1024)  /* 4MB simulated external memory */

#endif /* NPU_CONFIG_H */
