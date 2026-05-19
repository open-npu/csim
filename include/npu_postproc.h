/*
 * Open-NPU C Functional Simulator
 * npu_postproc.h — Post-processing pipeline declarations
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_POSTPROC_H
#define NPU_POSTPROC_H

#include "npu_types.h"

/*
 * Apply the full post-processing pipeline to an array of INT32 accumulator
 * values, producing the final INT8 (or INT16) output tensor.
 *
 * Pipeline order (fixed, each step gated by post_ctrl bits):
 *   ACC(INT32) → +Bias → >>Shift → ×Scale → +OutZP → Clamp → LUT → Output
 *
 * Parameters:
 *   cfg        - layer configuration (post_ctrl, shift, scale, etc.)
 *   acc        - input accumulator array [num_elements], per-channel indexed
 *   bias       - bias array [out_c] (INT32), NULL if no bias
 *   num_h      - output height
 *   num_w      - output width
 *   num_c      - output channels
 *   output     - output tensor (pre-allocated, NHWC)
 */
void npu_postprocess(const layer_config_t *cfg,
                     const int32_t *acc,
                     const int32_t *bias,
                     int num_h, int num_w, int num_c,
                     tensor_t *output);

/*
 * Apply post-processing to a single value (used internally and for testing).
 */
int32_t npu_postproc_single(const layer_config_t *cfg,
                            int32_t acc,
                            int32_t bias_val,
                            int oc);

#endif /* NPU_POSTPROC_H */
