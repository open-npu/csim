/*
 * Open-NPU C Functional Simulator
 * npu_postproc.h — Post-processing pipeline declarations
 *
 * Hardware pipeline (per CSR spec):
 *   acc(40b) → +bias_q[ch] → ×M[ch] → >>S[ch](+round) → +zp[ch] → clamp → ReLU/LUT → out
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_POSTPROC_H
#define NPU_POSTPROC_H

#include "npu_types.h"

/*
 * Per-channel post-processing (CONV_REQ mode).
 * Processes a single accumulator value for output channel `ch`.
 * Returns the final quantized output value.
 */
int32_t npu_postproc_perchannel(const layer_config_t *cfg, int64_t acc, int ch);

/*
 * Add mode post-processing for a single element.
 * Rescales two inputs with independent M/S, sums, then applies clamp+activation.
 */
int32_t npu_postproc_add(const layer_config_t *cfg, int32_t val_a, int32_t val_b);

/*
 * Full tensor post-processing (dispatches by PPU_MODE).
 * Handles CONV_REQ, RELU_ONLY, and PASSTHROUGH modes.
 *
 * acc: INT32 accumulator array [num_h * num_w * num_c] in NHWC order
 */
void npu_postprocess(const layer_config_t *cfg,
                     const int32_t *acc,
                     int num_h, int num_w, int num_c,
                     tensor_t *output);

/*
 * Add mode full tensor post-processing.
 * Reads from two input tensors, applies dual rescale + sum + activation.
 */
void npu_postprocess_add(const layer_config_t *cfg,
                         const tensor_t *input_a,
                         const tensor_t *input_b,
                         int num_h, int num_w, int num_c,
                         tensor_t *output);

#endif /* NPU_POSTPROC_H */
