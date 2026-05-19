/*
 * Open-NPU C Functional Simulator
 * npu_operators.h — Operator function declarations
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_OPERATORS_H
#define NPU_OPERATORS_H

#include "npu_types.h"

/*
 * All operators:
 *   - Input/output tensors in NHWC layout
 *   - Return raw INT32 accumulator results (before post-processing)
 *   - Weights in appropriate layout per operator
 *   - bias array: INT32, one per output channel
 */

/* Conv2D: standard 2D convolution
 * weights layout: [out_c][kernel_h][kernel_w][in_c] (NHWC-style for weight)
 */
void npu_conv2d(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int32_t *output_acc);  /* [out_h * out_w * out_c] */

/* Depthwise Conv: per-channel convolution
 * weights layout: [channels][kernel_h][kernel_w]
 */
void npu_dwconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int32_t *output_acc);

/* Fully Connected: equivalent to 1x1 conv with h=w=1
 * weights layout: [out_c][in_c]
 */
void npu_fc(const layer_config_t *cfg,
            const tensor_t *input,
            const int8_t *weights,
            const int32_t *bias,
            int32_t *output_acc);

/* Pooling: Max or Average
 * No weights needed. Output is INT32 (for Avg, sum before divide).
 */
void npu_pooling(const layer_config_t *cfg,
                 const tensor_t *input,
                 int32_t *output_acc);

/* Eltwise Add: element-wise addition of two tensors
 * Both inputs must have same dimensions.
 */
void npu_eltwise_add(const layer_config_t *cfg,
                     const tensor_t *input_a,
                     const tensor_t *input_b,
                     int32_t *output_acc);

/* Resize: nearest-neighbor or bilinear upsampling/downsampling */
void npu_resize(const layer_config_t *cfg,
                const tensor_t *input,
                int32_t *output_acc);

/* Deconv: transposed convolution (insert zeros then conv)
 * weights layout: same as conv2d [out_c][kernel_h][kernel_w][in_c]
 */
void npu_deconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int32_t *output_acc);

/* Concat: channel-wise concatenation
 * Copies input into output at the specified channel offset.
 */
void npu_concat(const layer_config_t *cfg,
                const tensor_t *input,
                tensor_t *output);

#endif /* NPU_OPERATORS_H */
