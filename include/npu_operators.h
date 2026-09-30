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
 *   - Return raw INT64 accumulator results (before post-processing)
 *     Hardware uses 40-bit accumulator; we model with 64-bit for correctness.
 *   - Weights in appropriate layout per operator
 *   - bias array: INT32, one per output channel
 */

/*
 * Weight blob layouts (cfg->wgt_layout), mirroring tools/model_packer.py.
 *
 * OC_MAJOR  dense [OC][KH][KW][IC], depthwise [C][KH][KW]
 * K_MAJOR   dense: output channels grouped by NPU_MAC_LANES, K-major inside a
 *           group so the 64-lane row gets a tap's weights in one SRAM beat.
 *           depthwise: tap-major [KH][KW][C].
 *
 * Both layouts occupy the same number of bytes.
 */
#define NPU_WGT_LAYOUT_OC_MAJOR 0
#define NPU_WGT_LAYOUT_K_MAJOR  1
#define NPU_MAC_LANES           64

/* Element index of dense weight (oc, k), k = fh*KW*IC + fw*IC + ic. */
static inline int npu_conv_w_index(int layout, int oc, int k,
                                   int out_c, int k_depth)
{
    int group, group_w;
    if (layout != NPU_WGT_LAYOUT_K_MAJOR)
        return oc * k_depth + k;
    group = oc / NPU_MAC_LANES;
    group_w = out_c - group * NPU_MAC_LANES;
    if (group_w > NPU_MAC_LANES) group_w = NPU_MAC_LANES;
    return group * NPU_MAC_LANES * k_depth + k * group_w
         + (oc - group * NPU_MAC_LANES);
}

/* Element index of depthwise weight (c, tap), tap = fh*KW + fw. */
static inline int npu_dw_w_index(int layout, int c, int tap,
                                 int channels, int taps)
{
    if (layout != NPU_WGT_LAYOUT_K_MAJOR)
        return c * taps + tap;
    return tap * channels + c;
}

/* Conv2D: standard 2D convolution
 * weights layout: [out_c][kernel_h][kernel_w][in_c] (NHWC-style for weight)
 */
void npu_conv2d(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc);  /* [out_h * out_w * out_c] */

/* Depthwise Conv: per-channel convolution
 * weights layout: [channels][kernel_h][kernel_w]
 */
void npu_dwconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc);

/* Fully Connected: equivalent to 1x1 conv with h=w=1
 * weights layout: [out_c][in_c]
 */
void npu_fc(const layer_config_t *cfg,
            const tensor_t *input,
            const int8_t *weights,
            const int32_t *bias,
            int64_t *output_acc);

/* Pooling: Max or Average
 * No weights needed. Output is INT64 accumulator.
 */
void npu_pooling(const layer_config_t *cfg,
                 const tensor_t *input,
                 int64_t *output_acc);

/* Eltwise Add: element-wise addition of two tensors
 * Both inputs must have same dimensions.
 */
void npu_eltwise_add(const layer_config_t *cfg,
                     const tensor_t *input_a,
                     const tensor_t *input_b,
                     int64_t *output_acc);

/* Resize: nearest-neighbor or bilinear upsampling/downsampling */
void npu_resize(const layer_config_t *cfg,
                const tensor_t *input,
                int64_t *output_acc);

/* Deconv: transposed convolution (insert zeros then conv)
 * weights layout: same as conv2d [out_c][kernel_h][kernel_w][in_c]
 */
void npu_deconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc);

/* Concat: channel-wise concatenation
 * Copies input into output at the specified channel offset.
 */
void npu_concat(const layer_config_t *cfg,
                const tensor_t *input,
                tensor_t *output);

#endif /* NPU_OPERATORS_H */
