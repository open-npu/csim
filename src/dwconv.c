/*
 * Open-NPU C Functional Simulator
 * dwconv.c — Depthwise convolution (bit-exact, NHWC)
 *
 * Weight layout: [channels][kernel_h][kernel_w]
 * Input layout:  NHWC [in_h][in_w][channels]
 * Output:        INT32 accumulator array [out_h][out_w][channels]
 *
 * Supports: INT8/INT16, dilation, stride, padding
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_dwconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc)
{
    const int out_h = cfg->out_h;
    const int out_w = cfg->out_w;
    const int ch    = cfg->in_c;  /* in_c == out_c for depthwise */
    const int kh    = cfg->kernel_h;
    const int kw    = cfg->kernel_w;
    const int dh    = cfg->dilation_h;
    const int dw    = cfg->dilation_w;
    const int sh    = cfg->stride_h;
    const int sw    = cfg->stride_w;
    const int pad_t = cfg->pad_top;
    const int pad_l = cfg->pad_left;
    const int in_h  = cfg->in_h;
    const int in_w  = cfg->in_w;

    const int w_per_channel = kh * kw;
    const int16_t *weights_i16 = (const int16_t *)weights;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < ch; c++) {
                int64_t acc = 0;

                for (int fh = 0; fh < kh; fh++) {
                    int ih = oh * sh - pad_t + fh * dh;
                    if (ih < 0 || ih >= in_h) continue;

                    for (int fw = 0; fw < kw; fw++) {
                        int iw = ow * sw - pad_l + fw * dw;
                        if (iw < 0 || iw >= in_w) continue;

                        int w_idx = c * w_per_channel + fh * kw + fw;

                        if (is_int16) {
                            int16_t in_val = tensor_get_i16(input, ih, iw, c);
                            int16_t w_val  = weights_i16[w_idx];
                            acc += (int64_t)in_val * (int64_t)w_val;
                        } else {
                            int8_t in_val = tensor_get_i8(input, ih, iw, c);
                            int8_t w_val  = weights[w_idx];
                            acc += (int64_t)in_val * (int64_t)w_val;
                        }
                    }
                }

                int out_idx = oh * out_w * ch + ow * ch + c;
                output_acc[out_idx] = acc;
            }
        }
    }

    (void)bias;
}
