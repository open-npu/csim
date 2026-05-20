/*
 * Open-NPU C Functional Simulator
 * conv2d.c — Standard 2D convolution (bit-exact, NHWC)
 *
 * Weight layout: [out_c][kernel_h][kernel_w][in_c]
 * Input layout:  NHWC [in_h][in_w][in_c]
 * Output:        INT32 accumulator array [out_h][out_w][out_c]
 *
 * Supports: INT8/INT16, dilation, stride, asymmetric padding
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_conv2d(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc)
{
    const int out_h = cfg->out_h;
    const int out_w = cfg->out_w;
    const int out_c = cfg->out_c;
    const int in_c  = cfg->in_c;
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

    /* Weight is [oc][kh][kw][ic] */
    const int w_stride_kw = in_c;
    const int w_stride_kh = kw * in_c;
    const int w_stride_oc = kh * kw * in_c;

    const int16_t *weights_i16 = (const int16_t *)weights;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                int64_t acc = 0;

                for (int fh = 0; fh < kh; fh++) {
                    int ih = oh * sh - pad_t + fh * dh;
                    if (ih < 0 || ih >= in_h) continue;

                    for (int fw = 0; fw < kw; fw++) {
                        int iw = ow * sw - pad_l + fw * dw;
                        if (iw < 0 || iw >= in_w) continue;

                        int w_offset = oc * w_stride_oc +
                                       fh * w_stride_kh +
                                       fw * w_stride_kw;

                        for (int ic = 0; ic < in_c; ic++) {
                            if (is_int16) {
                                int16_t in_val = tensor_get_i16(input, ih, iw, ic);
                                int16_t w_val  = weights_i16[w_offset + ic];
                                acc += (int64_t)in_val * (int64_t)w_val;
                            } else {
                                int8_t in_val = tensor_get_i8(input, ih, iw, ic);
                                int8_t w_val  = weights[w_offset + ic];
                                acc += (int64_t)in_val * (int64_t)w_val;
                            }
                        }
                    }
                }

                int out_idx = oh * out_w * out_c + ow * out_c + oc;
                output_acc[out_idx] = acc;
            }
        }
    }

    (void)bias; /* bias applied in postproc */
}
