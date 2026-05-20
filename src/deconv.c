/*
 * Open-NPU C Functional Simulator
 * deconv.c — Transposed convolution (bit-exact, NHWC)
 *
 * Implementation: insert zeros between input elements, then standard conv.
 * Weight layout: [out_c][kernel_h][kernel_w][in_c]
 *
 * Supports: INT8/INT16
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_deconv(const layer_config_t *cfg,
                const tensor_t *input,
                const int8_t *weights,
                const int32_t *bias,
                int64_t *output_acc)
{
    const int in_h   = cfg->in_h;
    const int in_w   = cfg->in_w;
    const int in_c   = cfg->in_c;
    const int out_h  = cfg->out_h;
    const int out_w  = cfg->out_w;
    const int out_c  = cfg->out_c;
    const int kh     = cfg->kernel_h;
    const int kw     = cfg->kernel_w;
    const int ins_h  = cfg->insert_h;
    const int ins_w  = cfg->insert_w;
    const int pad_t  = cfg->pad_top;
    const int pad_l  = cfg->pad_left;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    const int16_t *weights_i16 = (const int16_t *)weights;

    /* After zero-insertion, the "expanded" input size is: */
    const int exp_h = in_h + (in_h - 1) * ins_h;
    const int exp_w = in_w + (in_w - 1) * ins_w;

    /* Weight strides: [oc][kh][kw][ic] */
    const int w_stride_oc = kh * kw * in_c;
    const int w_stride_kh = kw * in_c;
    const int w_stride_kw = in_c;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                int64_t acc = 0;

                for (int fh = 0; fh < kh; fh++) {
                    int eh = oh + pad_t - fh;
                    if (eh < 0 || eh >= exp_h) continue;
                    if (eh % (ins_h + 1) != 0) continue;
                    int ih = eh / (ins_h + 1);

                    for (int fw = 0; fw < kw; fw++) {
                        int ew = ow + pad_l - fw;
                        if (ew < 0 || ew >= exp_w) continue;
                        if (ew % (ins_w + 1) != 0) continue;
                        int iw = ew / (ins_w + 1);

                        int w_base = oc * w_stride_oc +
                                     fh * w_stride_kh +
                                     fw * w_stride_kw;

                        for (int ic = 0; ic < in_c; ic++) {
                            if (is_int16) {
                                int16_t in_val = tensor_get_i16(input, ih, iw, ic);
                                int16_t w_val  = weights_i16[w_base + ic];
                                acc += (int64_t)in_val * (int64_t)w_val;
                            } else {
                                int8_t in_val = tensor_get_i8(input, ih, iw, ic);
                                int8_t w_val  = weights[w_base + ic];
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

    (void)bias;
    (void)exp_h;
    (void)exp_w;
}
