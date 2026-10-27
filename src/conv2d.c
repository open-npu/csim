/*
 * Open-NPU C Functional Simulator
 * conv2d.c — Standard 2D convolution (bit-exact, NHWC)
 *
 * Uses PRE-PACKED tile input data (loaded by DMA) for exact RTL match.
 * The 64-lane row computes output channels in parallel; each channel's dot
 * product is accumulated modulo ACC_WIDTH.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "npu_operators.h"

/* ACC_WIDTH is configurable. Default 40 (model_b), 44 for ResNet18 (C_in≤256).
 * CSIM reads it from environment variable ACC_WIDTH at runtime. */
static int get_acc_width(void) {
    const char *env = getenv("ACC_WIDTH");
    return env ? atoi(env) : 44;
}

static inline int64_t trunc_acc(int64_t v) {
    int w = get_acc_width();
    int shift = 64 - w;
    v = (int64_t)((uint64_t)v << shift >> shift);  // extract bits [w-1:0]
    v <<= shift; v >>= shift;                        // sign-extend bit w-1
    return v;
}

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

    const int w_stride_kw = in_c;
    const int w_stride_kh = kw * in_c;
    const int16_t *weights_i16 = (const int16_t *)weights;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);
    const int k_depth = kh * kw * in_c;

    int64_t dot_buf;
    int fh, fw, ic;
    int w_idx;
    int16_t in_val, w_val;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                dot_buf = 0;

                for (fh = 0; fh < kh; fh++) {
                    int ih = oh * sh - pad_t + fh * dh;
                    if (ih < 0 || ih >= in_h) continue;
                    for (fw = 0; fw < kw; fw++) {
                        int iw = ow * sw - pad_l + fw * dw;
                        if (iw < 0 || iw >= in_w) continue;
                        for (ic = 0; ic < in_c; ic++) {
                            in_val = is_int16
                                ? tensor_get_i16(input, ih, iw, ic)
                                : (int16_t)tensor_get_i8(input, ih, iw, ic);
                            w_idx = npu_conv_w_index(
                                cfg->wgt_layout, oc,
                                fh * w_stride_kh + fw * w_stride_kw + ic,
                                out_c, k_depth, is_int16 ? 2 : 1);
                            if (is_int16)
                                w_val = weights_i16[w_idx];
                            else
                                w_val = (int16_t)weights[w_idx];
                            dot_buf = trunc_acc(
                                dot_buf + (int64_t)in_val * (int64_t)w_val);
                        }
                    }
                }

                output_acc[oh * out_w * out_c + ow * out_c + oc] = dot_buf;

                if (getenv("DBG_ACC") && oc == 9 && oh == 0 && ow == 0)
                    fprintf(stderr, "[CSIM_ACC_TILE] pixel(0,0) ch9 acc=%ld\n", (long)dot_buf);
                if (getenv("DBG_ACC28") && oc == 12 && oh == 0 && ow == 0)
                    fprintf(stderr, "[CSIM_ACC28] pixel(0,0) chloc=%d oc=%d acc=%ld\n",
                            (int)oc, (int)cfg->out_c, (long)dot_buf);
            }
        }
    }
    (void)bias;
}
