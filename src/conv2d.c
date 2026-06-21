/*
 * Open-NPU C Functional Simulator
 * conv2d.c — Standard 2D convolution (bit-exact, NHWC)
 *
 * Uses PRE-PACKED tile input data (loaded by DMA) for exact RTL match.
 * Simulates RTL systolic array: per-pass PE accumulation with
 * per-element trunc40 in both PE accumulation and PE sum reduction.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "npu_operators.h"
#include "npu_config.h"

static inline int64_t trunc40(int64_t v) {
    v = (int64_t)((uint64_t)v << 24 >> 24);  // extract bits [39:0]
    // Sign-extend bit 39 to match Verilog signed register wrap
    v <<= 24; v >>= 24;
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
    const int w_stride_oc = kh * kw * in_c;
    const int16_t *weights_i16 = (const int16_t *)weights;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);
    const int k_depth = kh * kw * in_c;

    int64_t pe[NPU_ARRAY_SIZE];
    int64_t dot_buf;
    int elem_idx;
    int fh, fw, ic;
    int start, remain, pi;
    int16_t in_val, w_val;
    int64_t pass_sum;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                dot_buf = 0;

                for (int pass = 0; pass * NPU_ARRAY_SIZE < k_depth; pass++) {
                    start = pass * NPU_ARRAY_SIZE;
                    remain = k_depth - start;
                    if (remain > NPU_ARRAY_SIZE) remain = NPU_ARRAY_SIZE;

                    for (int i = 0; i < NPU_ARRAY_SIZE; i++)
                        pe[i] = 0;

                    elem_idx = 0;
                    for (fh = 0; fh < kh; fh++) {
                        int ih = oh * sh - pad_t + fh * dh;
                        if (ih < 0 || ih >= in_h) continue;
                        for (fw = 0; fw < kw; fw++) {
                            int iw = ow * sw - pad_l + fw * dw;
                            if (iw < 0 || iw >= in_w) continue;
                            for (ic = 0; ic < in_c; ic++, elem_idx++) {
                                if (elem_idx < start || elem_idx >= start + remain)
                                    continue;
                                in_val = is_int16
                                    ? tensor_get_i16(input, ih, iw, ic)
                                    : (int16_t)tensor_get_i8(input, ih, iw, ic);
                                w_val = weights_i16[oc * w_stride_oc + fh * w_stride_kh + fw * w_stride_kw + ic];
                                pi = elem_idx - start; /* PE index within this pass */
                                pe[pi] += (int64_t)in_val * (int64_t)w_val;
                                pe[pi] = trunc40(pe[pi]);
                            }
                        }
                    }

                    /* Sum PEs with per-element trunc40 (RTL dot_acc) */
                    pass_sum = 0;
                    for (int i = 0; i < remain; i++)
                        pass_sum = trunc40(pass_sum + pe[i]);

                    /* Accumulate into dot_buf (RTL dot_buf update) */
                    dot_buf = trunc40(dot_buf + pass_sum);
                }

                output_acc[oh * out_w * out_c + ow * out_c + oc] = dot_buf;

                if (getenv("DBG_ACC") && oc == 9 && oh == 0 && ow == 0)
                    fprintf(stderr, "[CSIM_ACC_TILE] pixel(0,0) ch9 acc=%ld\n", (long)dot_buf);
            }
        }
    }
    (void)bias;
}
