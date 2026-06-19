/*
 * Open-NPU C Functional Simulator
 * conv2d.c — Standard 2D convolution (bit-exact, NHWC)
 *
 * Simulates RTL systolic array exactly:
 *   - ARRAY_SIZE virtual PEs per column
 *   - kernel elements distributed round-robin within each pass
 *   - PE accumulators cleared between passes (RTL DRAIN clears each PE)
 *   - Per-pass PE sum accumulated into dot_buf (matches RTL dot_buf)
 *   - All intermediate values wrapped at 40-bit signed
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "npu_operators.h"
#include "npu_config.h"

static inline int64_t trunc40(int64_t v) {
    return (int64_t)((uint64_t)v << 24 >> 24);
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
    const int passes = (k_depth + NPU_ARRAY_SIZE - 1) / NPU_ARRAY_SIZE;

    int64_t pe[NPU_ARRAY_SIZE]; /* virtual PEs */
    int64_t dot_buf;             /* per-channel accumulator */

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                dot_buf = 0;

                for (int pass = 0; pass < passes; pass++) {
                    int start = pass * NPU_ARRAY_SIZE;
                    int remain = (pass == passes - 1)
                        ? (k_depth - start) : NPU_ARRAY_SIZE;

                    /* Clear PEs for this pass */
                    for (int i = 0; i < NPU_ARRAY_SIZE; i++)
                        pe[i] = 0;

                    int elem_cnt = 0;
                    for (int fh = 0; fh < kh && elem_cnt < k_depth; fh++) {
                        int ih = oh * sh - pad_t + fh * dh;
                        if (ih < 0 || ih >= in_h) continue;

                        for (int fw = 0; fw < kw && elem_cnt < k_depth; fw++) {
                            int iw = ow * sw - pad_l + fw * dw;
                            if (iw < 0 || iw >= in_w) continue;

                            for (int ic = 0; ic < in_c && elem_cnt < k_depth; ic++, elem_cnt++) {
                                if (elem_cnt < start || elem_cnt >= start + remain)
                                    continue;

                                int16_t in_val = is_int16
                                    ? tensor_get_i16(input, ih, iw, ic)
                                    : (int16_t)(int8_t)tensor_get_i8(input, ih, iw, ic);
                                int w_off = oc * w_stride_oc + fh * w_stride_kh + fw * w_stride_kw + ic;
                                int16_t w_val = is_int16
                                    ? weights_i16[w_off]
                                    : (int16_t)weights[w_off];

                                int pi = elem_cnt % NPU_ARRAY_SIZE;
                                pe[pi] += (int64_t)in_val * (int64_t)w_val;
                                pe[pi] = trunc40(pe[pi]);
                            }
                        }
                    }

                    /* Sum active PEs (k_pass_remain elements) */
                    int64_t pass_sum = 0;
                    for (int i = 0; i < remain; i++)
                        pass_sum += pe[i];
                    pass_sum = trunc40(pass_sum);

                    /* Accumulate into dot_buf */
                    dot_buf = trunc40(dot_buf + pass_sum);
                }

                int out_idx = oh * out_w * out_c + ow * out_c + oc;
                output_acc[out_idx] = dot_buf;

                if (getenv("DBG_ACC") && oc == 9 && oh == 0 && ow == 0)
                    fprintf(stderr, "[CSIM_ACC_TILE] pixel(0,0) ch9 acc=%ld\n", (long)dot_buf);
            }
        }
    }

    (void)bias;
}
