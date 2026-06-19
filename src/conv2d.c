/*
 * Open-NPU C Functional Simulator
 * conv2d.c — Standard 2D convolution (bit-exact, NHWC)
 *
 * Simulates RTL systolic array: ARRAY_SIZE virtual PEs, each with
 * independent 40-bit signed accumulator. Kernel elements are distributed
 * round-robin across PEs (PE[k] gets elements k, k+ARRAY_SIZE, k+2*ARRAY_SIZE...).
 * After all elements, PE values are summed with 40-bit wrap.
 *
 * Weight layout: [out_c][kernel_h][kernel_w][in_c]
 * Input layout:  NHWC [in_h][in_w][in_c]
 * Output:        40-bit accumulator array [out_h][out_w][out_c]
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "npu_operators.h"
#include "npu_config.h"

/* RTL PE: 40-bit signed accumulator with signed wrap at each MAC */
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

    /* Weight is [oc][kh][kw][ic] */
    const int w_stride_kw = in_c;
    const int w_stride_kh = kw * in_c;
    const int w_stride_oc = kh * kw * in_c;

    const int16_t *weights_i16 = (const int16_t *)weights;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);
    const int k_depth = kh * kw * in_c;  /* total MACs per pixel per channel */

    int64_t pe[NPU_ARRAY_SIZE];  /* RTL systolic array: ARRAY_SIZE virtual PEs */

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                /* Initialize all PEs to 0 for this (pixel, channel) */
                for (int i = 0; i < NPU_ARRAY_SIZE; i++)
                    pe[i] = 0;

                int elem_idx = 0;

                for (int fh = 0; fh < kh; fh++) {
                    int ih = oh * sh - pad_t + fh * dh;
                    if (ih < 0 || ih >= in_h) { elem_idx += kw * in_c; continue; }

                    for (int fw = 0; fw < kw; fw++) {
                        int iw = ow * sw - pad_l + fw * dw;
                        if (iw < 0 || iw >= in_w) { elem_idx += in_c; continue; }

                        int w_offset = oc * w_stride_oc +
                                       fh * w_stride_kh +
                                       fw * w_stride_kw;

                        for (int ic = 0; ic < in_c; ic++) {
                            int16_t in_val = is_int16
                                ? tensor_get_i16(input, ih, iw, ic)
                                : (int16_t)(int8_t)tensor_get_i8(input, ih, iw, ic);
                            int16_t w_val  = is_int16
                                ? weights_i16[w_offset + ic]
                                : (int16_t)weights[w_offset + ic];

                            /* Round-robin across ARRAY_SIZE PEs */
                            int pi = elem_idx % NPU_ARRAY_SIZE;
                            pe[pi] += (int64_t)in_val * (int64_t)w_val;
                            pe[pi] = trunc40(pe[pi]);

                            elem_idx++;
                        }
                    }
                }

                /* Sum all PE values with 40-bit wrap (matches RTL dot_buf accumulation) */
                int64_t sum = 0;
                for (int i = 0; i < NPU_ARRAY_SIZE; i++)
                    sum += pe[i];
                sum = trunc40(sum);

                int out_idx = oh * out_w * out_c + ow * out_c + oc;
                output_acc[out_idx] = sum;

                if (getenv("DBG_ACC") && oc == 9 && oh == 0 && ow == 0)
                    fprintf(stderr, "[CSIM_ACC_TILE] pixel(0,0) ch9 acc=%ld\n", (long)sum);
            }
        }
    }

    (void)bias;
}
