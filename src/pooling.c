/*
 * Open-NPU C Functional Simulator
 * pooling.c — Max and Average pooling (bit-exact, NHWC)
 *
 * Supports: INT8/INT16, Max/Avg, global pooling
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

/* Reciprocal LUT — must match npu_compute.v recip_pool() exactly */
static const int32_t pool_recip[] = {
    0,            /* 0 (unused) */
    0,            /* 1: identity, handled without multiplication */
    0x40000000,   /* 1/2 with >>31 (keeps multiplier signed-positive) */
    0x55555556,
    0x40000000,
    0x33333333,
    0x2AAAAAAB,
    0x24924925,
    0x20000000,
    0x1C71C71C,
    0x1999999A,
    0x1745D174,
    0x15555555,
    0x13B13B14,
    0x12492492,
    0x11111111,
    0x10000000,
    /* 17-24: fallback */
    0, 0, 0, 0, 0, 0, 0, 0,
    0x0A3D70A4,   /* 25 */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x071C71C7,   /* 36 */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x05405405,   /* 49 */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x04000000,   /* 64 */
};
static const int pool_recip_size = sizeof(pool_recip)/sizeof(pool_recip[0]);
#include <stdint.h>

void npu_pooling(const layer_config_t *cfg,
                 const tensor_t *input,
                 int64_t *output_acc)
{
    const int in_h  = cfg->in_h;
    const int in_w  = cfg->in_w;
    const int ch    = cfg->in_c;
    const int out_h = cfg->out_h;
    const int out_w = cfg->out_w;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    int pool_h, pool_w, pool_sh, pool_sw;

    if (cfg->global_pool) {
        pool_h  = in_h;
        pool_w  = in_w;
        pool_sh = in_h;
        pool_sw = in_w;
    } else {
        pool_h  = cfg->pool_h;
        pool_w  = cfg->pool_w;
        pool_sh = cfg->pool_stride_h;
        pool_sw = cfg->pool_stride_w;
    }

    const int pad_t = cfg->pad_top;
    const int pad_l = cfg->pad_left;
    const int is_avg = (cfg->pool_mode == 1);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < ch; c++) {
                int64_t result;
                int count = 0;

                if (is_avg) {
                    result = 0;
                } else {
                    result = INT64_MIN;
                }

                for (int ph = 0; ph < pool_h; ph++) {
                    int ih = oh * pool_sh - pad_t + ph;
                    if (ih < 0 || ih >= in_h) continue;

                    for (int pw = 0; pw < pool_w; pw++) {
                        int iw = ow * pool_sw - pad_l + pw;
                        if (iw < 0 || iw >= in_w) continue;

                        int64_t val;
                        if (is_int16) {
                            val = (int64_t)tensor_get_i16(input, ih, iw, c);
                        } else {
                            val = (int64_t)tensor_get_i8(input, ih, iw, c);
                        }

                        if (is_avg) {
                            result += val;
                            count++;
                        } else {
                            if (val > result) {
                                result = val;
                            }
                        }
                    }
                }

                /* For AvgPool: reciprocal LUT (matches RTL npu_compute.v S_POOL_DIV) */
                if (is_avg && count > 0) {
                    int32_t half_count = count >> 1;
                    int64_t rounded;
                    if (result >= 0)
                        rounded = (int64_t)result + half_count;
                    else
                        rounded = (int64_t)result - half_count;
                    if (count == 1) {
                        result = rounded;
                    } else {
                        int32_t recip = (count < pool_recip_size && pool_recip[count] != 0)
                                        ? pool_recip[count] : 0x01000000;
                        int64_t prod = rounded * (int64_t)recip;
                        result = (int32_t)(prod >> (count == 2 ? 31 : 32));
                    }
                }

                int out_idx = oh * out_w * ch + ow * ch + c;
                output_acc[out_idx] = result;
            }
        }
    }
}
