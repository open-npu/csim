/*
 * Open-NPU C Functional Simulator
 * pooling.c — Max and Average pooling (bit-exact, NHWC)
 *
 * Supports: INT8/INT16, Max/Avg, global pooling
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"
#include <limits.h>

void npu_pooling(const layer_config_t *cfg,
                 const tensor_t *input,
                 int32_t *output_acc)
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
                int32_t result;
                int count = 0;

                if (is_avg) {
                    result = 0;
                } else {
                    result = INT32_MIN;
                }

                for (int ph = 0; ph < pool_h; ph++) {
                    int ih = oh * pool_sh - pad_t + ph;
                    if (ih < 0 || ih >= in_h) continue;

                    for (int pw = 0; pw < pool_w; pw++) {
                        int iw = ow * pool_sw - pad_l + pw;
                        if (iw < 0 || iw >= in_w) continue;

                        int32_t val;
                        if (is_int16) {
                            val = (int32_t)tensor_get_i16(input, ih, iw, c);
                        } else {
                            val = (int32_t)tensor_get_i8(input, ih, iw, c);
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

                /* For AvgPool: integer division with rounding */
                if (is_avg && count > 0) {
                    if (result >= 0) {
                        result = (result + count / 2) / count;
                    } else {
                        result = (result - count / 2) / count;
                    }
                }

                int out_idx = oh * out_w * ch + ow * ch + c;
                output_acc[out_idx] = result;
            }
        }
    }
}
