/*
 * Open-NPU C Functional Simulator
 * eltwise.c — Element-wise addition (bit-exact, NHWC)
 *
 * Supports: INT8/INT16
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_eltwise_add(const layer_config_t *cfg,
                     const tensor_t *input_a,
                     const tensor_t *input_b,
                     int32_t *output_acc)
{
    const int h = cfg->in_h;
    const int w = cfg->in_w;
    const int c = cfg->in_c;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            for (int ic = 0; ic < c; ic++) {
                int32_t a, b;
                if (is_int16) {
                    a = (int32_t)tensor_get_i16(input_a, ih, iw, ic);
                    b = (int32_t)tensor_get_i16(input_b, ih, iw, ic);
                } else {
                    a = (int32_t)tensor_get_i8(input_a, ih, iw, ic);
                    b = (int32_t)tensor_get_i8(input_b, ih, iw, ic);
                }
                int idx = ih * w * c + iw * c + ic;
                output_acc[idx] = a + b;
            }
        }
    }
}
