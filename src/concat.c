/*
 * Open-NPU C Functional Simulator
 * concat.c — Channel-wise concatenation (NHWC)
 *
 * Supports: INT8/INT16
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_concat(const layer_config_t *cfg,
                const tensor_t *input,
                tensor_t *output)
{
    const int h = cfg->in_h;
    const int w = cfg->in_w;
    const int in_c = cfg->in_c;
    const int offset = cfg->concat_offset;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            for (int ic = 0; ic < in_c; ic++) {
                if (is_int16) {
                    int16_t val = tensor_get_i16(input, ih, iw, ic);
                    tensor_set_i16(output, ih, iw, offset + ic, val);
                } else {
                    int8_t val = tensor_get_i8(input, ih, iw, ic);
                    tensor_set_i8(output, ih, iw, offset + ic, val);
                }
            }
        }
    }
}
