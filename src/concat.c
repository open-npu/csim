/*
 * Open-NPU C Functional Simulator
 * concat.c — Channel-wise concatenation (NHWC) with optional per-branch requantize
 *
 * If cfg->add_params is set, each element is rescaled via:
 *   out = clamp( (val × M_A >> S_A) + round )
 * This aligns branches with different quantization scales to a unified output domain.
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
    const add_param_t *rq = cfg->add_params;   /* per-branch rescale (NULL = passthrough) */

    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            for (int ic = 0; ic < in_c; ic++) {
                if (is_int16) {
                    int16_t val = tensor_get_i16(input, ih, iw, ic);
                    if (rq) {
                        int64_t prod = (int64_t)val * (int64_t)(rq->M_A & 0x7FFF);
                        uint8_t shift = rq->S_A & 0x3F;
                        int32_t rescaled = shift > 0
                            ? (int32_t)((prod + ((int64_t)1 << (shift - 1))) >> shift)
                            : (int32_t)prod;
                        if (rescaled < cfg->clamp_min) rescaled = cfg->clamp_min;
                        if (rescaled > cfg->clamp_max) rescaled = cfg->clamp_max;
                        val = (int16_t)rescaled;
                    }
                    tensor_set_i16(output, ih, iw, offset + ic, val);
                } else {
                    int8_t val = tensor_get_i8(input, ih, iw, ic);
                    if (rq) {
                        int64_t prod = (int64_t)val * (int64_t)(rq->M_A & 0x7FFF);
                        uint8_t shift = rq->S_A & 0x3F;
                        int32_t rescaled = shift > 0
                            ? (int32_t)((prod + ((int64_t)1 << (shift - 1))) >> shift)
                            : (int32_t)prod;
                        if (rescaled < cfg->clamp_min) rescaled = cfg->clamp_min;
                        if (rescaled > cfg->clamp_max) rescaled = cfg->clamp_max;
                        val = (int8_t)rescaled;
                    }
                    tensor_set_i8(output, ih, iw, offset + ic, val);
                }
            }
        }
    }
}
