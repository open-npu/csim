/*
 * Open-NPU C Functional Simulator
 * resize.c — Nearest-neighbor and bilinear resize (bit-exact, NHWC)
 *
 * Supports: INT8/INT16, nearest/bilinear
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_resize(const layer_config_t *cfg,
                const tensor_t *input,
                int64_t *output_acc)
{
    const int in_h  = cfg->in_h;
    const int in_w  = cfg->in_w;
    const int out_h = cfg->out_h;
    const int out_w = cfg->out_w;
    const int ch    = cfg->in_c;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    if (cfg->resize_mode == 0) {
        /* Nearest neighbor */
        for (int oh = 0; oh < out_h; oh++) {
            int ih = (oh * in_h) / out_h;
            if (ih >= in_h) ih = in_h - 1;

            for (int ow = 0; ow < out_w; ow++) {
                int iw = (ow * in_w) / out_w;
                if (iw >= in_w) iw = in_w - 1;

                for (int c = 0; c < ch; c++) {
                    int64_t val;
                    if (is_int16) {
                        val = (int64_t)tensor_get_i16(input, ih, iw, c);
                    } else {
                        val = (int64_t)tensor_get_i8(input, ih, iw, c);
                    }
                    int out_idx = oh * out_w * ch + ow * ch + c;
                    output_acc[out_idx] = val;
                }
            }
        }
    } else {
        /* Bilinear interpolation using fixed-point Q8.8 */
        for (int oh = 0; oh < out_h; oh++) {
            int32_t src_h_q8;
            if (out_h > 1) {
                src_h_q8 = (int32_t)oh * ((in_h - 1) << 8) / (out_h - 1);
            } else {
                src_h_q8 = 0;
            }
            int ih0 = src_h_q8 >> 8;
            int ih1 = ih0 + 1;
            if (ih1 >= in_h) ih1 = in_h - 1;
            int frac_h = src_h_q8 & 0xFF;

            for (int ow = 0; ow < out_w; ow++) {
                int32_t src_w_q8;
                if (out_w > 1) {
                    src_w_q8 = (int32_t)ow * ((in_w - 1) << 8) / (out_w - 1);
                } else {
                    src_w_q8 = 0;
                }
                int iw0 = src_w_q8 >> 8;
                int iw1 = iw0 + 1;
                if (iw1 >= in_w) iw1 = in_w - 1;
                int frac_w = src_w_q8 & 0xFF;

                for (int c = 0; c < ch; c++) {
                    int32_t v00, v01, v10, v11;
                    if (is_int16) {
                        v00 = (int32_t)tensor_get_i16(input, ih0, iw0, c);
                        v01 = (int32_t)tensor_get_i16(input, ih0, iw1, c);
                        v10 = (int32_t)tensor_get_i16(input, ih1, iw0, c);
                        v11 = (int32_t)tensor_get_i16(input, ih1, iw1, c);
                    } else {
                        v00 = (int32_t)tensor_get_i8(input, ih0, iw0, c);
                        v01 = (int32_t)tensor_get_i8(input, ih0, iw1, c);
                        v10 = (int32_t)tensor_get_i8(input, ih1, iw0, c);
                        v11 = (int32_t)tensor_get_i8(input, ih1, iw1, c);
                    }

                    int32_t top = v00 * (256 - frac_w) + v01 * frac_w;
                    int32_t bot = v10 * (256 - frac_w) + v11 * frac_w;
                    int32_t val = top * (256 - frac_h) + bot * frac_h;
                    val = (val + (1 << 15)) >> 16;

                    int out_idx = oh * out_w * ch + ow * ch + c;
                    output_acc[out_idx] = (int64_t)val;
                }
            }
        }
    }
}
