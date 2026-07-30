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
    /* Resize coordinate mapping is GLOBAL (whole-tensor), even when tiled:
     * the source pixel of output row oh_global is (oh_global*full_in_h)/full_out_h.
     * When tiled, cfg->in_h/out_h describe only the current tile, so the full
     * dims and the tile origins come from the rsz_* context fields.
     * This mirrors RTL npu_compute.v S_RESIZE_COORD (global coord) + the
     * rsz_tile_i{h,w}_origin subtraction used for SRAM addressing. */
    const int tiled   = cfg->rsz_tiled;
    const int in_h    = tiled ? cfg->rsz_full_in_h  : cfg->in_h;
    const int in_w    = tiled ? cfg->rsz_full_in_w  : cfg->in_w;
    const int out_h   = tiled ? cfg->rsz_full_out_h : cfg->out_h;
    const int out_w   = tiled ? cfg->rsz_full_out_w : cfg->out_w;
    /* Local extent of this invocation's output buffer */
    const int loc_out_h = cfg->out_h;
    const int loc_out_w = cfg->out_w;
    /* Output-space origin of this tile (0 when non-tiled) */
    const int oh_base = tiled ? cfg->rsz_tile_oh_origin : 0;
    const int ow_base = tiled ? cfg->rsz_tile_ow_origin : 0;
    /* Input-space origin of the extracted tile buffer (0 when non-tiled) */
    const int ih_base = tiled ? cfg->rsz_tile_ih_origin : 0;
    const int iw_base = tiled ? cfg->rsz_tile_iw_origin : 0;

    const int ch    = cfg->in_c;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    if (cfg->resize_mode == 0) {
        /* Nearest neighbor */
        for (int oh = 0; oh < loc_out_h; oh++) {
            int ih = ((oh_base + oh) * in_h) / out_h;
            if (ih >= in_h) ih = in_h - 1;
            ih -= ih_base;   /* global input row → tile-local buffer row */

            for (int ow = 0; ow < loc_out_w; ow++) {
                int iw = ((ow_base + ow) * in_w) / out_w;
                if (iw >= in_w) iw = in_w - 1;
                iw -= iw_base;

                for (int c = 0; c < ch; c++) {
                    int64_t val;
                    if (is_int16) {
                        val = (int64_t)tensor_get_i16(input, ih, iw, c);
                    } else {
                        val = (int64_t)tensor_get_i8(input, ih, iw, c);
                    }
                    int out_idx = oh * loc_out_w * ch + ow * ch + c;
                    output_acc[out_idx] = val;
                }
            }
        }
    } else {
        /* Bilinear interpolation using fixed-point Q8.8 */
        for (int oh = 0; oh < loc_out_h; oh++) {
            int32_t src_h_q8;
            if (out_h > 1) {
                src_h_q8 = (int32_t)(oh_base + oh) * ((in_h - 1) << 8) / (out_h - 1);
            } else {
                src_h_q8 = 0;
            }
            int ih0 = src_h_q8 >> 8;
            int ih1 = ih0 + 1;
            if (ih1 >= in_h) ih1 = in_h - 1;
            int frac_h = src_h_q8 & 0xFF;
            ih0 -= ih_base;
            ih1 -= ih_base;

            for (int ow = 0; ow < loc_out_w; ow++) {
                int32_t src_w_q8;
                if (out_w > 1) {
                    src_w_q8 = (int32_t)(ow_base + ow) * ((in_w - 1) << 8) / (out_w - 1);
                } else {
                    src_w_q8 = 0;
                }
                int iw0 = src_w_q8 >> 8;
                int iw1 = iw0 + 1;
                if (iw1 >= in_w) iw1 = in_w - 1;
                int frac_w = src_w_q8 & 0xFF;
                iw0 -= iw_base;
                iw1 -= iw_base;

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

                    int out_idx = oh * loc_out_w * ch + ow * ch + c;
                    output_acc[out_idx] = (int64_t)val;
                }
            }
        }
    }
}
