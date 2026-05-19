/*
 * Open-NPU C Functional Simulator
 * dma_sim.c — DMA simulation: layout conversion + tiling
 *
 * Simulates the DMA engine behavior:
 * - NCHW ↔ NHWC format conversion (TRANSPOSE_EN)
 * - Tile extraction with halo (for convolution overlap)
 * - Tile output store back to full tensor
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_dma.h"

/* ─── Layout Conversion ─── */

void dma_nchw_to_nhwc_i8(const int8_t *src, int h, int w, int c, tensor_t *dst)
{
    /* src: [C][H][W], dst: [H][W][C] */
    for (int ch = 0; ch < c; ch++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                int8_t val = src[ch * h * w + ih * w + iw];
                tensor_set_i8(dst, ih, iw, ch, val);
            }
        }
    }
}

void dma_nchw_to_nhwc_i16(const int16_t *src, int h, int w, int c, tensor_t *dst)
{
    for (int ch = 0; ch < c; ch++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                int16_t val = src[ch * h * w + ih * w + iw];
                tensor_set_i16(dst, ih, iw, ch, val);
            }
        }
    }
}

void dma_nhwc_to_nchw_i8(const tensor_t *src, int8_t *dst)
{
    /* src: [H][W][C], dst: [C][H][W] */
    int h = src->h, w = src->w, c = src->c;
    for (int ch = 0; ch < c; ch++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                dst[ch * h * w + ih * w + iw] = tensor_get_i8(src, ih, iw, ch);
            }
        }
    }
}

void dma_nhwc_to_nchw_i16(const tensor_t *src, int16_t *dst)
{
    int h = src->h, w = src->w, c = src->c;
    for (int ch = 0; ch < c; ch++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                dst[ch * h * w + ih * w + iw] = tensor_get_i16(src, ih, iw, ch);
            }
        }
    }
}

/* ─── Tiling ─── */

void dma_extract_tile(const layer_config_t *cfg,
                      const tensor_t *input,
                      int tile_row, int tile_col,
                      tensor_t *tile_out)
{
    const int tile_h = cfg->tile_h;
    const int tile_w = cfg->tile_w;
    const int ch     = cfg->in_c;
    const int in_h   = cfg->in_h;
    const int in_w   = cfg->in_w;

    /* Compute the halo needed for convolution */
    int halo_top    = cfg->pad_top;
    int halo_left   = cfg->pad_left;
    int halo_bottom = (cfg->kernel_h - 1) * cfg->dilation_h - cfg->pad_top;
    int halo_right  = (cfg->kernel_w - 1) * cfg->dilation_w - cfg->pad_left;

    /* Tile input region in the original input space (before padding) */
    int start_h = tile_row * tile_h - halo_top;
    int start_w = tile_col * tile_w - halo_left;

    /* Tile output dimensions include halo */
    int out_tile_h = tile_h + halo_top + halo_bottom;
    int out_tile_w = tile_w + halo_left + halo_right;

    /* Ensure tile_out is properly sized */
    /* tile_out should be allocated as [out_tile_h][out_tile_w][ch] */

    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int th = 0; th < out_tile_h; th++) {
        int src_h = start_h + th;
        for (int tw = 0; tw < out_tile_w; tw++) {
            int src_w = start_w + tw;
            for (int tc = 0; tc < ch; tc++) {
                if (is_int16) {
                    int16_t val;
                    if (src_h < 0 || src_h >= in_h || src_w < 0 || src_w >= in_w) {
                        val = (int16_t)cfg->in_zp;
                    } else {
                        val = tensor_get_i16(input, src_h, src_w, tc);
                    }
                    tensor_set_i16(tile_out, th, tw, tc, val);
                } else {
                    int8_t val;
                    if (src_h < 0 || src_h >= in_h || src_w < 0 || src_w >= in_w) {
                        val = cfg->in_zp;
                    } else {
                        val = tensor_get_i8(input, src_h, src_w, tc);
                    }
                    tensor_set_i8(tile_out, th, tw, tc, val);
                }
            }
        }
    }
}

void dma_store_tile(const layer_config_t *cfg,
                    const tensor_t *tile_result,
                    int tile_row, int tile_col,
                    tensor_t *output)
{
    const int tile_h = cfg->tile_h;
    const int tile_w = cfg->tile_w;
    const int out_h  = cfg->out_h;
    const int out_w  = cfg->out_w;
    const int ch     = cfg->out_c;

    /* Compute output tile size (may be smaller for border tiles) */
    int start_h = tile_row * tile_h;
    int start_w = tile_col * tile_w;
    int actual_h = tile_h;
    int actual_w = tile_w;
    if (start_h + actual_h > out_h) actual_h = out_h - start_h;
    if (start_w + actual_w > out_w) actual_w = out_w - start_w;

    const int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int th = 0; th < actual_h; th++) {
        for (int tw = 0; tw < actual_w; tw++) {
            for (int tc = 0; tc < ch; tc++) {
                if (is_int16) {
                    int16_t val = tensor_get_i16(tile_result, th, tw, tc);
                    tensor_set_i16(output, start_h + th, start_w + tw, tc, val);
                } else {
                    int8_t val = tensor_get_i8(tile_result, th, tw, tc);
                    tensor_set_i8(output, start_h + th, start_w + tw, tc, val);
                }
            }
        }
    }
}
