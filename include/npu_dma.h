/*
 * Open-NPU C Functional Simulator
 * npu_dma.h — DMA simulation and data layout conversion
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_DMA_H
#define NPU_DMA_H

#include "npu_types.h"

/*
 * Convert tensor from NCHW to NHWC layout.
 * src: [C][H][W] contiguous in memory (NCHW)
 * dst: tensor_t with NHWC layout (pre-allocated)
 */
void dma_nchw_to_nhwc_i8(const int8_t *src, int h, int w, int c, tensor_t *dst);
void dma_nchw_to_nhwc_i16(const int16_t *src, int h, int w, int c, tensor_t *dst);

/*
 * Convert tensor from NHWC to NCHW layout.
 * src: tensor_t with NHWC layout
 * dst: [C][H][W] contiguous buffer (pre-allocated)
 */
void dma_nhwc_to_nchw_i8(const tensor_t *src, int8_t *dst);
void dma_nhwc_to_nchw_i16(const tensor_t *src, int16_t *dst);

/*
 * Extract a tile from the full input tensor (with padding).
 *
 * tile_row, tile_col: which tile (0-based index)
 * cfg: layer config (tile_h, tile_w, pad_*, stride_*, kernel_*)
 * input: full input tensor (NHWC)
 * tile_out: output tile tensor (pre-allocated, includes halo for conv)
 *
 * The tile includes the halo region needed for convolution (kernel overlap).
 * Padding pixels are filled with zero (or in_zp for quantized).
 */
void dma_extract_tile(const layer_config_t *cfg,
                      const tensor_t *input,
                      int tile_row, int tile_col,
                      tensor_t *tile_out);

/*
 * Store a computed tile back into the full output tensor.
 *
 * tile_row, tile_col: which tile
 * cfg: layer config
 * tile_result: computed tile output (NHWC)
 * output: full output tensor to write into
 */
void dma_store_tile(const layer_config_t *cfg,
                    const tensor_t *tile_result,
                    int tile_row, int tile_col,
                    tensor_t *output);

/*
 * Store a computed tile back into the full output tensor (with OC offset).
 *
 * Like dma_store_tile but writes only a subset of output channels.
 * tile_result has tile_oc channels which map to output channels
 * [oc_offset .. oc_offset+tile_oc-1].
 *
 * tile_row, tile_col: spatial tile index
 * oc_offset: first output channel index for this tile
 * cfg: layer config (tile_h, tile_w define spatial tile size)
 * tile_result: [tile_out_h][tile_out_w][tile_oc] NHWC
 * output: full output tensor [out_h][out_w][out_c]
 */
void dma_store_tile_oc(const layer_config_t *cfg,
                       const tensor_t *tile_result,
                       int tile_row, int tile_col,
                       int oc_offset,
                       tensor_t *output);

#endif /* NPU_DMA_H */
