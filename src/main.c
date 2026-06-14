/*
 * Open-NPU C Functional Simulator
 * main.c — Inference entry point (V2: per-channel requantize)
 *
 * Usage: npu_sim <model.bin> <input.bin> <output.bin>
 *
 * Model binary format "NPU1":
 *   Header (16 bytes):
 *     uint32 magic = 0x4E505531
 *     uint32 num_layers
 *     uint32 weight_offset
 *     uint32 weight_size
 *   Per-layer descriptors (variable length, concatenated):
 *     fixed_config_t (62 bytes) + per-channel params + [add params] + [LUT]
 *   Weight blob (at weight_offset)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "npu_types.h"
#include "npu_config.h"
#include "npu_operators.h"
#include "npu_postproc.h"
#include "npu_dma.h"

/* ─── Model binary format ─── */

#define MODEL_MAGIC 0x4E505531  /* "NPU1" */

typedef struct {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t weight_offset;
    uint32_t weight_size;
} model_header_t;

/*
 * Fixed part of per-layer descriptor (62 bytes).
 * Serialized in little-endian, packed.
 */
typedef struct {
    uint8_t  op_type;
    uint8_t  data_type;
    uint16_t in_h, in_w, in_c;
    uint16_t out_h, out_w, out_c;
    uint8_t  kernel_h, kernel_w;
    uint8_t  dilation_h, dilation_w;
    uint8_t  stride_h, stride_w;
    uint8_t  pad_top, pad_bottom, pad_left, pad_right;
    uint8_t  pool_mode;
    uint8_t  pool_h, pool_w;
    uint8_t  pool_stride_h, pool_stride_w;
    uint8_t  global_pool;
    uint8_t  resize_mode;
    uint8_t  scale_h, scale_w;
    uint8_t  insert_h, insert_w;
    uint16_t concat_offset;
    uint16_t concat_total_c;
    uint16_t tile_h, tile_w;
    uint16_t tile_num_h, tile_num_w;
    uint8_t  post_ctrl;
    uint8_t  sched_ctrl;       /* bit[0]: DB_EN (double-buffer enable) */
    int16_t  clamp_min;
    int16_t  clamp_max;
    int8_t   in_zp;
    uint8_t  _pad1;
    uint16_t param_ch_count;  /* number of output channels with per-ch params */
    uint8_t  has_lut;         /* 1 = LUT data follows */
    uint8_t  has_add;         /* 1 = add_param_t follows */
    int8_t   residual_src;   /* -1=none, 0..N = layer index for Add input_b */
    int16_t  input_src;      /* -1=previous layer, 0..N = layer index for input */
} __attribute__((packed)) fixed_config_t;

_Static_assert(sizeof(fixed_config_t) == 62, "fixed_config_t must be 62 bytes");

/* Diagnostic: verify field offsets with offsetof() */
#define CHECK_OFFSET(field, expected) \
    if (offsetof(fixed_config_t, field) != (expected)) { \
        fprintf(stderr, "LAYOUT BUG: " #field " at %zu, expected %d\n", \
                offsetof(fixed_config_t, field), (expected)); \
    }

static void check_fixed_config_layout(void) {
    CHECK_OFFSET(op_type,        0);
    CHECK_OFFSET(data_type,      1);
    CHECK_OFFSET(in_h,           2);
    CHECK_OFFSET(in_c,           6);
    CHECK_OFFSET(out_h,          8);
    CHECK_OFFSET(out_c,         12);
    CHECK_OFFSET(kernel_h,      14);
    CHECK_OFFSET(kernel_w,      15);
    CHECK_OFFSET(dilation_h,    16);
    CHECK_OFFSET(dilation_w,    17);
    CHECK_OFFSET(stride_h,      18);
    CHECK_OFFSET(stride_w,      19);
    CHECK_OFFSET(pad_top,       20);
    CHECK_OFFSET(pool_mode,     24);
    CHECK_OFFSET(resize_mode,   30);
    CHECK_OFFSET(concat_offset, 35);
    CHECK_OFFSET(tile_h,        39);
    CHECK_OFFSET(post_ctrl,     47);
    CHECK_OFFSET(sched_ctrl,    48);
    CHECK_OFFSET(clamp_min,     49);
    CHECK_OFFSET(clamp_max,     51);
    CHECK_OFFSET(in_zp,         53);
    CHECK_OFFSET(_pad1,         54);
    CHECK_OFFSET(param_ch_count,55);
    CHECK_OFFSET(has_lut,       57);
    CHECK_OFFSET(has_add,       58);
    CHECK_OFFSET(residual_src,  59);
    CHECK_OFFSET(input_src,     60);
}
#undef CHECK_OFFSET

/* ─── Load model from file ─── */

static int load_model(const char *path,
                      model_header_t *header,
                      layer_config_t **out_layers,
                      int8_t **out_weights)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open model file: %s\n", path);
        return -1;
    }

    if (fread(header, sizeof(*header), 1, f) != 1) {
        fprintf(stderr, "Error: failed to read model header\n");
        fclose(f);
        return -1;
    }

    if (header->magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: invalid model magic (expected 0x%08X, got 0x%08X)\n",
                MODEL_MAGIC, header->magic);
        fclose(f);
        return -1;
    }

    /* Allocate layer configs */
    layer_config_t *layers = (layer_config_t *)calloc(header->num_layers, sizeof(layer_config_t));
    if (!layers) {
        fclose(f);
        return -1;
    }

    /* Parse each layer descriptor */
    for (uint32_t i = 0; i < header->num_layers; i++) {
        fixed_config_t fc;
        if (fread(&fc, sizeof(fc), 1, f) != 1) {
            fprintf(stderr, "Error: failed to read layer %u config\n", i);
            goto fail;
        }

        layer_config_t *cfg = &layers[i];
        cfg->op_type     = fc.op_type;
        cfg->data_type   = fc.data_type;

        /* Diagnostic: print first layer deserialized fields */
        if (i == 0 && getenv("DUMP_LAYERS")) {
            fprintf(stderr, "[CSIM DIAG] Layer 0 raw bytes (first 62):");
            uint8_t raw[62];
            memcpy(raw, &fc, sizeof(fc));
            for (int j = 0; j < 62; j++) {
                if (j % 16 == 0) fprintf(stderr, "\n  %02d: ", j);
                fprintf(stderr, "%02x ", raw[j]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[CSIM DIAG] Layer 0 decoded: op=%d dtype=%d in=%dx%dx%d "
                    "out=%dx%dx%d kh=%d kw=%d sh=%d sw=%d "
                    "pad=[%d,%d,%d,%d] clamp=[%d,%d] in_zp=%d "
                    "post_ctrl=0x%02x param_ch_count=%d has_lut=%d has_add=%d "
                    "residual_src=%d input_src=%d\n",
                    fc.op_type, fc.data_type,
                    fc.in_h, fc.in_w, fc.in_c, fc.out_h, fc.out_w, fc.out_c,
                    fc.kernel_h, fc.kernel_w, fc.stride_h, fc.stride_w,
                    fc.pad_top, fc.pad_bottom, fc.pad_left, fc.pad_right,
                    fc.clamp_min, fc.clamp_max, fc.in_zp,
                    fc.post_ctrl, fc.param_ch_count, fc.has_lut, fc.has_add,
                    fc.residual_src, fc.input_src);
            if (fc.param_ch_count > 0 && fc.param_ch_count <= 4) {
                /* Peek at first few per-ch params */
                long cur = ftell(f);
                perchannel_param_t peek[4];
                if (fread(peek, sizeof(perchannel_param_t), fc.param_ch_count, f)
                    == fc.param_ch_count) {
                    fprintf(stderr, "[CSIM DIAG] First %d per-ch params:\n",
                            (int)fc.param_ch_count);
                    for (uint32_t k = 0; k < fc.param_ch_count; k++) {
                        fprintf(stderr, "  ch[%u]: M=0x%04x S=%u zp=%d bias=%ld\n",
                                k, peek[k].M, peek[k].S, peek[k].zp,
                                (long)peek[k].bias_q);
                    }
                }
                fseek(f, cur, SEEK_SET);
                /* Re-read properly after diagnostics */
            }
        }
        cfg->in_h        = fc.in_h;
        cfg->in_w        = fc.in_w;
        cfg->in_c        = fc.in_c;
        cfg->out_h       = fc.out_h;
        cfg->out_w       = fc.out_w;
        cfg->out_c       = fc.out_c;
        cfg->kernel_h    = fc.kernel_h;
        cfg->kernel_w    = fc.kernel_w;
        cfg->dilation_h  = fc.dilation_h;
        cfg->dilation_w  = fc.dilation_w;
        cfg->stride_h    = fc.stride_h;
        cfg->stride_w    = fc.stride_w;
        cfg->pad_top     = fc.pad_top;
        cfg->pad_bottom  = fc.pad_bottom;
        cfg->pad_left    = fc.pad_left;
        cfg->pad_right   = fc.pad_right;
        cfg->pool_mode   = fc.pool_mode;
        cfg->pool_h      = fc.pool_h;
        cfg->pool_w      = fc.pool_w;
        cfg->pool_stride_h = fc.pool_stride_h;
        cfg->pool_stride_w = fc.pool_stride_w;
        cfg->global_pool = fc.global_pool;
        cfg->resize_mode = fc.resize_mode;
        cfg->scale_h     = fc.scale_h;
        cfg->scale_w     = fc.scale_w;
        cfg->insert_h    = fc.insert_h;
        cfg->insert_w    = fc.insert_w;
        cfg->concat_offset  = fc.concat_offset;
        cfg->concat_total_c = fc.concat_total_c;
        cfg->tile_h      = fc.tile_h;
        cfg->tile_w      = fc.tile_w;
        cfg->tile_num_h  = fc.tile_num_h;
        cfg->tile_num_w  = fc.tile_num_w;
        cfg->post_ctrl   = fc.post_ctrl;
        cfg->sched_ctrl  = fc.sched_ctrl;
        cfg->clamp_min   = fc.clamp_min;
        cfg->clamp_max   = fc.clamp_max;
        cfg->in_zp       = fc.in_zp;
        cfg->residual_src = fc.residual_src;
        cfg->input_src    = fc.input_src;

        /* Read per-channel params */
        if (fc.param_ch_count > 0) {
            cfg->ch_params = (perchannel_param_t *)malloc(
                (size_t)fc.param_ch_count * sizeof(perchannel_param_t));
            if (!cfg->ch_params) goto fail;
            if (fread(cfg->ch_params, sizeof(perchannel_param_t),
                      fc.param_ch_count, f) != fc.param_ch_count) {
                fprintf(stderr, "Error: failed to read per-ch params for layer %u\n", i);
                goto fail;
            }
        }

        /* Read add params */
        if (fc.has_add) {
            cfg->add_params = (add_param_t *)malloc(sizeof(add_param_t));
            if (!cfg->add_params) goto fail;
            if (fread(cfg->add_params, sizeof(add_param_t), 1, f) != 1) {
                fprintf(stderr, "Error: failed to read add params for layer %u\n", i);
                goto fail;
            }
        }

        /* Read LUT */
        if (fc.has_lut) {
            if (fread(cfg->lut_i8, sizeof(cfg->lut_i8), 1, f) != 1) goto fail;
            if (fread(cfg->lut_i16, sizeof(cfg->lut_i16), 1, f) != 1) goto fail;
        }
    }

    /* Read weight blob */
    int8_t *weights = NULL;
    if (header->weight_size > 0) {
        fseek(f, header->weight_offset, SEEK_SET);
        weights = (int8_t *)malloc(header->weight_size);
        if (!weights) goto fail;
        if (fread(weights, 1, header->weight_size, f) != header->weight_size) {
            fprintf(stderr, "Error: failed to read weights\n");
            free(weights);
            goto fail;
        }
    }

    fclose(f);
    *out_layers = layers;
    *out_weights = weights;
    return 0;

fail:
    for (uint32_t i = 0; i < header->num_layers; i++) {
        free(layers[i].ch_params);
        free(layers[i].add_params);
    }
    free(layers);
    fclose(f);
    return -1;
}

/* ─── Free model resources ─── */

static void free_model(layer_config_t *layers, uint32_t num_layers, int8_t *weights)
{
    for (uint32_t i = 0; i < num_layers; i++) {
        free(layers[i].ch_params);
        free(layers[i].add_params);
    }
    free(layers);
    free(weights);
}

/* ─── Execute one layer ─── */

static int execute_layer(const layer_config_t *cfg,
                         tensor_t *input,
                         tensor_t *input_b,   /* for eltwise add (second branch) */
                         const int8_t *weights,
                         tensor_t *output)
{
    /* Eltwise Add: dedicated path via PPU ADD mode */
    if (cfg->op_type == OP_ELTWISE_ADD) {
        npu_postprocess_add(cfg, input, input_b,
                            cfg->out_h, cfg->out_w, cfg->out_c, output);
        return 0;
    }

    /* Concat: direct copy, no accumulator */
    if (cfg->op_type == OP_CONCAT) {
        /* For concat, input source is determined by residual_src:
         * residual_src >= 0: read from that layer's output (input_b)
         * residual_src < 0:  read from current input */
        tensor_t *src = (cfg->residual_src >= 0 && input_b) ? input_b : input;
        npu_concat(cfg, src, output);
        return 0;
    }

    /* All other operators: compute → accumulator → postprocess */
    int out_elements = cfg->out_h * cfg->out_w * cfg->out_c;
    int64_t *acc = (int64_t *)calloc(out_elements, sizeof(int64_t));
    if (!acc) {
        fprintf(stderr, "Error: failed to allocate accumulator buffer\n");
        return -1;
    }

    switch (cfg->op_type) {
    case OP_CONV2D:
        npu_conv2d(cfg, input, weights, NULL, acc);
        break;
    case OP_DW_CONV:
        npu_dwconv(cfg, input, weights, NULL, acc);
        break;
    case OP_FC:
        npu_fc(cfg, input, weights, NULL, acc);
        break;
    case OP_POOLING:
        npu_pooling(cfg, input, acc);
        break;
    case OP_RESIZE:
        npu_resize(cfg, input, acc);
        break;
    case OP_DECONV:
        npu_deconv(cfg, input, weights, NULL, acc);
        break;
    default:
        fprintf(stderr, "Error: unknown op_type %d\n", cfg->op_type);
        free(acc);
        return -1;
    }

    /* Post-processing: acc → output tensor */
    npu_postprocess(cfg, acc, cfg->out_h, cfg->out_w, cfg->out_c, output);

    free(acc);
    return 0;
}

/* ─── Tiled execution wrapper ─── */

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

static int execute_layer_tiled(const layer_config_t *cfg,
                               tensor_t *input,
                               tensor_t *input_b,
                               const int8_t *weights,
                               tensor_t *output)
{
    /* If no tiling configured, use direct (full-layer) execution */
    if (cfg->tile_num_h == 0 || cfg->tile_num_w == 0) {
        return execute_layer(cfg, input, input_b, weights, output);
    }

    /* Tiled execution: iterate over OC groups × spatial tiles */
    const int is_int16 = (cfg->data_type == DTYPE_INT16);
    const int elem_size = is_int16 ? 2 : 1;
    const int tile_h = cfg->tile_h;
    const int tile_w = cfg->tile_w;

    /* Compute tile_oc from weight buffer constraint */
    int tile_oc = cfg->out_c;
    int has_weights = (cfg->op_type == OP_CONV2D || cfg->op_type == OP_DECONV);
    if (has_weights) {
        int weight_per_oc = cfg->kernel_h * cfg->kernel_w * cfg->in_c * elem_size;
        if (weight_per_oc > 0) {
            tile_oc = MIN(cfg->out_c, (int)(NPU_WEIGHT_BUF_SIZE / (unsigned)weight_per_oc));
            if (tile_oc < 1) tile_oc = 1;
        }
    }

    /* Resolve effective kernel/stride/dilation for tiling.
     * For pooling, use pool_h/w/stride instead of conv kernel/stride. */
    int eff_kh, eff_kw, eff_sh, eff_sw, eff_dh, eff_dw;
    int eff_pad_top, eff_pad_left;
    if (cfg->op_type == OP_POOLING) {
        eff_kh = cfg->pool_h;
        eff_kw = cfg->pool_w;
        eff_sh = cfg->pool_stride_h ? cfg->pool_stride_h : 1;
        eff_sw = cfg->pool_stride_w ? cfg->pool_stride_w : 1;
        eff_dh = 1;
        eff_dw = 1;
        eff_pad_top = cfg->pad_top;
        eff_pad_left = cfg->pad_left;
    } else {
        eff_kh = cfg->kernel_h ? cfg->kernel_h : 1;
        eff_kw = cfg->kernel_w ? cfg->kernel_w : 1;
        eff_sh = cfg->stride_h ? cfg->stride_h : 1;
        eff_sw = cfg->stride_w ? cfg->stride_w : 1;
        eff_dh = cfg->dilation_h ? cfg->dilation_h : 1;
        eff_dw = cfg->dilation_w ? cfg->dilation_w : 1;
        eff_pad_top = cfg->pad_top;
        eff_pad_left = cfg->pad_left;
    }
    int kh_eff = (eff_kh - 1) * eff_dh + 1;
    int kw_eff = (eff_kw - 1) * eff_dw + 1;

    for (int oc_start = 0; oc_start < cfg->out_c; oc_start += tile_oc) {
        int cur_tile_oc = MIN(tile_oc, cfg->out_c - oc_start);

        for (int tr = 0; tr < cfg->tile_num_h; tr++) {
            for (int tc = 0; tc < cfg->tile_num_w; tc++) {
                /* Actual output tile spatial size (border tiles may be smaller) */
                int out_start_h = tr * tile_h;
                int out_start_w = tc * tile_w;
                int actual_out_h = MIN(tile_h, cfg->out_h - out_start_h);
                int actual_out_w = MIN(tile_w, cfg->out_w - out_start_w);

                /* Input tile dimensions (with halo) */
                int in_tile_h, in_tile_w;
                if (cfg->op_type == OP_POOLING) {
                    /* For pooling: extract exact input region without zp padding.
                     * Pooling handles padding via bounds checking internally. */
                    int raw_start_h = tr * tile_h * eff_sh - eff_pad_top;
                    int raw_start_w = tc * tile_w * eff_sw - eff_pad_left;
                    int raw_end_h = raw_start_h + actual_out_h * eff_sh + kh_eff - eff_sh;
                    int raw_end_w = raw_start_w + actual_out_w * eff_sw + kw_eff - eff_sw;
                    /* Clip to valid input range */
                    int clip_start_h = raw_start_h < 0 ? 0 : raw_start_h;
                    int clip_start_w = raw_start_w < 0 ? 0 : raw_start_w;
                    int clip_end_h = raw_end_h > cfg->in_h ? cfg->in_h : raw_end_h;
                    int clip_end_w = raw_end_w > cfg->in_w ? cfg->in_w : raw_end_w;
                    in_tile_h = clip_end_h - clip_start_h;
                    in_tile_w = clip_end_w - clip_start_w;
                    if (in_tile_h < 1) in_tile_h = 1;
                    if (in_tile_w < 1) in_tile_w = 1;
                } else {
                    in_tile_h = actual_out_h * eff_sh + kh_eff - eff_sh;
                    in_tile_w = actual_out_w * eff_sw + kw_eff - eff_sw;
                }

                /* Extract input tile from full input (includes halo + padding) */
                /* Build extract cfg with effective kernel/stride for dma_extract_tile.
                 * tile_h/tile_w must be the ORIGINAL (full) tile size, not border-clipped,
                 * because dma_extract_tile uses them for start position calculation.
                 * The copy extent is determined by tile_out->h/w (border-aware). */
                layer_config_t ext_cfg = *cfg;
                ext_cfg.tile_h = tile_h;
                ext_cfg.tile_w = tile_w;
                ext_cfg.kernel_h = eff_kh;
                ext_cfg.kernel_w = eff_kw;
                ext_cfg.stride_h = eff_sh;
                ext_cfg.stride_w = eff_sw;
                ext_cfg.dilation_h = eff_dh;
                ext_cfg.dilation_w = eff_dw;
                ext_cfg.pad_top = eff_pad_top;
                ext_cfg.pad_left = eff_pad_left;

                /* Extract input tile from full input */
                tensor_t tile_in;
                if (is_int16) {
                    tile_in = tensor_alloc_i16(in_tile_h, in_tile_w, cfg->in_c);
                } else {
                    tile_in = tensor_alloc_i8(in_tile_h, in_tile_w, cfg->in_c);
                }

                if (cfg->op_type == OP_POOLING) {
                    /* For pooling: direct copy of valid region (no zp padding) */
                    int raw_start_h = tr * tile_h * eff_sh - eff_pad_top;
                    int raw_start_w = tc * tile_w * eff_sw - eff_pad_left;
                    int clip_start_h = raw_start_h < 0 ? 0 : raw_start_h;
                    int clip_start_w = raw_start_w < 0 ? 0 : raw_start_w;
                    for (int th = 0; th < in_tile_h; th++) {
                        for (int tw = 0; tw < in_tile_w; tw++) {
                            for (int tcc = 0; tcc < cfg->in_c; tcc++) {
                                if (is_int16) {
                                    int16_t v = tensor_get_i16(input,
                                        clip_start_h + th, clip_start_w + tw, tcc);
                                    tensor_set_i16(&tile_in, th, tw, tcc, v);
                                } else {
                                    int8_t v = tensor_get_i8(input,
                                        clip_start_h + th, clip_start_w + tw, tcc);
                                    tensor_set_i8(&tile_in, th, tw, tcc, v);
                                }
                            }
                        }
                    }
                } else {
                    /* For conv/dw/etc: use DMA extract with halo + zp padding */
                    dma_extract_tile(&ext_cfg, input, tr, tc, &tile_in);
                }

                /* Also extract input_b tile for Add (same spatial position) */
                tensor_t tile_in_b = {0, 0, 0, NULL, NULL};
                if (input_b && cfg->op_type == OP_ELTWISE_ADD) {
                    if (is_int16) {
                        tile_in_b = tensor_alloc_i16(actual_out_h, actual_out_w, cfg->in_c);
                    } else {
                        tile_in_b = tensor_alloc_i8(actual_out_h, actual_out_w, cfg->in_c);
                    }
                    /* For Add, input_b is elementwise (no halo, no kernel).
                     * Use original tile_h for start position, actual_out_h for copy size. */
                    int b_start_h = tr * tile_h;
                    int b_start_w = tc * tile_w;
                    for (int bh = 0; bh < actual_out_h; bh++) {
                        for (int bw = 0; bw < actual_out_w; bw++) {
                            for (int bc = 0; bc < cfg->in_c; bc++) {
                                if (is_int16) {
                                    int16_t v = tensor_get_i16(input_b,
                                        b_start_h + bh, b_start_w + bw, bc);
                                    tensor_set_i16(&tile_in_b, bh, bw, bc, v);
                                } else {
                                    int8_t v = tensor_get_i8(input_b,
                                        b_start_h + bh, b_start_w + bw, bc);
                                    tensor_set_i8(&tile_in_b, bh, bw, bc, v);
                                }
                            }
                        }
                    }
                }

                /* Create tile-local config */
                layer_config_t tile_cfg = *cfg;
                tile_cfg.in_h = in_tile_h;
                tile_cfg.in_w = in_tile_w;
                tile_cfg.out_h = actual_out_h;
                tile_cfg.out_w = actual_out_w;
                tile_cfg.out_c = cur_tile_oc;

                /* Padding handling:
                 * For conv/dwconv: halo is already in tile, set pad=0.
                 * For pooling: pooling uses bounds-checking to skip padding pixels,
                 *   so we need to preserve pad for first/last tiles and let pooling
                 *   use the actual in_h for bounds. We set in_h to the full input
                 *   region size and adjust pad so pooling correctly skips edges. */
                if (cfg->op_type == OP_POOLING) {
                    /* For pooling: keep correct padding so bounds checking works */
                    int raw_start_h2 = tr * tile_h * eff_sh - eff_pad_top;
                    int raw_start_w2 = tc * tile_w * eff_sw - eff_pad_left;
                    tile_cfg.pad_top = (raw_start_h2 < 0) ? -raw_start_h2 : 0;
                    tile_cfg.pad_left = (raw_start_w2 < 0) ? -raw_start_w2 : 0;
                    tile_cfg.pad_bottom = 0;
                    tile_cfg.pad_right = 0;
                } else {
                    /* Conv/DW/etc: padding baked into extracted tile */
                    tile_cfg.pad_top = 0;
                    tile_cfg.pad_bottom = 0;
                    tile_cfg.pad_left = 0;
                    tile_cfg.pad_right = 0;
                }

                /* Allocate tile output */
                tensor_t tile_out;
                if (cfg->post_ctrl & POST_INT16_OUT) {
                    tile_out = tensor_alloc_i16(actual_out_h, actual_out_w, cur_tile_oc);
                } else {
                    tile_out = tensor_alloc_i8(actual_out_h, actual_out_w, cur_tile_oc);
                }

                /* Slice weights for this OC group */
                const int8_t *w_slice = weights;
                if (has_weights && weights) {
                    int w_per_oc = cfg->kernel_h * cfg->kernel_w * cfg->in_c * elem_size;
                    w_slice = weights + oc_start * w_per_oc;
                }

                /* Adjust per-channel params pointer for this OC group */
                perchannel_param_t *orig_ch_params = tile_cfg.ch_params;
                if (tile_cfg.ch_params) {
                    tile_cfg.ch_params = tile_cfg.ch_params + oc_start;
                }

                /* Execute single tile */
                tensor_t *b_ptr = tile_in_b.data || tile_in_b.data_i16 ? &tile_in_b : NULL;
                int ret = execute_layer(&tile_cfg, &tile_in, b_ptr, w_slice, &tile_out);

                /* Restore ch_params */
                tile_cfg.ch_params = orig_ch_params;

                if (ret != 0) {
                    tensor_free(&tile_in);
                    tensor_free(&tile_in_b);
                    tensor_free(&tile_out);
                    return ret;
                }

                /* Store tile output back to full output (with OC offset) */
                layer_config_t store_cfg = *cfg;
                store_cfg.tile_h = tile_h;
                store_cfg.tile_w = tile_w;
                dma_store_tile_oc(&store_cfg, &tile_out, tr, tc, oc_start, output);

                tensor_free(&tile_in);
                tensor_free(&tile_in_b);
                tensor_free(&tile_out);
            }
        }
    }

    return 0;
}

/* ─── Fused block execution (Conv1×1 → DW3×3 → Conv1×1) ─── */

static int execute_fused_block(const layer_config_t *cfg_a,  /* Conv1×1 expand */
                               const layer_config_t *cfg_b,  /* DW3×3 */
                               const layer_config_t *cfg_c,  /* Conv1×1 project */
                               tensor_t *input,
                               const int8_t *weights_a,
                               const int8_t *weights_b,
                               const int8_t *weights_c,
                               tensor_t *output)
{
    /*
     * Fused execution model (bit-exact with unfused):
     * For each spatial tile:
     *   1. Run Conv1×1 #1 on the expanded input region → valid_h × valid_w × c_mid
     *   2. Build DW input buffer: embed conv1 output at correct offset, fill borders
     *      with cfg_b->in_zp (matching unfused DW padding behavior)
     *   3. DW 3×3 (pad=0 on pre-padded input) → actual_out_h × actual_out_w × c_mid
     *   4. Conv1×1 #2 → actual_out_h × actual_out_w × c_out
     *   5. Store to output
     */

    const int tile_h = cfg_a->tile_h;
    const int tile_w = cfg_a->tile_w;
    const int tile_num_h = cfg_a->tile_num_h;
    const int tile_num_w = cfg_a->tile_num_w;

    /* Data type: 0=INT8, 1=INT16 */
    const int is_int16 = (cfg_a->data_type == 1);
    const int elem_size = is_int16 ? 2 : 1;
    (void)elem_size; /* used implicitly via alloc helpers */

    /* DW kernel info */
    const int dw_kh = cfg_b->kernel_h;
    const int dw_kw = cfg_b->kernel_w;
    const int dw_pad_top = cfg_b->pad_top;
    const int dw_pad_left = cfg_b->pad_left;

    /* Channel dimensions */
    const int c_in  = cfg_a->in_c;
    const int c_mid = cfg_a->out_c;
    const int c_out = cfg_c->out_c;

    /* Feature map spatial (Conv1×1 #1 output = DW input = feat_h × feat_w) */
    const int feat_h = cfg_a->out_h;
    const int feat_w = cfg_a->out_w;

    /* OC tiling for Conv1×1 #1 (expand) */
    int oc1_tile = c_mid;
    {
        int w_per_oc = cfg_a->kernel_h * cfg_a->kernel_w * c_in;
        if (w_per_oc > 0) {
            int max_oc = (int)(NPU_WEIGHT_BUF_SIZE / (unsigned)w_per_oc);
            if (max_oc < c_mid) oc1_tile = max_oc;
            if (oc1_tile < 1) oc1_tile = 1;
        }
    }

    /* OC tiling for Conv1×1 #2 (project) */
    int oc2_tile = c_out;
    {
        int w_per_oc = cfg_c->kernel_h * cfg_c->kernel_w * c_mid;
        if (w_per_oc > 0) {
            int max_oc = (int)(NPU_WEIGHT_BUF_SIZE / (unsigned)w_per_oc);
            if (max_oc < c_out) oc2_tile = max_oc;
            if (oc2_tile < 1) oc2_tile = 1;
        }
    }

    for (int tr = 0; tr < tile_num_h; tr++) {
        for (int tc = 0; tc < tile_num_w; tc++) {
            /* Actual output tile size (border handling) */
            int out_start_h = tr * tile_h;
            int out_start_w = tc * tile_w;
            int actual_out_h = MIN(tile_h, cfg_c->out_h - out_start_h);
            int actual_out_w = MIN(tile_w, cfg_c->out_w - out_start_w);

            /*
             * DW needs input of size (actual_out + K - 1) to produce actual_out.
             * This input comes from Conv1×1 #1 output, positioned at
             * (out_start - pad_top) .. (out_start + actual_out + pad_bottom - 1)
             * in the full feature map coordinate space.
             */
            int dw_in_h = actual_out_h + (dw_kh - 1);
            int dw_in_w = actual_out_w + (dw_kw - 1);
            int dw_in_start_h = out_start_h - dw_pad_top;
            int dw_in_start_w = out_start_w - dw_pad_left;

            /* Clip to valid feature map region for Conv1×1 computation */
            int valid_start_h = dw_in_start_h < 0 ? 0 : dw_in_start_h;
            int valid_start_w = dw_in_start_w < 0 ? 0 : dw_in_start_w;
            int valid_end_h = dw_in_start_h + dw_in_h;
            int valid_end_w = dw_in_start_w + dw_in_w;
            if (valid_end_h > feat_h) valid_end_h = feat_h;
            if (valid_end_w > feat_w) valid_end_w = feat_w;
            int valid_h = valid_end_h - valid_start_h;
            int valid_w = valid_end_w - valid_start_w;

            /* ── Step 1: Extract input for Conv1×1 #1 ── */
            tensor_t tile_in;
            if (is_int16) {
                tile_in = tensor_alloc_i16(valid_h, valid_w, c_in);
                for (int h = 0; h < valid_h; h++)
                    for (int w = 0; w < valid_w; w++)
                        for (int c = 0; c < c_in; c++)
                            tensor_set_i16(&tile_in, h, w, c,
                                tensor_get_i16(input, valid_start_h + h, valid_start_w + w, c));
            } else {
                tile_in = tensor_alloc_i8(valid_h, valid_w, c_in);
                for (int h = 0; h < valid_h; h++)
                    for (int w = 0; w < valid_w; w++)
                        for (int c = 0; c < c_in; c++)
                            tensor_set_i8(&tile_in, h, w, c,
                                tensor_get_i8(input, valid_start_h + h, valid_start_w + w, c));
            }

            /* ── Step 2: Conv1×1 #1 → conv1_out [valid_h × valid_w × c_mid] ── */
            tensor_t conv1_out;
            if (is_int16) {
                conv1_out = tensor_alloc_i16(valid_h, valid_w, c_mid);
            } else {
                conv1_out = tensor_alloc_i8(valid_h, valid_w, c_mid);
            }

            for (int oc_start = 0; oc_start < c_mid; oc_start += oc1_tile) {
                int cur_oc = MIN(oc1_tile, c_mid - oc_start);

                layer_config_t sub_a = *cfg_a;
                sub_a.in_h = valid_h;
                sub_a.in_w = valid_w;
                sub_a.out_h = valid_h;
                sub_a.out_w = valid_w;
                sub_a.out_c = cur_oc;
                sub_a.pad_top = sub_a.pad_bottom = sub_a.pad_left = sub_a.pad_right = 0;

                int acc_size = valid_h * valid_w * cur_oc;
                int64_t *acc = (int64_t *)calloc(acc_size, sizeof(int64_t));

                const int8_t *w_a = weights_a + oc_start * cfg_a->kernel_h * cfg_a->kernel_w * c_in * elem_size;

                perchannel_param_t *orig_ch = sub_a.ch_params;
                if (sub_a.ch_params) sub_a.ch_params = sub_a.ch_params + oc_start;

                npu_conv2d(&sub_a, &tile_in, w_a, NULL, acc);

                tensor_t tmp_out;
                if (is_int16) {
                    tmp_out = tensor_alloc_i16(valid_h, valid_w, cur_oc);
                } else {
                    tmp_out = tensor_alloc_i8(valid_h, valid_w, cur_oc);
                }
                npu_postprocess(&sub_a, acc, valid_h, valid_w, cur_oc, &tmp_out);
                free(acc);
                sub_a.ch_params = orig_ch;

                for (int h = 0; h < valid_h; h++) {
                    for (int w = 0; w < valid_w; w++) {
                        for (int c = 0; c < cur_oc; c++) {
                            if (is_int16) {
                                int16_t v = tensor_get_i16(&tmp_out, h, w, c);
                                tensor_set_i16(&conv1_out, h, w, oc_start + c, v);
                            } else {
                                int8_t v = tensor_get_i8(&tmp_out, h, w, c);
                                tensor_set_i8(&conv1_out, h, w, oc_start + c, v);
                            }
                        }
                    }
                }
                tensor_free(&tmp_out);
            }
            tensor_free(&tile_in);

            /* ── Step 3: Build DW input with border padding ── */
            tensor_t dw_input;
            if (is_int16) {
                dw_input = tensor_alloc_i16(dw_in_h, dw_in_w, c_mid);
                /* Fill with DW's in_zp (border padding value) */
                int16_t dw_pad_val = (int16_t)cfg_b->in_zp;
                int16_t *p = dw_input.data_i16;
                size_t n = (size_t)dw_in_h * dw_in_w * c_mid;
                for (size_t idx = 0; idx < n; idx++) p[idx] = dw_pad_val;
            } else {
                dw_input = tensor_alloc_i8(dw_in_h, dw_in_w, c_mid);
                /* Fill with DW's in_zp (border padding value) */
                int8_t dw_pad_val = cfg_b->in_zp;
                memset(dw_input.data, dw_pad_val, (size_t)dw_in_h * dw_in_w * c_mid);
            }

            /* Embed conv1_out at the correct offset within dw_input */
            int off_h = valid_start_h - dw_in_start_h;
            int off_w = valid_start_w - dw_in_start_w;
            for (int h = 0; h < valid_h; h++) {
                for (int w = 0; w < valid_w; w++) {
                    for (int c = 0; c < c_mid; c++) {
                        if (is_int16) {
                            int16_t v = tensor_get_i16(&conv1_out, h, w, c);
                            tensor_set_i16(&dw_input, off_h + h, off_w + w, c, v);
                        } else {
                            int8_t v = tensor_get_i8(&conv1_out, h, w, c);
                            tensor_set_i8(&dw_input, off_h + h, off_w + w, c, v);
                        }
                    }
                }
            }
            tensor_free(&conv1_out);

            /* ── Step 4: DW 3×3 (pad=0 on pre-padded input) → mid2 ── */
            tensor_t mid2;
            if (is_int16) {
                mid2 = tensor_alloc_i16(actual_out_h, actual_out_w, c_mid);
            } else {
                mid2 = tensor_alloc_i8(actual_out_h, actual_out_w, c_mid);
            }
            {
                layer_config_t sub_b = *cfg_b;
                sub_b.in_h = dw_in_h;
                sub_b.in_w = dw_in_w;
                sub_b.out_h = actual_out_h;
                sub_b.out_w = actual_out_w;
                sub_b.pad_top = sub_b.pad_bottom = sub_b.pad_left = sub_b.pad_right = 0;

                int acc_size = actual_out_h * actual_out_w * c_mid;
                int64_t *acc = (int64_t *)calloc(acc_size, sizeof(int64_t));

                npu_dwconv(&sub_b, &dw_input, weights_b, NULL, acc);
                npu_postprocess(&sub_b, acc, actual_out_h, actual_out_w, c_mid, &mid2);
                free(acc);
            }
            tensor_free(&dw_input);

            /* ── Step 5: Conv1×1 #2 (project) → tile_out ── */
            tensor_t tile_out;
            if (is_int16) {
                tile_out = tensor_alloc_i16(actual_out_h, actual_out_w, c_out);
            } else {
                tile_out = tensor_alloc_i8(actual_out_h, actual_out_w, c_out);
            }

            for (int oc_start = 0; oc_start < c_out; oc_start += oc2_tile) {
                int cur_oc = MIN(oc2_tile, c_out - oc_start);

                layer_config_t sub_c = *cfg_c;
                sub_c.in_h = actual_out_h;
                sub_c.in_w = actual_out_w;
                sub_c.out_h = actual_out_h;
                sub_c.out_w = actual_out_w;
                sub_c.out_c = cur_oc;
                sub_c.pad_top = sub_c.pad_bottom = sub_c.pad_left = sub_c.pad_right = 0;

                int acc_size = actual_out_h * actual_out_w * cur_oc;
                int64_t *acc = (int64_t *)calloc(acc_size, sizeof(int64_t));

                const int8_t *w_c = weights_c + oc_start * cfg_c->kernel_h * cfg_c->kernel_w * c_mid * elem_size;

                perchannel_param_t *orig_ch = sub_c.ch_params;
                if (sub_c.ch_params) sub_c.ch_params = sub_c.ch_params + oc_start;

                npu_conv2d(&sub_c, &mid2, w_c, NULL, acc);

                tensor_t tmp_out;
                if (is_int16) {
                    tmp_out = tensor_alloc_i16(actual_out_h, actual_out_w, cur_oc);
                } else {
                    tmp_out = tensor_alloc_i8(actual_out_h, actual_out_w, cur_oc);
                }
                npu_postprocess(&sub_c, acc, actual_out_h, actual_out_w, cur_oc, &tmp_out);
                free(acc);
                sub_c.ch_params = orig_ch;

                for (int h = 0; h < actual_out_h; h++) {
                    for (int w = 0; w < actual_out_w; w++) {
                        for (int c = 0; c < cur_oc; c++) {
                            if (is_int16) {
                                int16_t v = tensor_get_i16(&tmp_out, h, w, c);
                                tensor_set_i16(&tile_out, h, w, oc_start + c, v);
                            } else {
                                int8_t v = tensor_get_i8(&tmp_out, h, w, c);
                                tensor_set_i8(&tile_out, h, w, oc_start + c, v);
                            }
                        }
                    }
                }
                tensor_free(&tmp_out);
            }
            tensor_free(&mid2);

            /* ── Step 6: Store tile output to full output ── */
            layer_config_t store_cfg = *cfg_c;
            store_cfg.tile_h = tile_h;
            store_cfg.tile_w = tile_w;
            dma_store_tile_oc(&store_cfg, &tile_out, tr, tc, 0, output);

            tensor_free(&tile_out);
        }
    }

    return 0;
}

/* ─── Main ─── */

static void print_usage(const char *prog) {
    fprintf(stderr, "Open-NPU C Functional Simulator V2.0 (per-channel requantize)\n");
    fprintf(stderr, "Usage: %s <model.bin> <input.bin> <output.bin>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  model.bin   - Model file (NPU1 format)\n");
    fprintf(stderr, "  input.bin   - Input tensor (NCHW, INT8 or INT16)\n");
    fprintf(stderr, "  output.bin  - Output tensor (NCHW, INT8 or INT16)\n");
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path  = argv[1];
    const char *input_path  = argv[2];
    const char *output_path = argv[3];

    /* ─── Load model ─── */
    model_header_t header;
    layer_config_t *layers = NULL;
    int8_t *weights = NULL;

    if (load_model(model_path, &header, &layers, &weights) != 0) {
        return 1;
    }

    printf("Model: %u layers, %u bytes weights\n", header.num_layers, header.weight_size);

    /* ── Diagnostic: verify struct layout ── */
    check_fixed_config_layout();

    /* ─── Load input ─── */
    FILE *f_input = fopen(input_path, "rb");
    if (!f_input) {
        fprintf(stderr, "Error: cannot open input file: %s\n", input_path);
        free_model(layers, header.num_layers, weights);
        return 1;
    }

    layer_config_t *first = &layers[0];
    int is_int16_input = (first->data_type == DTYPE_INT16);
    size_t elem_size = is_int16_input ? 2 : 1;
    size_t input_elements = (size_t)first->in_h * first->in_w * first->in_c;
    size_t input_size = input_elements * elem_size;

    void *input_nchw = malloc(input_size);
    if (fread(input_nchw, 1, input_size, f_input) != input_size) {
        fprintf(stderr, "Error: failed to read input data (expected %zu bytes)\n", input_size);
        free(input_nchw);
        fclose(f_input);
        free_model(layers, header.num_layers, weights);
        return 1;
    }
    fclose(f_input);

    /* Convert input NCHW → NHWC */
    tensor_t current;
    if (is_int16_input) {
        current = tensor_alloc_i16(first->in_h, first->in_w, first->in_c);
        dma_nchw_to_nhwc_i16((int16_t *)input_nchw,
                             first->in_h, first->in_w, first->in_c, &current);
    } else {
        current = tensor_alloc_i8(first->in_h, first->in_w, first->in_c);
        dma_nchw_to_nhwc_i8((int8_t *)input_nchw,
                            first->in_h, first->in_w, first->in_c, &current);
    }
    free(input_nchw);

    /* ─── Layer-by-layer inference ─── */
    int8_t *weight_ptr = weights;

    /* Keep history of layer outputs for residual connections */
    tensor_t *layer_outputs = (tensor_t *)calloc(header.num_layers, sizeof(tensor_t));
    tensor_t input_tensor = current;  /* save reference to free later */

    for (uint32_t i = 0; i < header.num_layers; i++) {
        layer_config_t *cfg = &layers[i];

        /* ── Fused block detection ── */
        if ((cfg->sched_ctrl & SCHED_CTRL_FUSE_START) && i + 2 < header.num_layers) {
            layer_config_t *cfg_a = &layers[i];
            layer_config_t *cfg_b = &layers[i + 1];
            layer_config_t *cfg_c = &layers[i + 2];

            printf("  Layer %u-%u: FUSED BLOCK [%d×%d×%d] → [%d×%d×%d]\n",
                   i, i + 2,
                   cfg_a->in_h, cfg_a->in_w, cfg_a->in_c,
                   cfg_c->out_h, cfg_c->out_w, cfg_c->out_c);

            /* Resolve input source */
            if (cfg_a->input_src >= 0 && (uint32_t)cfg_a->input_src < i) {
                current = layer_outputs[(int)cfg_a->input_src];
            }

            /* Allocate output for the fused block (final layer's output) */
            tensor_t fused_output;
            if (cfg_c->post_ctrl & POST_INT16_OUT) {
                fused_output = tensor_alloc_i16(cfg_c->out_h, cfg_c->out_w, cfg_c->out_c);
            } else {
                fused_output = tensor_alloc_i8(cfg_c->out_h, cfg_c->out_w, cfg_c->out_c);
            }

            /* Compute weight pointers for all 3 sub-layers */
            int8_t *w_ptr_local = weight_ptr;
            /* Layer A (Conv1×1 expand) */
            size_t wa_bytes = (size_t)cfg_a->out_c * cfg_a->kernel_h * cfg_a->kernel_w * cfg_a->in_c * elem_size;
            int8_t *wa = w_ptr_local;
            w_ptr_local += wa_bytes;
            /* Layer B (DW3×3) */
            size_t wb_bytes = (size_t)cfg_b->in_c * cfg_b->kernel_h * cfg_b->kernel_w * elem_size;
            int8_t *wb = w_ptr_local;
            w_ptr_local += wb_bytes;
            /* Layer C (Conv1×1 project) */
            size_t wc_bytes = (size_t)cfg_c->out_c * cfg_c->kernel_h * cfg_c->kernel_w * cfg_c->in_c * elem_size;
            int8_t *wc = w_ptr_local;
            w_ptr_local += wc_bytes;

            /* Execute fused block */
            int ret = execute_fused_block(cfg_a, cfg_b, cfg_c,
                                          &current, wa, wb, wc, &fused_output);
            if (ret != 0) {
                fprintf(stderr, "Error: fused block %u-%u execution failed\n", i, i + 2);
                tensor_free(&fused_output);
                goto cleanup_fail;
            }

            /* Advance weight pointer past all 3 layers */
            weight_ptr = w_ptr_local;

            /* Store outputs: intermediate layers have no external output */
            layer_outputs[i] = (tensor_t){0, 0, 0, NULL, NULL};
            layer_outputs[i + 1] = (tensor_t){0, 0, 0, NULL, NULL};
            layer_outputs[i + 2] = fused_output;
            current = fused_output;

            /* Skip the next 2 layers */
            i += 2;
            continue;
        }

        printf("  Layer %u: op=%d, in=[%d×%d×%d], out=[%d×%d×%d]\n",
               i, cfg->op_type,
               cfg->in_h, cfg->in_w, cfg->in_c,
               cfg->out_h, cfg->out_w, cfg->out_c);

        /* Resolve input source: input_src >= 0 means read from that layer's output */
        if (cfg->input_src >= 0 && (uint32_t)cfg->input_src < i) {
            current = layer_outputs[(int)cfg->input_src];
        }

        /* Allocate output */
        tensor_t output;
        if (cfg->op_type == OP_CONCAT && i > 0 &&
            layers[i-1].op_type == OP_CONCAT &&
            layers[i-1].out_h == cfg->out_h &&
            layers[i-1].out_w == cfg->out_w &&
            layers[i-1].out_c == cfg->out_c) {
            /* Reuse previous Concat sub-layer's output buffer */
            output = layer_outputs[i-1];
        } else if (cfg->post_ctrl & POST_INT16_OUT) {
            output = tensor_alloc_i16(cfg->out_h, cfg->out_w, cfg->out_c);
        } else {
            output = tensor_alloc_i8(cfg->out_h, cfg->out_w, cfg->out_c);
        }

        /* Get weights for this layer (bias is now in ch_params) */
        int8_t *layer_weights = NULL;
        size_t weight_bytes = 0;

        switch (cfg->op_type) {
        case OP_CONV2D:
        case OP_DECONV:
            weight_bytes = (size_t)cfg->out_c * cfg->kernel_h * cfg->kernel_w * cfg->in_c * elem_size;
            break;
        case OP_DW_CONV:
            weight_bytes = (size_t)cfg->in_c * cfg->kernel_h * cfg->kernel_w * elem_size;
            break;
        case OP_FC:
            weight_bytes = (size_t)cfg->out_c * cfg->in_c * elem_size;
            break;
        default:
            break;
        }

        if (weight_bytes > 0) {
            layer_weights = weight_ptr;
            weight_ptr += weight_bytes;
        }

        /* Resolve residual input for Add and Concat nodes */
        tensor_t *input_b = NULL;
        if ((cfg->op_type == OP_ELTWISE_ADD || cfg->op_type == OP_CONCAT)
            && cfg->residual_src >= 0) {
            input_b = &layer_outputs[(int)cfg->residual_src];
        }

        /* Execute (tiled if tile_num_h/w > 0, else full-layer) */
        int ret = execute_layer_tiled(cfg, &current, input_b, layer_weights, &output);
        if (ret != 0) {
            fprintf(stderr, "Error: layer %u execution failed\n", i);
            tensor_free(&output);
            goto cleanup_fail;
        }

        /* Store output in history */
        layer_outputs[i] = output;
        /* current now points to latest output (don't free old current separately,
           it's already in layer_outputs from previous iteration) */
        current = output;

        /* Debug: dump intermediate layer outputs if DUMP_LAYERS env var is set */
        if (getenv("DUMP_LAYERS")) {
            char dump_path[256];
            snprintf(dump_path, sizeof(dump_path), "/tmp/csim_layer_%03u.bin", i);
            FILE *fdump = fopen(dump_path, "wb");
            if (fdump) {
                int is_out16 = (cfg->post_ctrl & POST_INT16_OUT) ? 1 : 0;
                size_t n = (size_t)cfg->out_h * cfg->out_w * cfg->out_c;
                if (is_out16) {
                    /* Write NHWC int16 data directly */
                    fwrite(output.data_i16, sizeof(int16_t), n, fdump);
                } else {
                    fwrite(output.data, sizeof(int8_t), n, fdump);
                }
                fclose(fdump);
                printf("    [DUMP] %s (%zu elements, %s)\n", dump_path, n, is_out16 ? "int16" : "int8");
            }
        }
    }

    if (0) {
cleanup_fail:
        tensor_free(&input_tensor);
        for (uint32_t j = 0; j < header.num_layers; j++) tensor_free(&layer_outputs[j]);
        free(layer_outputs);
        free_model(layers, header.num_layers, weights);
        return 1;
    }

    /* current is layer_outputs[num_layers-1], don't double free */

    /* ─── Save output (convert NHWC → NCHW) ─── */
    layer_config_t *last = &layers[header.num_layers - 1];
    int is_int16_output = (last->post_ctrl & POST_INT16_OUT) ? 1 : 0;
    size_t out_elements = (size_t)last->out_h * last->out_w * last->out_c;
    size_t out_elem_size = is_int16_output ? 2 : 1;
    size_t output_size = out_elements * out_elem_size;
    void *output_nchw = malloc(output_size);

    if (is_int16_output) {
        dma_nhwc_to_nchw_i16(&current, (int16_t *)output_nchw);
    } else {
        dma_nhwc_to_nchw_i8(&current, (int8_t *)output_nchw);
    }

    FILE *f_output = fopen(output_path, "wb");
    if (!f_output) {
        fprintf(stderr, "Error: cannot open output file: %s\n", output_path);
        free(output_nchw);
        tensor_free(&current);
        free_model(layers, header.num_layers, weights);
        return 1;
    }
    fwrite(output_nchw, 1, output_size, f_output);
    fclose(f_output);

    printf("Done. Output saved to: %s (%zu bytes)\n", output_path, output_size);

    /* Cleanup */
    free(output_nchw);
    tensor_free(&input_tensor);
    for (uint32_t i = 0; i < header.num_layers; i++) {
        /* Skip duplicate Concat buffers (shared with next sub-layer) */
        if (i + 1 < header.num_layers &&
            layers[i].op_type == OP_CONCAT &&
            layers[i+1].op_type == OP_CONCAT &&
            layer_outputs[i].data == layer_outputs[i+1].data) {
            continue;  /* will be freed when we reach the last sub-layer */
        }
        tensor_free(&layer_outputs[i]);
    }
    free(layer_outputs);
    free_model(layers, header.num_layers, weights);
    return 0;
}
