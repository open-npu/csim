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
 *     fixed_config_t (60 bytes) + per-channel params + [add params] + [LUT]
 *   Weight blob (at weight_offset)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
 * Fixed part of per-layer descriptor (60 bytes).
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
    int8_t   _pad0;
    int16_t  clamp_min;
    int16_t  clamp_max;
    int8_t   in_zp;
    uint8_t  _pad1;
    uint16_t param_ch_count;  /* number of output channels with per-ch params */
    uint8_t  has_lut;         /* 1 = LUT data follows */
    uint8_t  has_add;         /* 1 = add_param_t follows */
    int8_t   residual_src;   /* -1=none, 0..N = layer index for Add input_b */
} __attribute__((packed)) fixed_config_t;

_Static_assert(sizeof(fixed_config_t) == 60, "fixed_config_t must be 60 bytes");

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
        cfg->clamp_min   = fc.clamp_min;
        cfg->clamp_max   = fc.clamp_max;
        cfg->in_zp       = fc.in_zp;
        cfg->residual_src = fc.residual_src;

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
        npu_concat(cfg, input, output);
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
        printf("  Layer %u: op=%d, in=[%d×%d×%d], out=[%d×%d×%d]\n",
               i, cfg->op_type,
               cfg->in_h, cfg->in_w, cfg->in_c,
               cfg->out_h, cfg->out_w, cfg->out_c);

        /* Allocate output */
        tensor_t output;
        if (cfg->post_ctrl & POST_INT16_OUT) {
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

        /* Resolve residual input for Add nodes */
        tensor_t *input_b = NULL;
        if (cfg->op_type == OP_ELTWISE_ADD && cfg->residual_src >= 0) {
            input_b = &layer_outputs[(int)cfg->residual_src];
        }

        /* Execute */
        int ret = execute_layer(cfg, &current, input_b, layer_weights, &output);
        if (ret != 0) {
            fprintf(stderr, "Error: layer %u execution failed\n", i);
            tensor_free(&output);
            tensor_free(&current);
            for (uint32_t j = 0; j < i; j++) tensor_free(&layer_outputs[j]);
            free(layer_outputs);
            free_model(layers, header.num_layers, weights);
            return 1;
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
        tensor_free(&layer_outputs[i]);
    }
    free(layer_outputs);
    free_model(layers, header.num_layers, weights);
    return 0;
}
