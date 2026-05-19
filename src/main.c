/*
 * Open-NPU C Functional Simulator
 * main.c — Inference entry point
 *
 * Usage: npu_sim <model.json> <weights.bin> <input.bin> <output.bin>
 *
 * For V1.0, uses a simple JSON format for layer descriptions.
 * This file demonstrates the layer-by-layer execution flow.
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

/* ─── Simple binary model format for V1.0 ─── */

/*
 * Model binary format (header):
 *   uint32_t magic;        // 0x4E505530 "NPU0"
 *   uint32_t num_layers;
 *   uint32_t weight_offset; // offset to weight data
 *   uint32_t weight_size;
 *   // followed by num_layers × layer_config_t (serialized)
 *   // followed by weight data
 */

#define MODEL_MAGIC 0x4E505530

typedef struct {
    uint32_t magic;
    uint32_t num_layers;
    uint32_t weight_offset;
    uint32_t weight_size;
} model_header_t;

/* ─── Execute one layer ─── */

static int execute_layer(const layer_config_t *cfg,
                         tensor_t *input,
                         tensor_t *input_b,   /* for eltwise */
                         const int8_t *weights,
                         const int32_t *bias,
                         tensor_t *output)
{
    int out_elements = cfg->out_h * cfg->out_w * cfg->out_c;
    int32_t *acc = (int32_t *)calloc(out_elements, sizeof(int32_t));
    if (!acc) {
        fprintf(stderr, "Error: failed to allocate accumulator buffer\n");
        return -1;
    }

    /* Dispatch to operator */
    switch (cfg->op_type) {
    case OP_CONV2D:
        npu_conv2d(cfg, input, weights, bias, acc);
        break;
    case OP_DW_CONV:
        npu_dwconv(cfg, input, weights, bias, acc);
        break;
    case OP_FC:
        npu_fc(cfg, input, weights, bias, acc);
        break;
    case OP_POOLING:
        npu_pooling(cfg, input, acc);
        break;
    case OP_ELTWISE_ADD:
        npu_eltwise_add(cfg, input, input_b, acc);
        break;
    case OP_RESIZE:
        npu_resize(cfg, input, acc);
        break;
    case OP_DECONV:
        npu_deconv(cfg, input, weights, bias, acc);
        break;
    case OP_CONCAT:
        /* Concat is special: directly copies, no accumulator */
        npu_concat(cfg, input, output);
        free(acc);
        return 0;
    default:
        fprintf(stderr, "Error: unknown op_type %d\n", cfg->op_type);
        free(acc);
        return -1;
    }

    /* Post-processing: acc → output tensor */
    npu_postprocess(cfg, acc, bias, cfg->out_h, cfg->out_w, cfg->out_c, output);

    free(acc);
    return 0;
}

/* ─── Main ─── */

static void print_usage(const char *prog) {
    fprintf(stderr, "Open-NPU C Functional Simulator V1.0\n");
    fprintf(stderr, "Usage: %s <model.bin> <input.bin> <output.bin>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  model.bin   - Model file (header + layer configs + weights)\n");
    fprintf(stderr, "  input.bin   - Input tensor (NCHW, INT8)\n");
    fprintf(stderr, "  output.bin  - Output tensor (NCHW, INT8)\n");
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
    FILE *f_model = fopen(model_path, "rb");
    if (!f_model) {
        fprintf(stderr, "Error: cannot open model file: %s\n", model_path);
        return 1;
    }

    model_header_t header;
    if (fread(&header, sizeof(header), 1, f_model) != 1) {
        fprintf(stderr, "Error: failed to read model header\n");
        fclose(f_model);
        return 1;
    }

    if (header.magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: invalid model magic (expected 0x%08X, got 0x%08X)\n",
                MODEL_MAGIC, header.magic);
        fclose(f_model);
        return 1;
    }

    printf("Model: %u layers, %u bytes weights\n", header.num_layers, header.weight_size);

    /* Read layer configs */
    layer_config_t *layers = (layer_config_t *)calloc(header.num_layers, sizeof(layer_config_t));
    if (!layers) {
        fprintf(stderr, "Error: failed to allocate layer configs\n");
        fclose(f_model);
        return 1;
    }
    if (fread(layers, sizeof(layer_config_t), header.num_layers, f_model) != header.num_layers) {
        fprintf(stderr, "Error: failed to read layer configs\n");
        free(layers);
        fclose(f_model);
        return 1;
    }

    /* Read weights */
    int8_t *weights = (int8_t *)malloc(header.weight_size);
    if (!weights) {
        fprintf(stderr, "Error: failed to allocate weight buffer\n");
        free(layers);
        fclose(f_model);
        return 1;
    }
    if (fread(weights, 1, header.weight_size, f_model) != header.weight_size) {
        fprintf(stderr, "Error: failed to read weights\n");
        free(weights);
        free(layers);
        fclose(f_model);
        return 1;
    }
    fclose(f_model);

    /* ─── Load input ─── */
    FILE *f_input = fopen(input_path, "rb");
    if (!f_input) {
        fprintf(stderr, "Error: cannot open input file: %s\n", input_path);
        free(weights);
        free(layers);
        return 1;
    }

    /* First layer defines input dimensions */
    layer_config_t *first = &layers[0];
    size_t input_size = (size_t)first->in_h * first->in_w * first->in_c;
    int8_t *input_nchw = (int8_t *)malloc(input_size);
    if (fread(input_nchw, 1, input_size, f_input) != input_size) {
        fprintf(stderr, "Error: failed to read input data (expected %zu bytes)\n", input_size);
        free(input_nchw);
        free(weights);
        free(layers);
        fclose(f_input);
        return 1;
    }
    fclose(f_input);

    /* Convert input NCHW → NHWC */
    tensor_t current = tensor_alloc_i8(first->in_h, first->in_w, first->in_c);
    dma_nchw_to_nhwc_i8(input_nchw, first->in_h, first->in_w, first->in_c, &current);
    free(input_nchw);

    /* ─── Layer-by-layer inference ─── */
    int8_t *weight_ptr = weights;

    for (uint32_t i = 0; i < header.num_layers; i++) {
        layer_config_t *cfg = &layers[i];
        printf("  Layer %u: op=%d, in=[%d×%d×%d], out=[%d×%d×%d]\n",
               i, cfg->op_type,
               cfg->in_h, cfg->in_w, cfg->in_c,
               cfg->out_h, cfg->out_w, cfg->out_c);

        /* Allocate output */
        tensor_t output = tensor_alloc_i8(cfg->out_h, cfg->out_w, cfg->out_c);

        /* Get weights and bias for this layer */
        int8_t *layer_weights = NULL;
        int32_t *layer_bias = NULL;
        size_t weight_bytes = 0;
        size_t bias_bytes = 0;

        switch (cfg->op_type) {
        case OP_CONV2D:
        case OP_DECONV:
            weight_bytes = (size_t)cfg->out_c * cfg->kernel_h * cfg->kernel_w * cfg->in_c;
            bias_bytes = (size_t)cfg->out_c * sizeof(int32_t);
            break;
        case OP_DW_CONV:
            weight_bytes = (size_t)cfg->in_c * cfg->kernel_h * cfg->kernel_w;
            bias_bytes = (size_t)cfg->in_c * sizeof(int32_t);
            break;
        case OP_FC:
            weight_bytes = (size_t)cfg->out_c * cfg->in_c;
            bias_bytes = (size_t)cfg->out_c * sizeof(int32_t);
            break;
        default:
            break;
        }

        if (weight_bytes > 0) {
            layer_weights = weight_ptr;
            weight_ptr += weight_bytes;
        }
        if (bias_bytes > 0) {
            layer_bias = (int32_t *)weight_ptr;
            weight_ptr += bias_bytes;
        }

        /* Execute */
        int ret = execute_layer(cfg, &current, NULL, layer_weights, layer_bias, &output);
        if (ret != 0) {
            fprintf(stderr, "Error: layer %u execution failed\n", i);
            tensor_free(&output);
            tensor_free(&current);
            free(weights);
            free(layers);
            return 1;
        }

        /* Swap current ↔ output for next layer */
        tensor_free(&current);
        current = output;
    }

    /* ─── Save output (convert NHWC → NCHW) ─── */
    layer_config_t *last = &layers[header.num_layers - 1];
    size_t output_size = (size_t)last->out_h * last->out_w * last->out_c;
    int8_t *output_nchw = (int8_t *)malloc(output_size);
    dma_nhwc_to_nchw_i8(&current, output_nchw);

    FILE *f_output = fopen(output_path, "wb");
    if (!f_output) {
        fprintf(stderr, "Error: cannot open output file: %s\n", output_path);
        free(output_nchw);
        tensor_free(&current);
        free(weights);
        free(layers);
        return 1;
    }
    fwrite(output_nchw, 1, output_size, f_output);
    fclose(f_output);

    printf("Done. Output saved to: %s (%zu bytes)\n", output_path, output_size);

    /* Cleanup */
    free(output_nchw);
    tensor_free(&current);
    free(weights);
    free(layers);
    return 0;
}
