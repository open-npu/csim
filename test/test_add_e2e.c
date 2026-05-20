/*
 * Open-NPU C Functional Simulator
 * test_add_e2e.c — End-to-end test for Eltwise Add node
 *
 * Reads input_a, input_b (NCHW INT8), runs npu_postprocess_add, compares
 * bit-exact to Python-generated reference output.
 *
 * Usage: test_add_e2e <input_a.bin> <input_b.bin> <reference.bin> <H> <W> <C> <M_A> <S_A> <M_B> <S_B> <relu>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_types.h"
#include "npu_postproc.h"
#include "npu_dma.h"

int main(int argc, char *argv[])
{
    if (argc < 12) {
        fprintf(stderr, "Usage: %s <a.bin> <b.bin> <ref.bin> H W C M_A S_A M_B S_B relu\n", argv[0]);
        return 1;
    }

    const char *path_a   = argv[1];
    const char *path_b   = argv[2];
    const char *path_ref = argv[3];
    int H    = atoi(argv[4]);
    int W    = atoi(argv[5]);
    int C    = atoi(argv[6]);
    int M_A  = atoi(argv[7]);
    int S_A  = atoi(argv[8]);
    int M_B  = atoi(argv[9]);
    int S_B  = atoi(argv[10]);
    int relu = atoi(argv[11]);

    size_t num_elements = (size_t)H * W * C;
    size_t data_size = num_elements * sizeof(int8_t);

    /* Read input A (NCHW) */
    FILE *fa = fopen(path_a, "rb");
    if (!fa) { fprintf(stderr, "Cannot open %s\n", path_a); return 1; }
    int8_t *a_nchw = (int8_t *)malloc(data_size);
    if (fread(a_nchw, 1, data_size, fa) != data_size) {
        fprintf(stderr, "Failed to read input A\n"); return 1;
    }
    fclose(fa);

    /* Read input B (NCHW) */
    FILE *fb = fopen(path_b, "rb");
    if (!fb) { fprintf(stderr, "Cannot open %s\n", path_b); return 1; }
    int8_t *b_nchw = (int8_t *)malloc(data_size);
    if (fread(b_nchw, 1, data_size, fb) != data_size) {
        fprintf(stderr, "Failed to read input B\n"); return 1;
    }
    fclose(fb);

    /* Read reference output (NCHW) */
    FILE *fr = fopen(path_ref, "rb");
    if (!fr) { fprintf(stderr, "Cannot open %s\n", path_ref); return 1; }
    int8_t *ref_nchw = (int8_t *)malloc(data_size);
    if (fread(ref_nchw, 1, data_size, fr) != data_size) {
        fprintf(stderr, "Failed to read reference\n"); return 1;
    }
    fclose(fr);

    /* Convert NCHW → NHWC */
    tensor_t tensor_a = tensor_alloc_i8(H, W, C);
    tensor_t tensor_b = tensor_alloc_i8(H, W, C);
    dma_nchw_to_nhwc_i8(a_nchw, H, W, C, &tensor_a);
    dma_nchw_to_nhwc_i8(b_nchw, H, W, C, &tensor_b);

    /* Set up layer config for Add */
    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.op_type = OP_ELTWISE_ADD;
    cfg.data_type = DTYPE_INT8;
    cfg.in_h = H; cfg.in_w = W; cfg.in_c = C;
    cfg.out_h = H; cfg.out_w = W; cfg.out_c = C;
    cfg.post_ctrl = PPU_MODE_ADD;
    if (relu) cfg.post_ctrl |= POST_RELU_EN;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;

    add_param_t add_p = {
        .M_A = (uint16_t)M_A,
        .S_A = (uint8_t)S_A,
        ._reserved0 = 0,
        .M_B = (uint16_t)M_B,
        .S_B = (uint8_t)S_B,
        ._reserved1 = 0,
    };
    cfg.add_params = &add_p;

    /* Run Add post-processing */
    tensor_t output = tensor_alloc_i8(H, W, C);
    npu_postprocess_add(&cfg, &tensor_a, &tensor_b, H, W, C, &output);

    /* Convert output NHWC → NCHW for comparison */
    int8_t *out_nchw = (int8_t *)malloc(data_size);
    dma_nhwc_to_nchw_i8(&output, out_nchw);

    /* Compare bit-exact */
    int mismatches = 0;
    for (size_t i = 0; i < num_elements; i++) {
        if (out_nchw[i] != ref_nchw[i]) {
            if (mismatches < 10) {
                printf("  MISMATCH at idx %zu: got %d, expected %d\n",
                       i, (int)out_nchw[i], (int)ref_nchw[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("Add E2E test PASSED: %zu elements BIT-EXACT\n", num_elements);
    } else {
        printf("Add E2E test FAILED: %d/%zu mismatches\n", mismatches, num_elements);
    }

    /* Cleanup */
    free(a_nchw);
    free(b_nchw);
    free(ref_nchw);
    free(out_nchw);
    tensor_free(&tensor_a);
    tensor_free(&tensor_b);
    tensor_free(&output);

    return mismatches > 0 ? 1 : 0;
}
