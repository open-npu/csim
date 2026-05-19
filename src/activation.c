/*
 * Open-NPU C Functional Simulator
 * activation.c — LUT generation helpers for activation functions
 *
 * The actual LUT lookup is done in postproc.c (POST_LUT_EN).
 * This file provides utility functions to pre-compute LUT tables
 * for common activations (sigmoid, tanh, etc.) during model setup.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_types.h"
#include <math.h>

/*
 * Generate sigmoid LUT for INT8.
 * Input range: [-128..127] mapped to float via input_scale and input_zp.
 * Output: quantized INT8 via output_scale and output_zp.
 *
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
void npu_gen_sigmoid_lut_i8(int8_t *lut,
                            float input_scale, int8_t input_zp,
                            float output_scale, int8_t output_zp)
{
    for (int i = 0; i < 256; i++) {
        /* i is unsigned index [0..255], maps to signed input [-128..127] */
        int8_t signed_input = (int8_t)(i - 128);
        float x = ((float)signed_input - (float)input_zp) * input_scale;
        float y = 1.0f / (1.0f + expf(-x));
        int32_t q = (int32_t)roundf(y / output_scale) + (int32_t)output_zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        lut[i] = (int8_t)q;
    }
}

/*
 * Generate tanh LUT for INT8.
 * tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 */
void npu_gen_tanh_lut_i8(int8_t *lut,
                         float input_scale, int8_t input_zp,
                         float output_scale, int8_t output_zp)
{
    for (int i = 0; i < 256; i++) {
        int8_t signed_input = (int8_t)(i - 128);
        float x = ((float)signed_input - (float)input_zp) * input_scale;
        float y = tanhf(x);
        int32_t q = (int32_t)roundf(y / output_scale) + (int32_t)output_zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        lut[i] = (int8_t)q;
    }
}

/*
 * Generate ReLU6 LUT (alternative to using clamp in post-proc).
 * ReLU6(x) = min(max(x, 0), 6)
 */
void npu_gen_relu6_lut_i8(int8_t *lut,
                          float input_scale, int8_t input_zp,
                          float output_scale, int8_t output_zp)
{
    for (int i = 0; i < 256; i++) {
        int8_t signed_input = (int8_t)(i - 128);
        float x = ((float)signed_input - (float)input_zp) * input_scale;
        float y = x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
        int32_t q = (int32_t)roundf(y / output_scale) + (int32_t)output_zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        lut[i] = (int8_t)q;
    }
}
