/*
 * Open-NPU C Functional Simulator
 * postproc.c — Post-processing pipeline (bit-exact)
 *
 * Pipeline order (matches hardware):
 *   INT32 acc → +Bias → >>Shift → ×Scale → +OutZP → Clamp → LUT → Output
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_postproc.h"

/* Saturate INT32 to INT8 range */
static inline int8_t sat_i8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

/* Saturate INT32 to INT16 range */
static inline int16_t sat_i16(int32_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

int32_t npu_postproc_single(const layer_config_t *cfg,
                            int32_t acc,
                            int32_t bias_val,
                            int oc)
{
    (void)oc; /* oc reserved for per-channel params in future */

    /* Step 1: Add bias */
    if (cfg->post_ctrl & POST_BIAS_EN) {
        int32_t shifted_bias = bias_val >> cfg->bias_shift;
        acc += shifted_bias;
    }

    /* Step 2: Right shift (requantization) */
    if (cfg->post_ctrl & POST_SHIFT_EN) {
        if (cfg->round_en && cfg->shift_bits > 0) {
            /* Rounding: add 0.5 ULP before shift */
            acc += (1 << (cfg->shift_bits - 1));
        }
        acc >>= cfg->shift_bits;
    }

    /* Step 3: Multiply by scale */
    if (cfg->post_ctrl & POST_SCALE_EN) {
        /* scale is a 16-bit multiplier; result is (acc * scale) >> 15 */
        int64_t tmp = (int64_t)acc * (int64_t)cfg->scale;
        acc = (int32_t)(tmp >> 15);
    }

    /* Step 4: Add output zero point */
    acc += (int32_t)cfg->out_zp;

    /* Step 5: Clamp */
    if (cfg->post_ctrl & POST_CLAMP_EN) {
        if (acc < (int32_t)cfg->clamp_min) acc = (int32_t)cfg->clamp_min;
        if (acc > (int32_t)cfg->clamp_max) acc = (int32_t)cfg->clamp_max;
    }

    /* Step 6: LUT activation */
    if (cfg->post_ctrl & POST_LUT_EN) {
        /* Map to unsigned index: signed INT8 [-128..127] → unsigned [0..255] */
        uint8_t idx = (uint8_t)(int8_t)acc + 128;
        if (cfg->data_type == DTYPE_INT8) {
            acc = (int32_t)cfg->lut_i8[idx];
        } else {
            acc = (int32_t)cfg->lut_i16[idx];
        }
    }

    return acc;
}

void npu_postprocess(const layer_config_t *cfg,
                     const int32_t *acc,
                     const int32_t *bias,
                     int num_h, int num_w, int num_c,
                     tensor_t *output)
{
    int total = num_h * num_w * num_c;
    (void)total;

    for (int h = 0; h < num_h; h++) {
        for (int w = 0; w < num_w; w++) {
            for (int c = 0; c < num_c; c++) {
                int idx = h * num_w * num_c + w * num_c + c;
                int32_t val = acc[idx];
                int32_t bias_val = bias ? bias[c] : 0;

                val = npu_postproc_single(cfg, val, bias_val, c);

                /* Write output */
                if (cfg->post_ctrl & POST_OUT_INT16) {
                    tensor_set_i16(output, h, w, c, sat_i16(val));
                } else {
                    tensor_set_i8(output, h, w, c, sat_i8(val));
                }
            }
        }
    }
}
