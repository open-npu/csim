/*
 * Open-NPU C Functional Simulator
 * postproc.c — Per-channel requantize post-processing pipeline (bit-exact)
 *
 * Hardware pipeline order (matches CSR spec / PPU microarchitecture):
 *   acc(40b) → +bias_q[ch] → ×M[ch] → >>S[ch](+round) → +zp[ch] → clamp → ReLU/LUT → out
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "npu_postproc.h"

/* RTL PPU uses configurable ACC_WIDTH signed data path at Stage 1 (bias addition).
 * Simulate signed wrap: extract bits [ACC_WIDTH-1:0] then sign-extend.
 * ACC_WIDTH read from env var (default 40, set 44 for ResNet18). */
static int get_acc_width(void) {
    const char *env = getenv("ACC_WIDTH");
    return env ? atoi(env) : 40;
}

static inline void trunc_40bit(int64_t *v) {
    int w = get_acc_width();
    int shift = 64 - w;
    *v = (int64_t)((uint64_t)*v << shift >> shift);
    *v <<= shift; *v >>= shift;  // sign-extend bit w-1
}

/* ─── Internal: apply activation function ─── */
static inline int32_t apply_activation(const layer_config_t *cfg, int32_t val)
{
    if (cfg->post_ctrl & POST_RELU_EN) {
        if (val < 0) val = 0;
    } else if (cfg->post_ctrl & POST_RELU6_EN) {
        /* ReLU6: lower clamp to 0; upper clamp already handled by Stage 5
         * (converter sets clamp_max = round(6.0 / output_scale)) */
        if (val < 0) val = 0;
    }

    if (cfg->post_ctrl & POST_LUT_EN) {
        /* Map value to unsigned 8-bit index */
        uint8_t idx;
        if (cfg->post_ctrl & POST_INT16_OUT) {
            /* INT16: apply LUT_INPUT_SHIFT to map to 0-255 range */
            idx = (uint8_t)((val + 128) & 0xFF);
            val = (int32_t)cfg->lut_i16[idx];
        } else {
            idx = (uint8_t)((int8_t)val + 128);
            val = (int32_t)cfg->lut_i8[idx];
        }
    }

    return val;
}

/* ─── CONV_REQ mode: per-channel requantize ─── */
int32_t npu_postproc_perchannel(const layer_config_t *cfg, int64_t acc, int ch)
{
    const perchannel_param_t *p = &cfg->ch_params[ch];

    /* Stage 1: Add bias (RTL PPU wraps at 40-bit signed) */
    if (cfg->post_ctrl & POST_BIAS_EN) {
        acc += (int64_t)p->bias_q;
        /* RTL: s1_biased = {ACC_WIDTH}{s1_biased}, ACC_WIDTH=40, signed wrap */
        trunc_40bit((int64_t*)&acc);
    }

    /* Stage 2: Multiply by M (15-bit unsigned) — RTL product is PROD_W-bit signed */
    /* PROD_W = ACC_W + MULT_W + 1. SoC uses ACC_WIDTH=44 → PROD_W=60.
     * Product fits in 60 bits for all practical inputs (40-bit acc * 15-bit M). */
    int64_t product = acc * (int64_t)(p->M & 0x7FFF);

    /* Stage 3: Rounding right shift by S (6-bit, 0~63) */
    /* RTL PPU: PROD_W = ACC_W + MULT_W + 1. SoC uses ACC_WIDTH=44 → PROD_W=60. */
    uint8_t shift = p->S & 0x3F;
    int64_t result;
    if (shift > 0) {
        /* RTL: rounding add wraps at PROD_W bits. For S >= PROD_W, rounding = 0.
         * For S < PROD_W: rounded_v = product + (1 << (S-1)) in PROD_W-bit signed. */
        const int PROD_W = get_acc_width() + 15 + 1;  /* ACC_W + MULT_W + 1 */
        int64_t rbit = (shift < PROD_W) ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t rounded = product + rbit;
        result = rounded >> shift;
    } else {
        result = product;
    }

    /* Stage 3b: Intermediate 17-bit saturation (matches RTL PPU npu_ppu.v Stage 3)
     * RTL saturates the 55-bit shifted result to 17-bit signed [-65536, +65535]
     * before zp addition. This prevents large kernels (e.g. 7×7) from
     * accumulating values that exceed the PPU's intermediate precision. */
    if (result > 65535)
        result = 65535;
    else if (result < -65536)
        result = -65536;

    /* Stage 4: Add zero point */
    if (cfg->post_ctrl & POST_ZP_EN) {
        result += (int64_t)p->zp;
    }

    /* Stage 5: Clamp */
    if (result < (int64_t)cfg->clamp_min) result = (int64_t)cfg->clamp_min;
    if (result > (int64_t)cfg->clamp_max) result = (int64_t)cfg->clamp_max;

    /* Stage 6: Activation */
    int32_t out = apply_activation(cfg, (int32_t)result);
    /* DEBUG: dump ALL postproc calls for channel 0 */
    if (ch == 14 && getenv("DBG_POST")) {
        static int _dbg = 0;
        fprintf(stderr, "[CSIM_DBG] #%d acc_in=%ld M=%u S=%u zp=%d bias=%ld bias_en=%d prod=%ld out=%d\n",
                _dbg++,
                (long)acc, (unsigned)(p->M & 0x7FFF), (unsigned)(p->S & 0x3F),
                (int)p->zp, (long)p->bias_q,
                (cfg->post_ctrl & POST_BIAS_EN) ? 1 : 0,
                (long)product, out);
    }
    return out;
}

/* ─── ADD mode: dual rescale + sum ─── */
int32_t npu_postproc_add(const layer_config_t *cfg, int32_t val_a, int32_t val_b)
{
    const add_param_t *p = cfg->add_params;

    /* Rescale branch A: val_a × M_A >> S_A */
    int64_t prod_a = (int64_t)val_a * (int64_t)(p->M_A & 0x7FFF);
    uint8_t shift_a = p->S_A & 0x3F;
    int32_t rescaled_a;
    if (shift_a > 0) {
        rescaled_a = (int32_t)((prod_a + ((int64_t)1 << (shift_a - 1))) >> shift_a);
    } else {
        rescaled_a = (int32_t)prod_a;
    }

    /* Rescale branch B: val_b × M_B >> S_B */
    int64_t prod_b = (int64_t)val_b * (int64_t)(p->M_B & 0x7FFF);
    uint8_t shift_b = p->S_B & 0x3F;
    int32_t rescaled_b;
    if (shift_b > 0) {
        rescaled_b = (int32_t)((prod_b + ((int64_t)1 << (shift_b - 1))) >> shift_b);
    } else {
        rescaled_b = (int32_t)prod_b;
    }

    /* Sum */
    int64_t sum = (int64_t)rescaled_a + (int64_t)rescaled_b;

    /* Clamp */
    if (sum < (int64_t)cfg->clamp_min) sum = (int64_t)cfg->clamp_min;
    if (sum > (int64_t)cfg->clamp_max) sum = (int64_t)cfg->clamp_max;

    /* Activation */
    return apply_activation(cfg, (int32_t)sum);
}

/* ─── Full tensor post-processing (CONV_REQ / RELU_ONLY / PASSTHROUGH) ─── */
void npu_postprocess(const layer_config_t *cfg,
                     const int64_t *acc,
                     int num_h, int num_w, int num_c,
                     tensor_t *output)
{
    uint8_t ppu_mode = cfg->post_ctrl & POST_PPU_MODE_MASK;
    /* DEBUG: print dimensions and first few values */
    if (getenv("DBG_POST")) {
        fprintf(stderr, "[CSIM_DBG_PP] num_h=%d num_w=%d num_c=%d ppu_mode=%d first_acc=%ld first_val=%d\n",
                num_h, num_w, num_c, ppu_mode,
                (long)(acc[0]), (int)(acc[0]));
    }

    for (int h = 0; h < num_h; h++) {
        for (int w = 0; w < num_w; w++) {
            for (int c = 0; c < num_c; c++) {
                int idx = h * num_w * num_c + w * num_c + c;
                int32_t out_val;

                switch (ppu_mode) {
                case PPU_MODE_CONV_REQ:
                    out_val = npu_postproc_perchannel(cfg, acc[idx], c);
                    break;

                case PPU_MODE_RELU_ONLY: {
                    int32_t val = (int32_t)acc[idx];
                    if (val < (int32_t)cfg->clamp_min) val = (int32_t)cfg->clamp_min;
                    if (val > (int32_t)cfg->clamp_max) val = (int32_t)cfg->clamp_max;
                    out_val = apply_activation(cfg, val);
                    break;
                }

                case PPU_MODE_PASSTHROUGH:
                default:
                    out_val = (int32_t)acc[idx];
                    break;
                }

                /* Write output */
                if (cfg->post_ctrl & POST_INT16_OUT) {
                    tensor_set_i16(output, h, w, c, (int16_t)out_val);
                } else {
                    tensor_set_i8(output, h, w, c, (int8_t)out_val);
                }
            }
        }
    }
}

/* ─── Add mode full tensor post-processing ─── */
void npu_postprocess_add(const layer_config_t *cfg,
                         const tensor_t *input_a,
                         const tensor_t *input_b,
                         int num_h, int num_w, int num_c,
                         tensor_t *output)
{
    int is_int16 = (cfg->data_type == DTYPE_INT16);

    for (int h = 0; h < num_h; h++) {
        for (int w = 0; w < num_w; w++) {
            for (int c = 0; c < num_c; c++) {
                int32_t val_a, val_b;

                if (is_int16) {
                    val_a = (int32_t)tensor_get_i16(input_a, h, w, c);
                    val_b = (int32_t)tensor_get_i16(input_b, h, w, c);
                } else {
                    val_a = (int32_t)tensor_get_i8(input_a, h, w, c);
                    val_b = (int32_t)tensor_get_i8(input_b, h, w, c);
                }

                int32_t out_val = npu_postproc_add(cfg, val_a, val_b);

                if (cfg->post_ctrl & POST_INT16_OUT) {
                    tensor_set_i16(output, h, w, c, (int16_t)out_val);
                } else {
                    tensor_set_i8(output, h, w, c, (int8_t)out_val);
                }
            }
        }
    }
}
