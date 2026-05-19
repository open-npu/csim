/*
 * Open-NPU C Functional Simulator
 * test_postproc.c — Unit tests for post-processing pipeline
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>
#include "npu_types.h"
#include "npu_postproc.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("  FAIL: %s (expected %d, got %d)\n", msg, (int)(b), (int)(a)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ─── Test 1: Bias only ─── */
static void test_postproc_bias(void)
{
    printf("Test: postproc bias only...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_BIAS_EN | POST_CLAMP_EN;
    cfg.bias_shift = 0;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;

    int32_t result = npu_postproc_single(&cfg, 100, 50, 0);
    ASSERT_EQ(result, 127, "bias+clamp saturate");  /* 150 clamped to 127 */

    result = npu_postproc_single(&cfg, 10, 5, 0);
    ASSERT_EQ(result, 15, "bias no clamp");
}

/* ─── Test 2: Shift with rounding ─── */
static void test_postproc_shift(void)
{
    printf("Test: postproc shift...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_SHIFT_EN | POST_CLAMP_EN;
    cfg.shift_bits = 4;
    cfg.round_en = 1;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;

    /* 100 + 8 (rounding) = 108, >> 4 = 6 */
    int32_t result = npu_postproc_single(&cfg, 100, 0, 0);
    ASSERT_EQ(result, 6, "shift 100>>4 rounded");

    /* 255 + 8 = 263, >> 4 = 16 */
    result = npu_postproc_single(&cfg, 255, 0, 0);
    ASSERT_EQ(result, 16, "shift 255>>4 rounded");

    /* Without rounding */
    cfg.round_en = 0;
    result = npu_postproc_single(&cfg, 100, 0, 0);
    ASSERT_EQ(result, 6, "shift 100>>4 truncate");

    result = npu_postproc_single(&cfg, 15, 0, 0);
    ASSERT_EQ(result, 0, "shift 15>>4 truncate");
}

/* ─── Test 3: Scale ─── */
static void test_postproc_scale(void)
{
    printf("Test: postproc scale...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_SCALE_EN | POST_CLAMP_EN;
    cfg.scale = 16384;  /* 0.5 in Q0.15 (16384/32768) */
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;

    /* 100 * 16384 / 32768 = 50 */
    int32_t result = npu_postproc_single(&cfg, 100, 0, 0);
    ASSERT_EQ(result, 50, "scale 100*0.5");

    cfg.scale = 32767;  /* ~1.0 in Q0.15 */
    result = npu_postproc_single(&cfg, 50, 0, 0);
    /* 50 * 32767 / 32768 ≈ 49 (integer truncation) */
    ASSERT_EQ(result, 49, "scale 50*~1.0");
}

/* ─── Test 4: ReLU via clamp ─── */
static void test_postproc_relu(void)
{
    printf("Test: postproc ReLU (clamp min=0)...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_CLAMP_EN;
    cfg.clamp_min = 0;
    cfg.clamp_max = 127;

    int32_t result = npu_postproc_single(&cfg, -50, 0, 0);
    ASSERT_EQ(result, 0, "ReLU negative → 0");

    result = npu_postproc_single(&cfg, 50, 0, 0);
    ASSERT_EQ(result, 50, "ReLU positive pass");

    result = npu_postproc_single(&cfg, 200, 0, 0);
    ASSERT_EQ(result, 127, "ReLU saturate 127");
}

/* ─── Test 5: LUT ─── */
static void test_postproc_lut(void)
{
    printf("Test: postproc LUT...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_LUT_EN | POST_CLAMP_EN;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.data_type = DTYPE_INT8;

    /* Fill LUT: identity mapping with offset */
    for (int i = 0; i < 256; i++) {
        cfg.lut_i8[i] = (int8_t)(i - 128);  /* identity */
    }
    /* Override some entries */
    cfg.lut_i8[128 + 10] = 99;  /* input 10 → output 99 */
    cfg.lut_i8[128 - 5]  = -50; /* input -5 → output -50 */

    /* Input value 10 → signed 10 → unsigned index 10+128=138 → lut[138]=99 */
    int32_t result = npu_postproc_single(&cfg, 10, 0, 0);
    ASSERT_EQ(result, 99, "LUT[10]=99");

    result = npu_postproc_single(&cfg, -5, 0, 0);
    ASSERT_EQ(result, -50, "LUT[-5]=-50");
}

/* ─── Test 6: Full pipeline (bias + shift + scale + clamp) ─── */
static void test_postproc_full_pipeline(void)
{
    printf("Test: postproc full pipeline...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_SCALE_EN | POST_CLAMP_EN;
    cfg.bias_shift = 0;
    cfg.shift_bits = 8;
    cfg.round_en = 1;
    cfg.scale = 32767;  /* ~1.0 */
    cfg.out_zp = 0;
    cfg.clamp_min = 0;    /* ReLU */
    cfg.clamp_max = 127;

    /* acc=1024, bias=256 → 1280 → >>8 (round: +128=1408, >>8=5) → *32767/32768≈5 → clamp=[0,127]→5 */
    int32_t result = npu_postproc_single(&cfg, 1024, 256, 0);
    /* 1024+256=1280, +128=1408, >>8=5, *32767>>15=4 (truncation) */
    ASSERT_EQ(result, 4, "full pipeline positive");

    /* Negative: acc=-2048, bias=0 → -2048 → +128=-1920 → >>8=-7 → *32767>>15=-6 → clamp=0 */
    result = npu_postproc_single(&cfg, -2048, 0, 0);
    ASSERT_EQ(result, 0, "full pipeline ReLU clamps negative");
}

/* ─── Test 7: Output zero point ─── */
static void test_postproc_zero_point(void)
{
    printf("Test: postproc output zero point...\n");

    layer_config_t cfg = {0};
    cfg.post_ctrl = POST_CLAMP_EN;
    cfg.out_zp = 10;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;

    int32_t result = npu_postproc_single(&cfg, 50, 0, 0);
    ASSERT_EQ(result, 60, "out_zp=10, 50+10=60");
}

/* ─── Main ─── */
int main(void)
{
    printf("=== Open-NPU Post-Processing Tests ===\n\n");

    test_postproc_bias();
    test_postproc_shift();
    test_postproc_scale();
    test_postproc_relu();
    test_postproc_lut();
    test_postproc_full_pipeline();
    test_postproc_zero_point();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
