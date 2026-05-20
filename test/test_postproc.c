/*
 * Open-NPU C Functional Simulator
 * test_postproc.c — Unit tests for per-channel post-processing pipeline
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

/* ─── Test 1: Per-channel requantize basic ─── */
static void test_perchannel_basic(void)
{
    printf("Test: per-channel basic requantize...\n");

    perchannel_param_t params[2] = {
        { .M = 16384, .S = 15, .zp = 0, .bias_q = 0 },
        { .M = 8192,  .S = 14, .zp = 0, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = PPU_MODE_CONV_REQ;  /* no bias, no zp, just M*>>S */
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* ch0: 100 * 16384 = 1638400, + (1<<14) = 1654784, >> 15 = 50 */
    int32_t r = npu_postproc_perchannel(&cfg, 100, 0);
    ASSERT_EQ(r, 50, "ch0: 100*16384>>15 = 50");

    /* ch1: 200 * 8192 = 1638400, + (1<<13) = 1646592, >> 14 = 100 */
    r = npu_postproc_perchannel(&cfg, 200, 1);
    ASSERT_EQ(r, 100, "ch1: 200*8192>>14 = 100");
}

/* ─── Test 2: Per-channel with bias ─── */
static void test_perchannel_bias(void)
{
    printf("Test: per-channel with bias...\n");

    perchannel_param_t params[1] = {
        { .M = 16384, .S = 15, .zp = 0, .bias_q = 50 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = POST_BIAS_EN | PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* (100 + 50) * 16384, + round, >> 15 = 75 */
    int32_t r = npu_postproc_perchannel(&cfg, 100, 0);
    ASSERT_EQ(r, 75, "bias: (100+50)*16384>>15 = 75");
}

/* ─── Test 3: Per-channel with zero point ─── */
static void test_perchannel_zp(void)
{
    printf("Test: per-channel with zero point...\n");

    perchannel_param_t params[1] = {
        { .M = 16384, .S = 15, .zp = 10, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = POST_ZP_EN | PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* 100 * 16384 >> 15 = 50, + zp=10 → 60 */
    int32_t r = npu_postproc_perchannel(&cfg, 100, 0);
    ASSERT_EQ(r, 60, "zp: 100*M>>S + 10 = 60");
}

/* ─── Test 4: Clamp saturation ─── */
static void test_perchannel_clamp(void)
{
    printf("Test: per-channel clamp saturation...\n");

    perchannel_param_t params[1] = {
        { .M = 16384, .S = 10, .zp = 0, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* 1000 * 16384 >> 10 = 16000000 >> 10 ≈ 15625, clamped to 127 */
    int32_t r = npu_postproc_perchannel(&cfg, 1000, 0);
    ASSERT_EQ(r, 127, "clamp overflow → 127");

    /* -1000 * 16384 >> 10 ≈ -15625, clamped to -128 */
    r = npu_postproc_perchannel(&cfg, -1000, 0);
    ASSERT_EQ(r, -128, "clamp underflow → -128");
}

/* ─── Test 5: ReLU activation ─── */
static void test_perchannel_relu(void)
{
    printf("Test: per-channel with ReLU...\n");

    perchannel_param_t params[1] = {
        { .M = 16384, .S = 15, .zp = 0, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = POST_RELU_EN | PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* -100 * 16384 >> 15 = -50, relu → 0 */
    int32_t r = npu_postproc_perchannel(&cfg, -100, 0);
    ASSERT_EQ(r, 0, "relu: negative → 0");

    /* 100 → 50, relu → 50 */
    r = npu_postproc_perchannel(&cfg, 100, 0);
    ASSERT_EQ(r, 50, "relu: positive passes");
}

/* ─── Test 6: Add mode dual rescale ─── */
static void test_add_mode(void)
{
    printf("Test: add mode dual rescale...\n");

    add_param_t add_p = { .M_A = 16384, .S_A = 14, .M_B = 8192, .S_B = 13 };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = PPU_MODE_ADD;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.add_params = &add_p;

    /* A=10: 10*16384 + (1<<13) >> 14 = (163840+8192)>>14 = 172032>>14 = 10 */
    /* B=20: 20*8192 + (1<<12) >> 13 = (163840+4096)>>13 = 167936>>13 = 20 */
    /* sum = 10 + 20 = 30 */
    int32_t r = npu_postproc_add(&cfg, 10, 20);
    ASSERT_EQ(r, 30, "add: rescale(10)+rescale(20) = 30");
}

/* ─── Test 7: Add mode with ReLU ─── */
static void test_add_relu(void)
{
    printf("Test: add mode with ReLU...\n");

    add_param_t add_p = { .M_A = 16384, .S_A = 14, .M_B = 16384, .S_B = 14 };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = PPU_MODE_ADD | POST_RELU_EN;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.add_params = &add_p;

    /* A=-30 → -30, B=10 → 10, sum=-20, relu → 0 */
    int32_t r = npu_postproc_add(&cfg, -30, 10);
    ASSERT_EQ(r, 0, "add+relu: negative sum → 0");

    /* A=30 → 30, B=10 → 10, sum=40 */
    r = npu_postproc_add(&cfg, 30, 10);
    ASSERT_EQ(r, 40, "add+relu: positive sum passes");
}

/* ─── Test 8: INT16 output with 16-bit zp ─── */
static void test_int16_zp(void)
{
    printf("Test: INT16 output with 16-bit zero point...\n");

    perchannel_param_t params[1] = {
        { .M = 16384, .S = 15, .zp = 1000, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = POST_ZP_EN | POST_INT16_OUT | PPU_MODE_CONV_REQ;
    cfg.clamp_min = -32768;
    cfg.clamp_max = 32767;
    cfg.ch_params = params;

    /* 200 * 16384 >> 15 = 100, + zp=1000 → 1100 */
    int32_t r = npu_postproc_perchannel(&cfg, 200, 0);
    ASSERT_EQ(r, 1100, "INT16 zp=1000: 200*M>>S+1000 = 1100");
}

/* ─── Test 9: Edge cases ─── */
static void test_edge_cases(void)
{
    printf("Test: edge cases...\n");

    perchannel_param_t params[1] = {
        { .M = 1, .S = 0, .zp = 0, .bias_q = 0 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.ch_params = params;

    /* M=1, S=0: output = acc * 1 = acc (identity) */
    int32_t r = npu_postproc_perchannel(&cfg, 42, 0);
    ASSERT_EQ(r, 42, "M=1,S=0: identity");

    /* S=63 (max shift) */
    params[0].M = 32767;
    params[0].S = 63;
    r = npu_postproc_perchannel(&cfg, 1000, 0);
    /* 1000 * 32767 = 32767000, + (1<<62) >> 63 ≈ 0 (practically zero) */
    ASSERT_EQ(r, 0, "S=63: large shift → 0");
}

/* ─── Test 10: Full tensor postprocess ─── */
static void test_full_tensor(void)
{
    printf("Test: full tensor postprocess...\n");

    perchannel_param_t params[2] = {
        { .M = 16384, .S = 15, .zp = 0, .bias_q = 10 },
        { .M = 8192,  .S = 14, .zp = 5, .bias_q = -5 },
    };

    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.post_ctrl = POST_BIAS_EN | POST_ZP_EN | PPU_MODE_CONV_REQ;
    cfg.clamp_min = -128;
    cfg.clamp_max = 127;
    cfg.out_h = 1;
    cfg.out_w = 1;
    cfg.out_c = 2;
    cfg.ch_params = params;

    /* Input: [50, 100] as NHWC 1×1×2 */
    int64_t acc[2] = {50, 100};
    tensor_t output = tensor_alloc_i8(1, 1, 2);

    npu_postprocess(&cfg, acc, 1, 1, 2, &output);

    /* ch0: (50+10)*16384 + (1<<14) >> 15 = 60*16384+16384 >> 15 = 998400>>15 = 30 + zp=0→30 */
    /* Actually: (50+10)=60, 60*16384=983040, +16384=999424, >>15=30 */
    ASSERT_EQ(tensor_get_i8(&output, 0, 0, 0), 30, "tensor ch0");

    /* ch1: (100-5)=95, 95*8192=778240, +(1<<13)=786432, >>14=48, +zp=5→53 */
    ASSERT_EQ(tensor_get_i8(&output, 0, 0, 1), 53, "tensor ch1");

    tensor_free(&output);
}

/* ─── Main ─── */
int main(void)
{
    printf("=== Open-NPU Per-Channel Post-Processing Tests ===\n\n");

    test_perchannel_basic();
    test_perchannel_bias();
    test_perchannel_zp();
    test_perchannel_clamp();
    test_perchannel_relu();
    test_add_mode();
    test_add_relu();
    test_int16_zp();
    test_edge_cases();
    test_full_tensor();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
