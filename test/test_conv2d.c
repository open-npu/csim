/*
 * Open-NPU C Functional Simulator
 * test_conv2d.c — Unit tests for Conv2D and DW Conv
 *
 * Hand-computed test cases with small tensors.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "npu_types.h"
#include "npu_operators.h"
#include "npu_postproc.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("  FAIL: %s (expected %ld, got %ld)\n", msg, (long)(b), (long)(a)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ─── Test 1: Conv2D 3×3, 1 input channel, 1 output channel ─── */
static void test_conv2d_3x3_basic(void)
{
    printf("Test: conv2d 3x3 basic...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_CONV2D;
    cfg.in_h = 4; cfg.in_w = 4; cfg.in_c = 1;
    cfg.out_h = 2; cfg.out_w = 2; cfg.out_c = 1;
    cfg.kernel_h = 3; cfg.kernel_w = 3;
    cfg.dilation_h = 1; cfg.dilation_w = 1;
    cfg.stride_h = 1; cfg.stride_w = 1;
    cfg.pad_top = 0; cfg.pad_bottom = 0;
    cfg.pad_left = 0; cfg.pad_right = 0;

    /* Input 4×4×1 (NHWC): values 1-16 */
    tensor_t input = tensor_alloc_i8(4, 4, 1);
    for (int i = 0; i < 16; i++) input.data[i] = (int8_t)(i + 1);

    /* Weight 3×3×1 (all ones) → [oc=1][kh=3][kw=3][ic=1] */
    int8_t weights[9];
    for (int i = 0; i < 9; i++) weights[i] = 1;

    /* Expected output (sum of 3×3 window):
     * pos(0,0): 1+2+3+5+6+7+9+10+11 = 54
     * pos(0,1): 2+3+4+6+7+8+10+11+12 = 63
     * pos(1,0): 5+6+7+9+10+11+13+14+15 = 90
     * pos(1,1): 6+7+8+10+11+12+14+15+16 = 99
     */
    int64_t acc[4] = {0};
    npu_conv2d(&cfg, &input, weights, NULL, acc);

    ASSERT_EQ(acc[0], 54, "conv2d[0,0]");
    ASSERT_EQ(acc[1], 63, "conv2d[0,1]");
    ASSERT_EQ(acc[2], 90, "conv2d[1,0]");
    ASSERT_EQ(acc[3], 99, "conv2d[1,1]");

    tensor_free(&input);
}

/* ─── Test 2: Conv2D with padding ─── */
static void test_conv2d_3x3_pad1(void)
{
    printf("Test: conv2d 3x3 with pad=1...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_CONV2D;
    cfg.in_h = 3; cfg.in_w = 3; cfg.in_c = 1;
    cfg.out_h = 3; cfg.out_w = 3; cfg.out_c = 1;
    cfg.kernel_h = 3; cfg.kernel_w = 3;
    cfg.dilation_h = 1; cfg.dilation_w = 1;
    cfg.stride_h = 1; cfg.stride_w = 1;
    cfg.pad_top = 1; cfg.pad_bottom = 1;
    cfg.pad_left = 1; cfg.pad_right = 1;

    /* Input 3×3: all ones */
    tensor_t input = tensor_alloc_i8(3, 3, 1);
    for (int i = 0; i < 9; i++) input.data[i] = 1;

    /* Weight: all ones */
    int8_t weights[9];
    for (int i = 0; i < 9; i++) weights[i] = 1;

    /* With pad=1, center pixel sees all 9 neighbors = 9
     * Corner pixels see 4, edge pixels see 6 */
    int64_t acc[9] = {0};
    npu_conv2d(&cfg, &input, weights, NULL, acc);

    ASSERT_EQ(acc[0], 4, "conv2d_pad[0,0] corner");
    ASSERT_EQ(acc[1], 6, "conv2d_pad[0,1] edge");
    ASSERT_EQ(acc[4], 9, "conv2d_pad[1,1] center");

    tensor_free(&input);
}

/* ─── Test 3: Depthwise Conv 3×3, 2 channels ─── */
static void test_dwconv_3x3(void)
{
    printf("Test: dwconv 3x3, 2 channels...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_DW_CONV;
    cfg.in_h = 3; cfg.in_w = 3; cfg.in_c = 2;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 2;
    cfg.kernel_h = 3; cfg.kernel_w = 3;
    cfg.dilation_h = 1; cfg.dilation_w = 1;
    cfg.stride_h = 1; cfg.stride_w = 1;
    cfg.pad_top = 0; cfg.pad_bottom = 0;
    cfg.pad_left = 0; cfg.pad_right = 0;

    /* Input 3×3×2 (NHWC) */
    tensor_t input = tensor_alloc_i8(3, 3, 2);
    /* Channel 0: all 1s, Channel 1: all 2s */
    for (int h = 0; h < 3; h++)
        for (int w = 0; w < 3; w++) {
            tensor_set_i8(&input, h, w, 0, 1);
            tensor_set_i8(&input, h, w, 1, 2);
        }

    /* Weights [2][3][3]: ch0=all 1, ch1=all 1 */
    int8_t weights[18];
    for (int i = 0; i < 18; i++) weights[i] = 1;

    int64_t acc[2] = {0};
    npu_dwconv(&cfg, &input, weights, NULL, acc);

    /* ch0: 9 × 1 = 9, ch1: 9 × 2 = 18 */
    ASSERT_EQ(acc[0], 9,  "dwconv ch0");
    ASSERT_EQ(acc[1], 18, "dwconv ch1");

    tensor_free(&input);
}

/* ─── Test 4: FC ─── */
static void test_fc_basic(void)
{
    printf("Test: FC basic...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_FC;
    cfg.in_h = 1; cfg.in_w = 1; cfg.in_c = 4;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 2;

    /* Input: [1, 2, 3, 4] */
    tensor_t input = tensor_alloc_i8(1, 1, 4);
    input.data[0] = 1; input.data[1] = 2;
    input.data[2] = 3; input.data[3] = 4;

    /* Weights [2][4]: row0=[1,1,1,1], row1=[2,2,2,2] */
    int8_t weights[8] = {1,1,1,1, 2,2,2,2};

    int64_t acc[2] = {0};
    npu_fc(&cfg, &input, weights, NULL, acc);

    /* oc0: 1+2+3+4=10, oc1: 2+4+6+8=20 */
    ASSERT_EQ(acc[0], 10, "fc oc0");
    ASSERT_EQ(acc[1], 20, "fc oc1");

    tensor_free(&input);
}

/* ─── Test 5: Pooling ─── */
static void test_pooling_max(void)
{
    printf("Test: MaxPool 2x2...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_POOLING;
    cfg.in_h = 4; cfg.in_w = 4; cfg.in_c = 1;
    cfg.out_h = 2; cfg.out_w = 2; cfg.out_c = 1;
    cfg.pool_mode = 0; /* Max */
    cfg.pool_h = 2; cfg.pool_w = 2;
    cfg.pool_stride_h = 2; cfg.pool_stride_w = 2;

    tensor_t input = tensor_alloc_i8(4, 4, 1);
    for (int i = 0; i < 16; i++) input.data[i] = (int8_t)(i + 1);

    int64_t acc[4] = {0};
    npu_pooling(&cfg, &input, acc);

    /* Max of each 2×2 window:
     * [0,0]: max(1,2,5,6) = 6
     * [0,1]: max(3,4,7,8) = 8
     * [1,0]: max(9,10,13,14) = 14
     * [1,1]: max(11,12,15,16) = 16
     */
    ASSERT_EQ(acc[0], 6,  "maxpool[0,0]");
    ASSERT_EQ(acc[1], 8,  "maxpool[0,1]");
    ASSERT_EQ(acc[2], 14, "maxpool[1,0]");
    ASSERT_EQ(acc[3], 16, "maxpool[1,1]");

    tensor_free(&input);
}

static void test_pooling_avg_recip_edges(void)
{
    printf("Test: AvgPool reciprocal edge counts...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_POOLING;
    cfg.pool_mode = 1; /* Average */
    cfg.in_c = cfg.out_c = 1;

    /* count=1 must be identity, not sign inversion. */
    cfg.in_h = cfg.in_w = cfg.out_h = cfg.out_w = 1;
    cfg.pool_h = cfg.pool_w = cfg.pool_stride_h = cfg.pool_stride_w = 1;
    tensor_t one = tensor_alloc_i8(1, 1, 1);
    one.data[0] = 5;
    int64_t acc_one[1] = {0};
    npu_pooling(&cfg, &one, acc_one);
    ASSERT_EQ(acc_one[0], 5, "avgpool count=1");
    tensor_free(&one);

    /* count=2 uses the RTL's positive 0x40000000 multiplier and >>31. */
    cfg.in_h = 1; cfg.in_w = 2;
    cfg.out_h = cfg.out_w = 1;
    cfg.pool_h = 1; cfg.pool_w = 2;
    cfg.pool_stride_h = 1; cfg.pool_stride_w = 2;
    tensor_t two = tensor_alloc_i8(1, 2, 1);
    two.data[0] = 2;
    two.data[1] = 4;
    int64_t acc_two[1] = {0};
    npu_pooling(&cfg, &two, acc_two);
    ASSERT_EQ(acc_two[0], 3, "avgpool count=2");
    tensor_free(&two);

    /* 8x8 global average exercises reciprocal index 64. */
    cfg.in_h = cfg.in_w = 8;
    cfg.out_h = cfg.out_w = 1;
    cfg.global_pool = 1;
    tensor_t sixty_four = tensor_alloc_i8(8, 8, 1);
    memset(sixty_four.data, 64, 64);
    int64_t acc_sixty_four[1] = {0};
    npu_pooling(&cfg, &sixty_four, acc_sixty_four);
    ASSERT_EQ(acc_sixty_four[0], 64, "avgpool count=64");
    tensor_free(&sixty_four);
}

/* ─── Test 6: Conv2D INT16 ─── */
static void test_conv2d_int16(void)
{
    printf("Test: conv2d 3x3 INT16...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_CONV2D;
    cfg.data_type = DTYPE_INT16;
    cfg.in_h = 3; cfg.in_w = 3; cfg.in_c = 1;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 1;
    cfg.kernel_h = 3; cfg.kernel_w = 3;
    cfg.dilation_h = 1; cfg.dilation_w = 1;
    cfg.stride_h = 1; cfg.stride_w = 1;
    cfg.pad_top = 0; cfg.pad_bottom = 0;
    cfg.pad_left = 0; cfg.pad_right = 0;

    /* Input 3×3×1 INT16: values 100-108 (exceed INT8 range) */
    tensor_t input = tensor_alloc_i16(3, 3, 1);
    for (int i = 0; i < 9; i++) input.data_i16[i] = (int16_t)(100 + i);

    /* Weight 3×3 INT16: all 2 */
    int16_t weights[9];
    for (int i = 0; i < 9; i++) weights[i] = 2;

    /* Expected: sum(2 * (100..108)) = 2 * (100+101+...+108) = 2 * 936 = 1872 */
    int64_t acc[1] = {0};
    npu_conv2d(&cfg, &input, (const int8_t *)weights, NULL, acc);

    ASSERT_EQ(acc[0], 1872, "conv2d INT16");

    tensor_free(&input);
}

/* ─── Test 7: DWConv INT16 ─── */
static void test_dwconv_int16(void)
{
    printf("Test: dwconv 3x3 INT16...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_DW_CONV;
    cfg.data_type = DTYPE_INT16;
    cfg.in_h = 3; cfg.in_w = 3; cfg.in_c = 2;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 2;
    cfg.kernel_h = 3; cfg.kernel_w = 3;
    cfg.dilation_h = 1; cfg.dilation_w = 1;
    cfg.stride_h = 1; cfg.stride_w = 1;

    /* Input: ch0 = 200, ch1 = -300 (values outside INT8 range) */
    tensor_t input = tensor_alloc_i16(3, 3, 2);
    for (int h = 0; h < 3; h++)
        for (int w = 0; w < 3; w++) {
            tensor_set_i16(&input, h, w, 0, 200);
            tensor_set_i16(&input, h, w, 1, -300);
        }

    /* Weights: ch0 all=3, ch1 all=2 */
    int16_t weights[18];
    for (int i = 0; i < 9; i++) weights[i] = 3;
    for (int i = 9; i < 18; i++) weights[i] = 2;

    int64_t acc[2] = {0};
    npu_dwconv(&cfg, &input, (const int8_t *)weights, NULL, acc);

    /* ch0: 9 * 200 * 3 = 5400, ch1: 9 * (-300) * 2 = -5400 */
    ASSERT_EQ(acc[0], 5400,  "dwconv INT16 ch0");
    ASSERT_EQ(acc[1], -5400, "dwconv INT16 ch1");

    tensor_free(&input);
}

/* ─── Test 8: FC INT16 ─── */
static void test_fc_int16(void)
{
    printf("Test: FC INT16...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_FC;
    cfg.data_type = DTYPE_INT16;
    cfg.in_h = 1; cfg.in_w = 1; cfg.in_c = 3;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 2;

    /* Input: [1000, -500, 300] (outside INT8) */
    tensor_t input = tensor_alloc_i16(1, 1, 3);
    input.data_i16[0] = 1000;
    input.data_i16[1] = -500;
    input.data_i16[2] = 300;

    /* Weights [2][3]: row0=[1,2,3], row1=[-1,1,-1] */
    int16_t weights[6] = {1, 2, 3, -1, 1, -1};

    int64_t acc[2] = {0};
    npu_fc(&cfg, &input, (const int8_t *)weights, NULL, acc);

    /* oc0: 1000*1 + (-500)*2 + 300*3 = 1000 - 1000 + 900 = 900 */
    /* oc1: 1000*(-1) + (-500)*1 + 300*(-1) = -1000 - 500 - 300 = -1800 */
    ASSERT_EQ(acc[0], 900,   "fc INT16 oc0");
    ASSERT_EQ(acc[1], -1800, "fc INT16 oc1");

    tensor_free(&input);
}

/* ─── Test 9: Pooling INT16 ─── */
static void test_pooling_int16(void)
{
    printf("Test: MaxPool 2x2 INT16...\n");

    layer_config_t cfg = {0};
    cfg.op_type = OP_POOLING;
    cfg.data_type = DTYPE_INT16;
    cfg.in_h = 2; cfg.in_w = 2; cfg.in_c = 1;
    cfg.out_h = 1; cfg.out_w = 1; cfg.out_c = 1;
    cfg.pool_mode = 0; /* Max */
    cfg.pool_h = 2; cfg.pool_w = 2;
    cfg.pool_stride_h = 2; cfg.pool_stride_w = 2;

    tensor_t input = tensor_alloc_i16(2, 2, 1);
    input.data_i16[0] = -500;
    input.data_i16[1] = 1000;
    input.data_i16[2] = 200;
    input.data_i16[3] = -100;

    int64_t acc[1] = {0};
    npu_pooling(&cfg, &input, acc);

    ASSERT_EQ(acc[0], 1000, "maxpool INT16");

    tensor_free(&input);
}

/* ─── Main ─── */
int main(void)
{
    printf("=== Open-NPU Operator Tests ===\n\n");

    test_conv2d_3x3_basic();
    test_conv2d_3x3_pad1();
    test_dwconv_3x3();
    test_fc_basic();
    test_pooling_max();
    test_pooling_avg_recip_edges();
    test_conv2d_int16();
    test_dwconv_int16();
    test_fc_int16();
    test_pooling_int16();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
