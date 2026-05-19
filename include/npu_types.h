/*
 * Open-NPU C Functional Simulator
 * npu_types.h — Core data types and layer configuration
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_TYPES_H
#define NPU_TYPES_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Operator type encoding (matches CSR LAYER_MODE.OP_TYPE) ─── */
enum {
    OP_CONV2D      = 0,
    OP_DW_CONV     = 1,
    OP_FC          = 2,
    OP_POOLING     = 3,
    OP_ELTWISE_ADD = 4,
    OP_RESIZE      = 5,
    OP_DECONV      = 6,
    OP_CONCAT      = 7
};

/* ─── Data type encoding ─── */
enum {
    DTYPE_INT8  = 0,
    DTYPE_INT16 = 1
};

/* ─── Post-processing control bits (matches CSR POST_CTRL) ─── */
#define POST_BIAS_EN     (1 << 0)
#define POST_SHIFT_EN    (1 << 1)
#define POST_SCALE_EN    (1 << 2)
#define POST_CLAMP_EN    (1 << 3)
#define POST_LUT_EN      (1 << 4)
#define POST_ELTWISE_EN  (1 << 5)
#define POST_POOL_EN     (1 << 6)
#define POST_OUT_INT16   (1 << 7)

/* ─── Tensor (NHWC layout in memory) ─── */
typedef struct {
    int h, w, c;
    int8_t  *data;      /* INT8 mode */
    int16_t *data_i16;  /* INT16 mode */
} tensor_t;

/* ─── Layer configuration (mirrors CSR register groups) ─── */
typedef struct {
    /* Group 1: Layer mode */
    uint8_t  op_type;       /* 0-7, see OP_xxx */
    uint8_t  data_type;     /* 0=INT8, 1=INT16 */

    /* Dimensions */
    uint16_t in_h, in_w, in_c;
    uint16_t out_h, out_w, out_c;

    /* Convolution kernel */
    uint8_t  kernel_h, kernel_w;
    uint8_t  dilation_h, dilation_w;
    uint8_t  stride_h, stride_w;
    uint8_t  pad_top, pad_bottom, pad_left, pad_right;

    /* Pooling */
    uint8_t  pool_mode;         /* 0=Max, 1=Avg */
    uint8_t  pool_h, pool_w;
    uint8_t  pool_stride_h, pool_stride_w;
    uint8_t  global_pool;       /* 1=global pooling */

    /* Resize */
    uint8_t  resize_mode;       /* 0=nearest, 1=bilinear */
    uint8_t  scale_h, scale_w;  /* Q4.4 fixed-point */

    /* Deconv */
    uint8_t  insert_h, insert_w;

    /* Concat */
    uint16_t concat_offset;     /* channel offset in output */
    uint16_t concat_total_c;    /* total output channels after concat */

    /* Tiling */
    uint16_t tile_h, tile_w;
    uint16_t tile_num_h, tile_num_w;

    /* Post-processing */
    uint8_t  post_ctrl;         /* bitmask of POST_xxx_EN */
    uint8_t  shift_bits;        /* 0-31 */
    uint8_t  round_en;          /* 1=rounding shift */
    int16_t  scale;             /* quantization scale (Q0.15 or multiplier) */
    int8_t   in_zp;             /* input zero point */
    int8_t   weight_zp;         /* weight zero point */
    int8_t   out_zp;            /* output zero point */
    int8_t   clamp_min;
    int8_t   clamp_max;
    uint8_t  bias_shift;        /* bias right-shift */

    /* LUT (256 entries) */
    int8_t   lut_i8[256];
    int16_t  lut_i16[256];
} layer_config_t;

/* ─── Helper: tensor element access (NHWC) ─── */
static inline int8_t tensor_get_i8(const tensor_t *t, int h, int w, int c) {
    return t->data[h * t->w * t->c + w * t->c + c];
}

static inline void tensor_set_i8(tensor_t *t, int h, int w, int c, int8_t val) {
    t->data[h * t->w * t->c + w * t->c + c] = val;
}

static inline int16_t tensor_get_i16(const tensor_t *t, int h, int w, int c) {
    return t->data_i16[h * t->w * t->c + w * t->c + c];
}

static inline void tensor_set_i16(tensor_t *t, int h, int w, int c, int16_t val) {
    t->data_i16[h * t->w * t->c + w * t->c + c] = val;
}

/* ─── Tensor allocation helpers ─── */
static inline tensor_t tensor_alloc_i8(int h, int w, int c) {
    tensor_t t = {h, w, c, NULL, NULL};
    t.data = (int8_t *)calloc((size_t)h * w * c, sizeof(int8_t));
    return t;
}

static inline tensor_t tensor_alloc_i16(int h, int w, int c) {
    tensor_t t = {h, w, c, NULL, NULL};
    t.data_i16 = (int16_t *)calloc((size_t)h * w * c, sizeof(int16_t));
    return t;
}

static inline void tensor_free(tensor_t *t) {
    free(t->data);
    free(t->data_i16);
    t->data = NULL;
    t->data_i16 = NULL;
}

#endif /* NPU_TYPES_H */
