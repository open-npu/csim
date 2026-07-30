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

/* ─── POST_CTRL bits (matches CSR 0x180 POST_CTRL) ─── */
#define POST_PPU_MODE_MASK   0x03   /* bit[1:0] */
#define PPU_MODE_CONV_REQ    0      /* requantize after conv/dw/fc */
#define PPU_MODE_ADD         1      /* eltwise add with dual rescale */
#define PPU_MODE_RELU_ONLY   2      /* only activation, no requant */
#define PPU_MODE_PASSTHROUGH 3      /* bypass PPU */

#define POST_RELU_EN     (1 << 2)
#define POST_RELU6_EN    (1 << 3)
#define POST_LUT_EN      (1 << 4)
#define POST_ZP_EN       (1 << 5)
#define POST_BIAS_EN     (1 << 6)
#define POST_INT16_OUT   (1 << 7)

/* ─── SCHED_CTRL bits (model descriptor scheduling, NOT HW DMA_CTRL CSR) ─── */
#define SCHED_CTRL_DB_EN       (1 << 0)   /* bit[0]: double-buffer enable (ping-pong) */
#define SCHED_CTRL_FUSE_START  (1 << 1)   /* bit[1]: first layer of fused block */
#define SCHED_CTRL_FUSE_MID    (1 << 2)   /* bit[2]: middle layer of fused block */
#define SCHED_CTRL_FUSE_END    (1 << 3)   /* bit[3]: last layer of fused block */
#define SCHED_CTRL_PER_TILE_STORE (1 << 4) /* bit[4]: per-tile store to NHWC DDR (cascaded inference) */

/* ─── Per-channel requantize parameters (14 bytes/channel, packed) ─── */
typedef struct {
    uint16_t M;          /* [14:0] 15-bit unsigned multiplier, bit15=0 */
    uint8_t  S;          /* [5:0]  6-bit shift amount */
    uint8_t  _reserved0;
    int16_t  zp;         /* [15:0] 16-bit signed output zero point */
    int64_t  bias_q;     /* [63:0] 64-bit signed quantized bias */
} __attribute__((packed)) perchannel_param_t;

/* ─── Add node rescale parameters (8 bytes total) ─── */
typedef struct {
    uint16_t M_A;        /* [14:0] branch A multiplier */
    uint8_t  S_A;        /* [5:0]  branch A shift */
    uint8_t  _reserved0;
    uint16_t M_B;        /* [14:0] branch B multiplier */
    uint8_t  S_B;        /* [5:0]  branch B shift */
    uint8_t  _reserved1;
} __attribute__((packed)) add_param_t;

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

    /* Resize tiling context (runtime-only, not serialized).
     * Resize coordinates are GLOBAL: source pixel for output row oh_global is
     * (oh_global * full_in_h) / full_out_h. When a Resize layer is tiled, the
     * per-tile config carries in_h/in_w/out_h/out_w of the *tile*, so these
     * fields preserve the full-tensor dims plus the tile origins needed to
     * convert global coords into tile-local buffer indices.
     * rsz_tiled == 0 (calloc default) → non-tiled, resize.c uses cfg->in_h etc.
     * Mirrors RTL npu_compute.v tile_oh_origin / rsz_tile_ih_origin. */
    uint8_t  rsz_tiled;
    uint16_t rsz_full_in_h,  rsz_full_in_w;
    uint16_t rsz_full_out_h, rsz_full_out_w;
    uint16_t rsz_tile_oh_origin, rsz_tile_ow_origin;  /* output-space tile origin */
    uint16_t rsz_tile_ih_origin, rsz_tile_iw_origin;  /* input-space tile origin */

    /* Deconv */
    uint8_t  insert_h, insert_w;

    /* Concat */
    uint16_t concat_offset;     /* channel offset in output */
    uint16_t concat_total_c;    /* total output channels after concat */

    /* Tiling */
    uint16_t tile_h, tile_w;
    uint16_t tile_num_h, tile_num_w;

    /* Post-processing (per-channel requantize architecture) */
    uint8_t  post_ctrl;         /* PPU_MODE[1:0] + enable bits */
    uint8_t  sched_ctrl;        /* SCHED_CTRL: bit[0]=DB_EN (double-buffer) */
    int16_t  clamp_min;         /* 16-bit signed min (INT8: -128, INT16: -32768) */
    int16_t  clamp_max;         /* 16-bit signed max (INT8: 127, INT16: 32767) */
    int8_t   in_zp;             /* input zero point (for padding fill value) */

    /* Per-channel params (pointer, not serialized in fixed config) */
    perchannel_param_t *ch_params;   /* [out_c], NULL if PASSTHROUGH */

    /* Add node params (only valid when PPU_MODE_ADD) */
    add_param_t *add_params;         /* single instance or NULL */

    /* Residual source layer index for Add (-1 = none) */
    int8_t   residual_src;

    /* Input source layer index (-1 = previous layer) */
    int16_t  input_src;

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
