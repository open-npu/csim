/*
 * Open-NPU C Functional Simulator — tiled Resize regression test
 *
 * Proves that a TILED Resize produces bit-identical output to
 *   (a) the same layer run UNTILED, and
 *   (b) an independent textbook global resize reference implemented here.
 *
 * Background: the tiled path used to derive the input-space tile origin from
 * `tile_idx * tile * stride` (stride==1 for Resize), i.e. it fed an
 * OUTPUT-space origin into an INPUT-space read, and then asked resize.c to map
 * output→input using the TILE-local in/out ratio. Only tile (0,0) was correct;
 * later tiles read the wrong rows and eventually ran off the end of the input.
 *
 * Includes src/main.c directly so the *real* execute_layer_tiled() is tested.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define main csim_main_unused
#include "../src/main.c"
#undef main

/* ─── Independent textbook reference (global coords, no tiling concept) ─── */

static void ref_resize(const layer_config_t *cfg,
                       const tensor_t *in, tensor_t *out)
{
    const int in_h = cfg->in_h, in_w = cfg->in_w;
    const int out_h = cfg->out_h, out_w = cfg->out_w;
    const int ch = cfg->in_c;
    const int i16 = (cfg->data_type == DTYPE_INT16);
    const int o16 = (cfg->post_ctrl & POST_INT16_OUT) != 0;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < ch; c++) {
                int32_t val;
                if (cfg->resize_mode == 0) {
                    int ih = (oh * in_h) / out_h; if (ih >= in_h) ih = in_h - 1;
                    int iw = (ow * in_w) / out_w; if (iw >= in_w) iw = in_w - 1;
                    val = i16 ? tensor_get_i16(in, ih, iw, c)
                              : tensor_get_i8(in, ih, iw, c);
                } else {
                    int32_t sh_q8 = (out_h > 1)
                        ? (int32_t)oh * ((in_h - 1) << 8) / (out_h - 1) : 0;
                    int32_t sw_q8 = (out_w > 1)
                        ? (int32_t)ow * ((in_w - 1) << 8) / (out_w - 1) : 0;
                    int ih0 = sh_q8 >> 8, ih1 = ih0 + 1;
                    int iw0 = sw_q8 >> 8, iw1 = iw0 + 1;
                    if (ih1 >= in_h) ih1 = in_h - 1;
                    if (iw1 >= in_w) iw1 = in_w - 1;
                    int fh = sh_q8 & 0xFF, fw = sw_q8 & 0xFF;
                    int32_t v00, v01, v10, v11;
                    if (i16) {
                        v00 = tensor_get_i16(in, ih0, iw0, c);
                        v01 = tensor_get_i16(in, ih0, iw1, c);
                        v10 = tensor_get_i16(in, ih1, iw0, c);
                        v11 = tensor_get_i16(in, ih1, iw1, c);
                    } else {
                        v00 = tensor_get_i8(in, ih0, iw0, c);
                        v01 = tensor_get_i8(in, ih0, iw1, c);
                        v10 = tensor_get_i8(in, ih1, iw0, c);
                        v11 = tensor_get_i8(in, ih1, iw1, c);
                    }
                    int32_t top = v00 * (256 - fw) + v01 * fw;
                    int32_t bot = v10 * (256 - fw) + v11 * fw;
                    val = (top * (256 - fh) + bot * fh + (1 << 15)) >> 16;
                }
                if (o16) tensor_set_i16(out, oh, ow, c, (int16_t)val);
                else     tensor_set_i8(out, oh, ow, c, (int8_t)val);
            }
        }
    }
}

/* ─── Helpers ─── */

static tensor_t alloc_for(const layer_config_t *cfg, int h, int w, int c, int is_out)
{
    int wide = is_out ? ((cfg->post_ctrl & POST_INT16_OUT) != 0)
                      : (cfg->data_type == DTYPE_INT16);
    return wide ? tensor_alloc_i16(h, w, c) : tensor_alloc_i8(h, w, c);
}

static void fill_random(tensor_t *t, int is_int16, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < t->h * t->w * t->c; i++) {
        if (is_int16) t->data_i16[i] = (int16_t)((rand() % 2001) - 1000);
        else          t->data[i]     = (int8_t)((rand() % 255) - 127);
    }
}

/* Returns number of mismatching elements between a and b */
static int diff_count(const tensor_t *a, const tensor_t *b, int wide)
{
    int n = a->h * a->w * a->c, bad = 0;
    for (int i = 0; i < n; i++) {
        if (wide) { if (a->data_i16[i] != b->data_i16[i]) bad++; }
        else      { if (a->data[i]     != b->data[i])     bad++; }
    }
    return bad;
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* ─── One test case ─── */

static int fails = 0;

static tensor_t g_last_tiled;   /* kept for the real-data case */

static void run_case(const char *name,
                     int in_h, int in_w, int ch,
                     int out_h, int out_w,
                     int tile_h, int tile_w,
                     int resize_mode, int int16,
                     const tensor_t *preset_in)
{
    layer_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.op_type     = OP_RESIZE;
    cfg.data_type   = int16 ? DTYPE_INT16 : DTYPE_INT8;
    cfg.in_h = in_h; cfg.in_w = in_w; cfg.in_c = ch;
    cfg.out_h = out_h; cfg.out_w = out_w; cfg.out_c = ch;
    cfg.kernel_h = cfg.kernel_w = 1;
    cfg.stride_h = cfg.stride_w = 1;
    cfg.dilation_h = cfg.dilation_w = 1;
    cfg.resize_mode = (uint8_t)resize_mode;
    cfg.post_ctrl   = PPU_MODE_PASSTHROUGH | (int16 ? POST_INT16_OUT : 0);
    cfg.clamp_min   = int16 ? -32768 : -128;
    cfg.clamp_max   = int16 ?  32767 :  127;
    cfg.residual_src = -1;
    cfg.input_src    = -1;

    tensor_t in;
    if (preset_in) {
        in = *preset_in;
    } else {
        in = alloc_for(&cfg, in_h, in_w, ch, 0);
        fill_random(&in, int16, (unsigned)(in_h * 7919 + in_w * 104729 + out_h * 31 + out_w));
    }

    /* Reference */
    tensor_t ref = alloc_for(&cfg, out_h, out_w, ch, 1);
    ref_resize(&cfg, &in, &ref);

    /* Untiled through the real CSIM path */
    layer_config_t cfg_untiled = cfg;
    cfg_untiled.tile_h = cfg_untiled.tile_w = 0;
    cfg_untiled.tile_num_h = cfg_untiled.tile_num_w = 0;
    tensor_t out_untiled = alloc_for(&cfg, out_h, out_w, ch, 1);
    execute_layer_tiled(&cfg_untiled, &in, NULL, NULL, &out_untiled);

    /* Tiled through the real CSIM path */
    layer_config_t cfg_tiled = cfg;
    cfg_tiled.tile_h = (uint16_t)tile_h;
    cfg_tiled.tile_w = (uint16_t)tile_w;
    cfg_tiled.tile_num_h = (uint16_t)ceil_div(out_h, tile_h);
    cfg_tiled.tile_num_w = (uint16_t)ceil_div(out_w, tile_w);
    tensor_t out_tiled = alloc_for(&cfg, out_h, out_w, ch, 1);
    execute_layer_tiled(&cfg_tiled, &in, NULL, NULL, &out_tiled);

    int wide = (cfg.post_ctrl & POST_INT16_OUT) != 0;
    int d_ut_ref = diff_count(&out_untiled, &ref, wide);
    int d_t_ut   = diff_count(&out_tiled, &out_untiled, wide);
    int d_t_ref  = diff_count(&out_tiled, &ref, wide);
    int total    = out_h * out_w * ch;

    int ok = (d_ut_ref == 0 && d_t_ut == 0 && d_t_ref == 0);
    if (!ok) fails++;

    printf("%-4s %-30s %2dx%-2d ->%3dx%-3d c=%-3d tile %2dx%-2d (%dx%d) %-8s %-5s "
           "| untiled^ref %d/%d  tiled^untiled %d/%d\n",
           ok ? "PASS" : "FAIL", name,
           in_h, in_w, out_h, out_w, ch, tile_h, tile_w,
           cfg_tiled.tile_num_h, cfg_tiled.tile_num_w,
           resize_mode ? "bilinear" : "nearest", int16 ? "int16" : "int8",
           d_ut_ref, total, d_t_ut, total);

    /* Hand the tiled result to the caller for the real-data case */
    tensor_free(&g_last_tiled);
    g_last_tiled = out_tiled;

    tensor_free(&ref);
    tensor_free(&out_untiled);
    if (!preset_in) tensor_free(&in);
}

/* ─── main ─── */

int main(int argc, char *argv[])
{
    memset(&g_last_tiled, 0, sizeof(g_last_tiled));

    printf("=== Tiled Resize == Untiled Resize == textbook global reference ===\n\n");

    /* The real failing case: model_c_int8 layer 12.
     * If a raw int8 NHWC input file is supplied, use the true layer-11 output. */
    tensor_t real_in;
    memset(&real_in, 0, sizeof(real_in));
    const tensor_t *preset = NULL;
    if (argc >= 3) {
        real_in = tensor_alloc_i8(13, 13, 64);
        FILE *f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
        size_t n = 13 * 13 * 64;
        if (fread(real_in.data, 1, n, f) != n) {
            fprintf(stderr, "short read on %s\n", argv[1]); fclose(f); return 2;
        }
        fclose(f);
        preset = &real_in;
        printf("(using real model_c_int8 layer-11 output as input)\n");
    }
    run_case("model_c_int8 L12 (real cfg)", 13, 13, 64, 26, 26, 8, 12, 0, 0, preset);

    if (preset && argc >= 3) {
        FILE *f = fopen(argv[2], "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 2; }
        fwrite(g_last_tiled.data, 1, 26 * 26 * 64, f);
        fclose(f);
        printf("(tiled result written to %s)\n", argv[2]);
    }
    printf("\n");

    /* Integer 2x, tiles divide evenly */
    run_case("2x exact tiles",           8,  8,  4, 16, 16,  8,  8, 0, 0, NULL);
    /* Integer 2x, border tiles in both dims */
    run_case("2x border tiles",          8,  8,  4, 16, 16,  5,  5, 0, 0, NULL);
    /* Different H and W ratios */
    run_case("H 2x / W 3x",              6,  4,  8, 12, 12,  5,  5, 0, 0, NULL);
    run_case("H 3x / W 2x",              4,  6,  8, 12, 12,  5,  7, 0, 0, NULL);
    /* Non-integer ratios (the hard case for any scale-based shortcut) */
    run_case("non-integer 5x7->13x11",   5,  7,  3, 13, 11,  4,  4, 0, 0, NULL);
    run_case("non-integer 13x13->26x26", 13, 13, 3, 26, 26,  8, 12, 0, 0, NULL);
    run_case("non-integer 7x9->17x23",   7,  9,  5, 17, 23,  6,  5, 0, 0, NULL);
    /* Downsample */
    run_case("downsample 16x16->6x6",   16, 16,  4,  6,  6,  4,  4, 0, 0, NULL);
    run_case("downsample 13x11->5x7",   13, 11,  4,  5,  7,  2,  3, 0, 0, NULL);
    /* Degenerate tiles */
    run_case("single tile == output",     9,  9,  2, 18, 18, 18, 18, 0, 0, NULL);
    run_case("tile 1x1 (per-pixel)",      3,  3,  2,  7,  7,  1,  1, 0, 0, NULL);
    run_case("tile_h=1 strips",           6,  6,  2, 12, 12,  1, 12, 0, 0, NULL);
    run_case("out == in (1:1)",           7,  7,  3,  7,  7,  3,  3, 0, 0, NULL);
    /* INT16 */
    run_case("int16 2x",                  8,  8,  4, 16, 16,  5,  5, 0, 1, NULL);
    run_case("int16 non-integer",        13, 13,  8, 26, 26,  8, 12, 0, 1, NULL);
    /* Bilinear */
    run_case("bilinear 2x",               8,  8,  4, 16, 16,  5,  5, 1, 0, NULL);
    run_case("bilinear non-integer",      7,  5,  3, 12,  9,  4,  3, 1, 0, NULL);
    run_case("bilinear downsample",      16, 16,  3,  6,  6,  4,  4, 1, 0, NULL);
    run_case("bilinear int16",           13, 13,  4, 26, 26,  8, 12, 1, 1, NULL);

    tensor_free(&g_last_tiled);
    if (preset) tensor_free(&real_in);

    printf("\n%s\n", fails == 0 ? "ALL RESIZE TILING TESTS PASSED"
                                : "SOME RESIZE TILING TESTS FAILED");
    return fails == 0 ? 0 : 1;
}
