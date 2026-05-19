/*
 * Open-NPU C Functional Simulator
 * fc.c — Fully connected layer (bit-exact)
 *
 * Weight layout: [out_c][in_c]
 * Input layout:  NHWC with h=1, w=1, c=in_c
 *
 * Supports: INT8/INT16
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_operators.h"

void npu_fc(const layer_config_t *cfg,
            const tensor_t *input,
            const int8_t *weights,
            const int32_t *bias,
            int32_t *output_acc)
{
    const int out_c = cfg->out_c;
    const int in_c  = cfg->in_c;
    const int is_int16 = (cfg->data_type == DTYPE_INT16);
    const int16_t *weights_i16 = (const int16_t *)weights;

    for (int oc = 0; oc < out_c; oc++) {
        int32_t acc = 0;

        for (int ic = 0; ic < in_c; ic++) {
            if (is_int16) {
                int16_t in_val = input->data_i16[ic];
                int16_t w_val  = weights_i16[oc * in_c + ic];
                acc += (int32_t)in_val * (int32_t)w_val;
            } else {
                int8_t in_val = input->data[ic];
                int8_t w_val  = weights[oc * in_c + ic];
                acc += (int32_t)in_val * (int32_t)w_val;
            }
        }

        output_acc[oc] = acc;
    }

    (void)bias;
}
