#!/usr/bin/env python3
"""
Generate test data for Add node end-to-end test.

Produces:
  testdata/add_input_a.bin  (NCHW INT8)
  testdata/add_input_b.bin  (NCHW INT8)
  testdata/add_reference.bin (NCHW INT8 output)
  testdata/add_params.txt   (H W C M_A S_A M_B S_B relu)

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import numpy as np

# Add tools dir to path for model_packer reference functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from model_packer import AddParam, ref_postproc_add, PPU_MODE_ADD, POST_RELU_EN

def generate_test_cases(output_dir):
    """Generate multiple Add test cases."""
    os.makedirs(output_dir, exist_ok=True)

    test_cases = [
        # (name, H, W, C, M_A, S_A, M_B, S_B, relu, seed)
        ("basic",     4, 4, 8,  16000, 14, 12000, 13, 1, 999),
        ("no_relu",   4, 4, 8,  16000, 14, 12000, 13, 0, 42),
        ("large",     8, 8, 16, 20000, 15, 10000, 14, 1, 123),
        ("symmetric", 4, 4, 4,  16384, 14, 16384, 14, 0, 7),
        ("edge",      2, 2, 32, 32000, 16, 32000, 16, 1, 555),
    ]

    results = []
    for name, H, W, C, M_A, S_A, M_B, S_B, relu, seed in test_cases:
        np.random.seed(seed)

        # Generate inputs
        input_a = np.random.randint(-50, 50, (H, W, C), dtype=np.int8)
        input_b = np.random.randint(-40, 40, (H, W, C), dtype=np.int8)

        # Compute reference output
        add_params = AddParam(M_A=M_A, S_A=S_A, M_B=M_B, S_B=S_B)

        class Cfg:
            post_ctrl = PPU_MODE_ADD | (POST_RELU_EN if relu else 0)
            clamp_min = -128
            clamp_max = 127

        ref_out = ref_postproc_add(input_a, input_b, add_params, Cfg())

        # Save as NCHW
        a_nchw = input_a.transpose(2, 0, 1)  # [C][H][W]
        b_nchw = input_b.transpose(2, 0, 1)
        out_nchw = ref_out.transpose(2, 0, 1)

        prefix = os.path.join(output_dir, f"add_{name}")
        a_nchw.tofile(f"{prefix}_a.bin")
        b_nchw.tofile(f"{prefix}_b.bin")
        out_nchw.tofile(f"{prefix}_ref.bin")

        # Save params
        with open(f"{prefix}_params.txt", 'w') as f:
            f.write(f"{H} {W} {C} {M_A} {S_A} {M_B} {S_B} {relu}\n")

        results.append((name, H, W, C, M_A, S_A, M_B, S_B, relu))
        print(f"  {name}: input=[{H}x{W}x{C}], M_A={M_A}, S_A={S_A}, "
              f"M_B={M_B}, S_B={S_B}, relu={relu}")

    # Write test list for runner script
    with open(os.path.join(output_dir, "add_tests.txt"), 'w') as f:
        for name, H, W, C, M_A, S_A, M_B, S_B, relu in results:
            f.write(f"{name} {H} {W} {C} {M_A} {S_A} {M_B} {S_B} {relu}\n")

    return results


if __name__ == '__main__':
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'testdata')
    print("=== Generating Add node E2E test data ===")
    cases = generate_test_cases(test_dir)
    print(f"\nGenerated {len(cases)} test cases in: {os.path.abspath(test_dir)}")
