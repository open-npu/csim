# Open-NPU C Simulator

[中文版](README_CN.md)

Cycle-approximate C simulator for the Open-NPU accelerator. Used as golden reference for RTL verification and model validation.

## Supported Operators

- Conv2D (with per-channel requantization)
- Depthwise Conv2D
- Fully Connected
- Pooling (Max / Average)
- Element-wise (Add / Multiply)
- Concat
- Activation (ReLU, ReLU6, Leaky ReLU)
- Resize (Nearest)
- Deconv (Transposed Convolution)

## Directory Structure

```
include/        Header files
  npu_config.h      Hardware configuration parameters
  npu_types.h       Data types and layer descriptors
  npu_operators.h   Operator function declarations
  npu_postproc.h    Post-processing (requantization)
  npu_dma.h         DMA simulation interface
src/            Operator implementations
test/           Unit tests
testdata/       Test model binaries
```

## Building & Running

```bash
make            # Build simulator
make test       # Run unit tests
./npu_sim <model.bin> <input.bin> <output.bin>
```

Override hardware config at compile time:

```bash
make CFLAGS="-DARRAY_SIZE=4 -DSPAD_KB=64"
```

## Related Repositories

- [open-npu/rtl](https://github.com/open-npu/rtl) — Synthesizable Verilog implementation
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX converter & quantization toolchain
- [open-npu/design](https://github.com/open-npu/design) — Architecture specifications

## License

Apache-2.0
