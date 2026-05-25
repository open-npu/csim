# Open-NPU C 模拟器

[English](README.md)

Open-NPU 加速器的周期近似 C 模拟器，用作 RTL 验证的 golden reference 和模型验证。

## 支持的算子

- Conv2D（per-channel 重量化）
- Depthwise Conv2D
- 全连接 (FC)
- 池化（Max / Average）
- 逐元素运算（Add / Multiply）
- Concat
- 激活函数（ReLU、ReLU6、Leaky ReLU）
- Resize（最近邻）
- 反卷积（Transposed Convolution）

## 目录结构

```
include/        头文件
  npu_config.h      硬件配置参数
  npu_types.h       数据类型与层描述符
  npu_operators.h   算子函数声明
  npu_postproc.h    后处理（重量化）
  npu_dma.h         DMA 模拟接口
src/            算子实现
test/           单元测试
testdata/       测试模型二进制文件
```

## 编译与运行

```bash
make            # 编译模拟器
make test       # 运行单元测试
./npu_sim <model.bin> <input.bin> <output.bin>
```

编译时覆盖硬件配置：

```bash
make CFLAGS="-DARRAY_SIZE=4 -DSPAD_KB=64"
```

## 相关仓库

- [open-npu/rtl](https://github.com/open-npu/rtl) — 可综合 Verilog 实现
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX 转换器与量化工具链
- [open-npu/design](https://github.com/open-npu/design) — 架构设计文档

## 许可证

Apache-2.0
