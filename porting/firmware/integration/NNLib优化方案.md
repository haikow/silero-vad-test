# Silero VAD 的 HiFi5 NNLib 优化方案(GCC 软浮点 → SIMD 内核)

> 2026-09-29 定稿。动因:7036AC 上板首测 WDT 崩溃(见 ../移植指南.md §7),根因
> GCC 软浮点无 DSP 加速。本方案=用 Cadence NNLib 的 HiFi5 f32 SIMD 内核替换热点循环,
> 与"贵司 xt-clang 环境重编"合并为同一条路(NNLib 内核含 Xtensa 汇编,只能 xt-clang 编)。
> 资料来源:HiFi5-NNLib-ProgrammersGuide(-API/-Performance).pdf + 本地
> XNNC/assets/library/nnlib 全套源码。

## 1. 可行性结论(先说答案)

- **f32 全链路可映射,零量化**——数值保持桌面 7.7e-7 等级,不引入 int16 那类精度损失
- 本地 NNLib(XNNC 3.2.2 assets)已有全部所需 hifi5 f32 内核源码,**拷进 SDK 即用**
- HiFi5 f32 双 MAC SIMD:matvec/conv 类 ≈2 MAC/cycle;GCC 软浮点 ≈每 50~80 cycle 1 MAC
  → **整体预估 50~100 倍提速**
- 窗口耗时预估:~700K MAC → <1M cycles → **<5ms @192MHz**,32ms 实时预算余量 6 倍+,
  WDT 问题根除,Opus 编码线程不再被饿死

## 2. 算子映射表(Silero 层 → NNLib 内核)

| Silero 层 | 计算量占比 | 现实现 | 替换为 | 源文件(nnlib/src/algo/kernels/) |
|---|---|---|---|---|
| STFT Conv1d(258×256,s64) | ~大 | 手写 C 循环 | `xa_nn_conv1d_std_f32` | cnn/hifi5/xa_nn_conv1d_std_f32.c |
| first_layer / encoder 的 proj、pw_conv(1×1) | 中 | 手写 C | `xa_nn_conv2d_pointwise_f32` | cnn/hifi5/xa_nn_conv2d_pointwise_f32.c |
| dw_conv(逐通道 k7) | 中 | 手写 C | `xa_nn_conv2d_depthwise_f32` | cnn/hifi5/xa_nn_conv2d_depthwise_f32.c |
| **LSTM 门 matvec(2 层×8 步×4 门)** | **~大头** | 手写 C | `xa_nn_matXvec_f32xf32_f32`(batch 版) | matXvec/hifi5/xa_nn_matXvec_f32[_batch].c |
| LSTM sigmoid/tanh 激活 | 中 | expf 手写 | `xa_nn_vec_sigmoid_32_32` / `xa_nn_vec_tanh_32_32` | rnn|activate 相关 hifi5 内核 |
| ReLU | 小 | 手写 C | `xa_nn_vec_relu_32_32` | — |
| 残差相加 | 小 | 手写 C | `xa_nn_elm_add_f32xf32_f32` | — |
| decoder FC | 小 | 手写 C | `xa_nn_matmul_f32xf32_f32` 或 matXvec | matXvec/hifi5/xa_nn_matmul_f32.c |
| 归一化 sqrt/div | 小 | sqrtf | NatureDSP 库(本地同套资料) | — |

注:NNLib 另有整只 `xa_nnlib_lstm_process`(xa_nnlib_lstm_api.h),但 hifi5 内核是
int8 版——float 路线不用它,用 matXvec+vec 激活组合。性能表里还出现
`matXvec_f32xf32_f32_sigmoid/tanh` 融合版(若本地版本无,分两步调,开销差异小)。

## 3. 实施步骤(在现有"交接三步"之上增量)

1. **拷库**:把 `XNNC/assets/library/nnlib/src/algo/kernels/{cnn,matXvec,rnn,...}/hifi5/`
   中上表涉及的 .c + `src/include/nnlib/` 头,拷入 SDK `components/audio_algorithm/processor/src/nnlib/`
   (SConscript 自动扫描会编入;xt-clang 编译这些文件,GCC 编不了——所以这一步天然属于
   贵司 license 环境)。
2. **替换热点**:新写 `silero_vad_nn.c`(接口同 `sv_process/sv_reset`,内部改调 NNLib),
   保持 `wq_silero_vad.c` 五函数封装不动。**替换版先在 Mac 上用 NNLib 的 C 参考实现
   验证数值一致**(NNLib 带 testbench/参考 C),再上芯片。
3. **编译**:defconfig 保持 `CONFIG_XT_TOOLCHAIN_XT_CLANG=y`,正常 scons。
   权重/内存布局不变(601KB 权重照旧,SV_WSEC 宏二选一)。

## 4. 预期收益与验证

| 项 | GCC 软浮点(现状) | xt-clang 标量 | xt-clang + NNLib |
|---|---|---|---|
| 单窗耗时(估) | 数百 ms(超 WDT) | ~15-30ms | **<5ms** |
| 实时性(32ms 预算) | ✗ | 勉强 | ✅ 余量 6 倍+ |
| 数值 | 基准 | 同 | 同(f32 全程) |

验证:同一 [SHIL] 自检跑 hil_parse(线 0.1)+ cpu_usage(wq-debug-tools)量占用,
两项过=移植收官。

## 5. 风险与备注

- NNLib 内核对输入输出指针有对齐要求(通常 8/16 字节),LSTM 状态缓冲需按其对齐分配
- hifi4/hifi5 内核同名目录并存,拷贝时**只取 hifi5**
- NNLib 许可:随 XNNC SDK 分发,限 Cadence Xtensa 核使用(与 TFLM SDK 同源的宽松条款,
  仓库已有先例);不入公开仓库,集成时从 XNNC 包取

## 6. GCC 编译 NNLib 的实测结论(2026-09-29,重要)

用 xtensa-wuqi-elf-gcc 12.2 + RI-2020.4 的 cstub 头实测编译 hifi5 f32 内核:

| 实验 | 结果 |
|---|---|
| 加 `-DCOMPILER_XTENSA=1` 绕过编译器检查 | ✅ 通过(xa_nn_common.h 的 #error 门) |
| dot_prod/conv1d_std/pointwise_f32 | ✅ 可编 |
| matXvec/matmul/depthwise_f32 | ❌ XCC 向量字面量转型(`(xtfloatx2)0.0f`)GCC 不认 |
| **cstub 本质** | ❌ **C 模拟层**——MADD_SX2 等指令的实现是查表+位操作仿真(为 x86 功能仿真设计),**编进去也不会快,反而更慢** |
| GCC 汇编器认 TIE/FPU 指令 | ❌ f 寄存器/madd.s 均不认(工具链按无 FPU 通用配置构建) |

**定论:NNLib 快速版只能由 xt-clang(license 环境)编译——真 TIE 指令只有 XCC/xt-clang
代码生成器会发射。GCC 侧的天花板=朴素 C 软浮点(即 B 包)。**
include 路径备忘(供 xt-clang 集成参考):nnlib/src/include(+ /nnlib)、algo/common/include、
algo/ndsp/hifi5/include、algo/kernels/basic/hifi5、RI-2020.4 wq_hifi5_asic/src/cstub。
