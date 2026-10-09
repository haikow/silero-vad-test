# Silero VAD 的 HiFi5 NNLib 优化方案(GCC 软浮点 → SIMD 内核)

> 2026-09-29 定稿, **2026-10-09 v2 修订(按实测实现校正)**。动因:7036AC 上板首测 WDT 崩溃,根因
> GCC 软浮点无 DSP 加速。本方案=用 Cadence NNLib 的 HiFi5 f32 SIMD 内核替换热点循环,
> 与"贵司 xt-clang 环境重编"合并为同一条路(NNLib 内核含 Xtensa 汇编,只能 xt-clang 编)。
> 资料来源:HiFi5-NNLib-ProgrammersGuide(-API/-Performance).pdf + 本地
> XNNC/assets/library/nnlib 全套源码。
>
> **v2 状态: 已实现并集成(X 包, silero_vad_nn.c), 待上板验收。v1 映射表有三处与实测
> 实现不符,已在 §2 修正——尤其注意激活函数版本(见⚠行)。**

## 1. 可行性结论(先说答案)

- **f32 全链路可映射,零量化**——数值保持桌面 7.7e-7 等级,不引入 int16 那类精度损失
- 本地 NNLib(XNNC 3.2.2 assets)已有全部所需 hifi5 f32 内核源码,**拷进 SDK 即用**
- HiFi5 f32 双 MAC SIMD:matvec/conv 类 ≈2 MAC/cycle;GCC 软浮点 ≈每 50~80 cycle 1 MAC
  → **整体预估 50~100 倍提速**
- 窗口耗时预估:~700K MAC → <1M cycles → **<5ms @192MHz**,32ms 实时预算余量 6 倍+,
  WDT 问题根除,Opus 编码线程不再被饿死

## 2. 算子映射表(Silero 层 → NNLib 内核)

| Silero 层 | 计算量占比 | v1 原方案 | **v2 实测实现** | 备注 |
|---|---|---|---|---|
| STFT Conv1d(258×256,s64) | ~80%(52.8万MAC) | conv1d_std_f32 | `xa_nn_conv1d_std_f32` ✓同 | IC=1 时 CHW/HWC 同构, fmt=0 出 [8][258]; **必须给 bias(零数组)和 p_scratch(状态结构,不能 NULL——参数检查拒)** |
| 1×1 卷积(proj/pw_conv) | 中 | ~~conv2d_pointwise~~ | `xa_nn_matmul_f32xf32_f32` **CHW 路径** | 全程 [C][T] 零转置零重排: out_offset=1 / out_stride=vec_count |
| dw_conv(逐通道 k7/k5) | 中 | ~~conv2d_depthwise~~ | **保留标量**(~3% 算量) | NNLib depthwise 两分支布局文档含混, 降险不动 |
| **LSTM 门 matvec** | 中(~66K MAC) | matXvec batch | `xa_nn_matXvec_f32xf32_f32` ✓同 | 256 行=4 门, Wb+Rb 预合并 |
| ⚠ LSTM sigmoid/tanh | 中 | ~~`_32_32`~~ | **`xa_nn_vec_sigmoid_f32_f32` / `xa_nn_vec_tanh_f32_f32`** | **v1 表把 32_32 写进了映射表——那是定点 Q 格式, 用了数值必错!** relu 同理用 f32_f32(threshold 参数 0.0f) |
| 残差相加 | 小 | elm_add ✓ | `xa_nn_elm_add_f32xf32_f32` ✓同 | |
| decoder FC | 小 | matmul/matXvec | `xa_nn_dot_prod_f32xf32_f32` | 注意 +num_vecs 参数 |
| 归一化 sqrt/div | 小 | NatureDSP | 保持 libm 标量 | 量小不值得换 |

注:NNLib 另有整只 `xa_nnlib_lstm_process`(xa_nnlib_lstm_api.h),但 hifi5 内核是
int8 版——float 路线不用它,用 matXvec+vec 激活组合。性能表里还出现
`matXvec_f32xf32_f32_sigmoid/tanh` 融合版(若本地版本无,分两步调,开销差异小)。

## 3. 实施步骤(v2 实测路径,比 v1 简化一半)

> **v2 重大简化: ADK 1.4 SDK 自带同核预编译 `lib/xa_nnlib/libxa_nnlib.a`(3.4MB, XCC 编),
> 所需 9 个 f32 符号全齐——不需要拷内核源码,只接头文件!**
> (实测拷源码编译会撞 multiple definition, 且要平 NatureDSP 头链/相对include/-Werror 豁免)

1. **接头文件**(不是拷源码!): 从 `XNNC/assets/library/nnlib/src/` 拷入 SDK
   `processor/inc/`: `include/nnlib/` 目录 + `include/xa_type_def.h` +
   `algo/common/include/*.h` + `algo/ndsp/hifi5/include/*.h`(NatureDSP 头和 tbl 表)+
   kernels 的 `*_state.h`。SConscript 加两个 `-I`(inc 和 inc/nnlib)——见补丁包 0005。
2. **替换实现**: `silero_vad.c` → `silero_vad_nn.c`(本仓库 porting/firmware/nn/,
   接口同 sv_process/sv_reset, 内部全调 libxa_nnlib.a 内核, Mac 已语法验证)。
   `wq_silero_vad.c` 五函数封装不动。
3. **编译**: defconfig 保持 xt-clang, 正常 scons。链接器自动从 .a 拉内核符号。

数值验证: 上板 [SHIL] 40 窗 vs PC 基准 ≤0.1(hil_parse.py)——NNLib 数值正确性的验收器。

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

## 7. v2 实施记录(2026-10-09, X 包)

- 集成载体: 正确基线(ADK 1.4.0.69 glass-stereo-dac)之上的 X 包; 前序 T 包(标量版)
  实测 WDT2 咬死在 sv_process STFT 循环(silero_vad.c:129)——印证本方案"标量不可实时"的预判
- 头文件平铺清单与两个 -I 路径: 见 adk_patch_1.4.0.69/0001+0005(干净树 apply 验证过)
- 集成踩坑实录(全部已解): 拷内核源码→multiple definition(用预编译库即免) /
  xa_nnlib_api.h 在 nnlib/ 子目录互引(加 -I inc/nnlib) / activations 的相对 include
  "../../../ndsp/..."(改平引) / -Werror 与 SDK 头冲突(macro-redefined, 源码版需 pragma
  豁免——用预编译库则整个不存在)
- 待验收: 三核正常 + 无 WDT2 + [SHIL] 40 窗 ≤0.1 + 单窗 <5ms(时间戳差-100ms 限速)
