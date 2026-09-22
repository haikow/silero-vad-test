# libwq_silero_vad —— Silero VAD 的物奇 libwq_sw_vad 替换库

用纯 C99 实现 Silero VAD v4(float),对齐物奇 `libwq_sw_vad` 的 5 函数接口,
**processor 层(wq-adk 的 sw_vad.c)零改动**,替换链接库即可把原厂 VAD(WebRTC VAD)
换成 Silero。

## 验证结果(2026-09-22, Mac/clang 实测)

| 测试 | 结果 |
|---|---|
| 浮点直通(C vs onnxruntime float) | 40 窗最大偏差 **7.7e-7** ✅ |
| WQ 接口攒帧管道纯度(同量化数据) | 最大偏差 **0.0** ✅ |
| int16 输入量化效应(vs float 基准) | 最大偏差 1.7e-2(HIL 验收线 0.1)✅ |

## 文件

- `silero_vad.c/h` — 网络前向传播(可移植,无依赖,只用 libm)
- `silero_vad_weights.c` — 权重(153,796 floats / 601KB const,由 `gen_weights.py` 生成)
- `silero_vad_consts.h` — 归一化常数(log(幅度·2²⁰+1))
- `wq_silero_vad.c/h` — 物奇 5 函数接口 + 320→512 攒帧(20ms 帧→32ms 窗,无重叠,
  320×8=2560=512×5 自然对齐)
- `test_ref.c` — 浮点一致性测试;`test_wq.c` — 接口/攒帧/量化三合一测试
- `gen_weights.py` / `gen_golden.py` — 权重与黄金向量再生成(需 onnx + onnxruntime
  + soundfile 环境,如 XNNC 容器)
- `golden_x.f32 / golden_prob.f32` — 黄金向量(smoke_test 音频 40 窗)

## 网络结构(逐节点从 ONNX 转写,时间维度经 ORT 数值校准)

```
x(512) → reflect pad(96,96) → STFT Conv(258×256, stride64) → 8帧×129幅度
→ 自适应归一化: log(mag·2²⁰+1),减去 reflect 补边平滑均值(标量)
→ feat 258×8 = [mag | norm-log]
→ first_layer 残差块 258→16 (dw k5 + pw)
→ encoder: s2(16→16) 残差(16→32) s2(32→32) 残差(32→32) s2(32→32) 残差(32→64) conv(64→64)
  时间 8→4→2→1
→ 2 层 LSTM(64, 门序 i,o,f,c) → relu → 1×1(64→1) → sigmoid → prob
```

调试历程中校准的三个易错点(给后续维护者):STFT 是 96/96 reflect(不是 256/256,
输出 8 帧);归一化被减项是均值序列 reflect 补边 3+3 再 conv7 取均值的全局标量;
conv1x1 输入布局是 [C][T] 行主序。

## 固件集成(wq-adk)

1. `make lib` → `libwq_silero_vad.a`
2. wq-adk 构建中把 `audio_algorithm/lib/wq_sw_vad/libwq_sw_vad.a` 替换为本库
   (头文件接口同名,`wq_sw_vad.h` 直接用本目录的 `wq_silero_vad.h` 或保持原头)
3. 阈值默认 0.5,`-DWQ_SILERO_THRESHOLD=0.6` 可调;迟滞用 processor 层现成的
   `continue_hit_cnt/continue_stop_cnt`(20ms 帧:0.5s=25 帧,0.25s≈13 帧)
4. 内存:权重 601KB const(flash)+ 静态缓冲 ~26KB(.bss)+ 句柄 3KB;
   HiFi5 有单精度 FPU,float 实现可直接跑(约 1M MAC/窗,余量极大)
5. HIL 验收:`wq_silero_last_prob()` 取逐窗概率,UART 上报比对 PC 基准(≤0.1)

## 后续优化项(按需)

- int8/int16 量化版(权重 601KB→154KB,功耗更优;当前 float 精度已是满级)
- Xtensa 向量化(HiFi5 DSP 指令,conv/LSTM 点积用 intrinsics)
- 权重放 flash 由 init 加载(当前编译进固件 const 区)
