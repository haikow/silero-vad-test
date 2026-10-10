# v4 int8 vs v4 float32 全语料对比报告(2026-10-11)

> 动机: 评估"更小更量化的权重模型"能否替代现行 float32(609KB 权重/单窗 96.7ms)。
> 方法: 两模型同口径流式(512 样本窗)跑全部 19 条 CI 语料(25384 窗),
> 同一四件套判定(SpeechSegmenter: 0.6/0.35 迟滞+最短语音/静音)后比对。

## 结论(先说)

**int8 质量可接受、收益真实, 但建议作为"厂商 preset 重配"之外的自主备选路线**,
决策点=真麦时间线是否等得起厂商(见 §4)。

## 1. 模型

| | 权重 | 量化范围 |
|---|---|---|
| v4 float32(现行) | 609KB | 全 float |
| **v4 int8(sherpa)** | **158KB(↓3.9×)** | 18 个 Conv→ConvInteger(u8/i8), LSTM/偏置保持 float |

## 2. 逐窗精度(golden 40 窗)

平均偏差 0.011 / 最大 0.24(窗 23 过渡窗: 0.15 vs 0.39) / **阈值 0.5 判定翻转 0/40**。
静音窗几乎无差(0.0415 vs 0.0415); 差异集中在过渡窗。

## 3. 全语料判定对比(19 条 / 25384 窗)

- 门控(实时 on/off)不一致: **1060 窗(4.18%)**; 10/19 条语料零翻转(噪音/古典乐/雨声/
  冒烟全零), 差异集中在两类:
  - **musan_noise_wrtc_u8: 611 翻转, float 1 段 vs int8 0 段** —— int8 在该噪音上
    **不误触发**, float 有一段假触发(int8 更优)
  - musan_music_jamendo: 254 翻转, float 5 段 10.0s 假触发 vs int8 2 段 1.8s(int8 更优)
- 语音类(fleurs/libri)边界小碎: int8 段数略多(段内中谷跌破阈值), fleurs_a 106 翻转
- **音乐假触发 int8 反而更好**: jamendo 纯音乐 float 误触发 5 段 10.0s, int8 仅 2 段 1.8s
- **语音分段 int8 略碎**: fleurs_a(GT 10 段) float 12 段 / int8 14 段(段内中谷跌破阈值多分);
  边界最大差 0.6s(musan_noise_hard 的 1 个假触发段)
- 与 CI 评分一致(int8 90.7 vs float 92.2): 误差轮廓不同、总量级相当; 判定级一致率 95.8%

## 4. 路线建议

| 路线 | 全窗预估 | 代价 | 前提 |
|---|---|---|---|
| A. 现状 float32 + 厂商 preset 重配 | ~25-30ms | 零(等厂商) | 厂商释放 ~140KB TCM |
| B. **LSTM W 换 f16 混合精度**(中间档) | ~85ms | 小(f16 内核+精度重验) | 无 |
| C. **全 int8**(本报告) | ~60ms | 1-2 天: matmul/matXvec 8x8 内核+量化标定+验收标准重构(判定一致性为准) | 无 |

- B: W 128KB→64KB(f16 流量减半), HiFi5 有 f16 SIMD/NNLib 有 f16 混合内核, 精度损失远小于 int8
- C: 流量 1/4(LSTM W 23→6ms, first/enc 14→4ms), 且权重 158KB 直接收编;
  验收基线从"逐窗 0.0001"改为"判定一致性"(本报告即基线)
- 若真麦时间线紧(等不及厂商), C 是唯一自主达 <32ms 的路; 否则 A 最省

## 复现

对比脚本: [int8_vs_float32.py](int8_vs_float32.py)(VadRunner 流式 + SpeechSegmenter 判定,
两模型同口径 19 条语料对头)。数据: golden 40 窗在 `porting/firmware/`, 语料在 `test_fixtures/`。


## 5. 量化标定与板上内核勘察(2026-10-11, quant_sim.py)

### PC 标定结果(golden 40 窗, 工具=quant_sim.py)

| 模式 | 最大偏差 | 判定翻转 |
|---|---|---|
| per-tensor int8 权重 | 0.9987 | **28/40** ❌(离群值压死 scale, 死路) |
| **per-row int8 权重, 激活 float(W8A32)** | **0.0026** | **0/40** ✅ |
| **per-row int8 权重 + 激活 int8(W8A8)** | **0.18(窗23)** | **1/40**(边界窗) ✅ |

→ **per-row scale 是必需的**(per-tensor 死); W8A8 的 1 次翻转在窗 23(0.39/0.5 边界),
迟滞(continue_hit_cnt)可吸收。

### 板上内核勘察(libxa_nnlib.a 实测)

- **matXvec_8x8_32 / 8x16_32 存在**(int8 权重×激活 → **int32 输出**)——int32 出口让我们
  在标量侧做 per-row 反量化(256 次乘法/调用, 忽略不计), 绕开"uniform-shift 只能 per-tensor"的死路
- matmul 只有 8x8_8/16x16_16(int 出口, 无 int32 版) → 1x1 层改用 **matXvec_8x16_32 逐时间步**
  (激活 int16 更细, 权重复用)

### 修正后的 C 路线板上设计

1. 全部权重 int8 per-row(85KB: LSTM 64 + first/enc 19 + dw 2)+ scales 表; **basis 砍到窗行**
   (1KB, 原 258KB)——数据段 859KB → ~90KB
2. LSTM = matXvec_8x8_32 + per-row 反量化; 1x1 = matXvec_8x16_32 逐 t + per-row 反量化;
   dwconv5 标量直读 int8; 激活 x/h int8 per-tensor(静态标定)
3. **关键红利: int8 权重 85KB ≤ TCM 运行时余量(量产 197KB)**——全权重 TCM 暂存,
   flash 流量归零, **无需厂商 preset 重配**
4. 预估全窗 **~23-30ms** ✓(STFT 9.7 + norm 5.9 + 各层计算 ~3 + 激活 ~5 + 余量)
5. 验收标准: 判定一致性(本报告 §3 基线)+ p 容差 ±0.25(float32 板基线对照)

### 工作量与风险

- PC 侧已就绪: quant_sim.py(标定+仿真)即数值蓝本
- 板侧待做: int8 权重 C 数组生成、matXvec_8x16_32/8x8_32 接入(注意 8x8_32 的
  vec_offset/acc 语义需读源码)、TCM 暂存、HIL 重验收 —— 约 1-2 天
- 风险: matXvec_8x8_32 的累加器饱和(int8×int8×64 项 ≤ 127×127×64 ≈ 1M < 2^31 ✓ 无饱和);
  激活范围漂移(CI 语料 vs golden 标定) → 饱和监测 + 5% 余量已留
