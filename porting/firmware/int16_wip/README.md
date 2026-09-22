# int16 定点版(WIP,功能原型级)—— 已决策: 留档暂停, 不继续投入

**决策(2026-09-22)**: 生产用 float+xt-clang(位精确 7.7e-7, FPU 余量 ~11% CPU);
本目录留档不删(含量化生成器/验证方法论/7 个 bug 修复记录), 暂停优化。
**重启条件(上板实测触发其一)**: ① always-on 功耗超标; ② flash 分区紧张。
重启时首选改进: 残差链激活保 int32(半天, 预计一致率 92%→97%+), 其次 Q15 LSTM 状态、QAT。


目标: 摆脱 FPU 依赖(软浮点 GCC 也能实时)+ 权重 601KB→309KB。

## 当前状态(2026-09-22)

| 指标 | 结果 |
|---|---|
| 5325 窗场景语料(smoke/咖啡馆/雨声×2) | 阈值(0.5)判定一致率 **92.1%**, 平均偏差 0.092, 最大 0.69 |
| 40 窗 golden | 静音窗基本吻合; 语音窗偏低 ~0.1-0.15 |

**结论: 功能原型级, 未达逐窗验收线(0.05)。生产请用 float 版(已位精确 7.7e-7)+
xt-clang(FPU)——license 在物奇固件团队手里, 交接见 ../integration/README.md。**

## 已验证正确(逐级 numpy/ORT 对照, 差=0)

STFT(手算逐位一致)、feat 转换、全部 conv 层(en3/enc12 等隔离验证)、
LSTM 单步(零状态)、量化常数/权重数值、isqrt(重写为精确位扫描)、ln LUT(513 点+插值)。

## 已修的 7 个 bug(存档供续作)

STFT 存储溢出(int32)→ 右移 8 位重定标; enc7 残差 Q10/Q9 错位; LSTM c 更新
Q15→Q11 差 4 位; h 更新 Q15→Q14 差 1 位; h 反馈预截断(改先乘后移); requant
截断改舍入; ln LUT 向下偏(中点采样+插值)。

## 剩余误差根因(下一步方向)

逐层 int16 静态 Q 格式: 小值激活相对误差大(±0.5 LSB / 小值), 8 层 CNN 链式累积
~2-5%, 再被 LSTM 状态动力学复利放大 ~6 倍。改进选项:
1. 残差链激活保 int32(Q 同格式), 只在块边界压 int16;
2. LSTM h/c 提到 Q15/int32;
3. 逐层 Q 按实际分布(而非 absmax)细调;
4. 或直接 int8 权重 + int16 激活的标准量化训练(QAT)——精度最优但工作量大。

## 复现

```bash
cc -O2 -std=c99 -o test_fixed test_fixed.c silero_vad_fixed.c silero_vad_fixed_weights.c -lm && ./test_fixed
cc -O2 -std=c99 -o cmp_corpus cmp_corpus.c silero_vad_fixed.c silero_vad_fixed_weights.c \
   ../silero_vad.c ../silero_vad_weights.c -lm && ./cmp_corpus   # 需 calib_x.f32 + ../golden*
# 权重再生成(需 onnx): python3 gen_weights_fixed.py silero_vad_v4_float.onnx
```
