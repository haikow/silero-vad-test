
## 7. 落地结果(2026-10-11, 三步优化完成)

原 §2 映射表的最终形态(实测 232→96.7ms, 40 窗全程 ≤0.0001):

| 层 | 最终实现 | 备注 |
|---|---|---|
| STFT | **加窗 256 点基2 FFT(自写 C)** | basis 结构=行0窗/行j=cos/行129+j=-sin(PC 验证); conv1d 弃用(布局坑+流量) |
| 1x1 卷积 | **xa_nn_matmul_f32xf32_f32** | 输入[T][IC]/输出[T][OC] 前后转置; 权重 [OC][IC] 原样 |
| LSTM 门 | **xa_nn_matXvec_f32xf32_f32** | 一次调用 W·x+R·h+bias; R 暂存 TCM; W 留 flash |
| dwconv5 | 标量 + 内嵌 ReLU | ONNX 每块 dw→ReLU→pw |
| 残差/decoder | 标量 | 量小 |
| 激活 | 标量 expf/tanhf | 可再上 fused matXvec_sigmoid/tanh(非主项) |

**剩余 96.7ms 的构成**: STFT 9.7 / norm 5.9 / first 30 / enc 14.8 / LSTM 35.7 / dec 0.4。
无法自行压缩的 = 权重 flash 流量(LSTM W 128KB + first/enc 75KB, TCM 剩余 69KB 装不下)。
**<32ms(真麦实时线)的最后一公里 = 厂商 preset 重配释放 ~140KB TCM** → 全权重暂存 → ~25-30ms。
历史教训全记录: matmul 输入 [T][IC](CC)、conv1d kernel [OC][ICW_pad×KH](GG)、
分块输出行距(MM)、暂存两路径预算(KK/PP)、PSRAM≈flash 带宽(PP)。
