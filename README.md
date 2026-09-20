# Silero VAD 量化模型本地测试（uint8 vs int8）

目标：在 PC 上验证两个量化版 Silero VAD 的效果，为移植到 WQ7036AC 嵌入式设备做准备。

## 目录结构

```
silero-vad-test/
├── download_models.py          # 下载两个量化模型到 models/
├── models/                     # (gitignore, 脚本下载)
│   ├── silero_vad_uint8.onnx   # 639 KB, onnx-community 导出, v5 接口
│   └── silero_vad_int8.onnx    # 208 KB, sherpa-onnx 导出, v4 接口
├── audio/                      # (gitignore, 测试时自动生成)
│   ├── speech_raw.wav          # Windows TTS 生成的原始语音 (22.05 kHz)
│   └── test_mixed_16k.wav      # 测试音频: 白噪声(1.5s) + 语音 + 白噪声(1.5s), 16 kHz
└── test_vad.py                 # 对比测试脚本
```

## 运行

```bash
pip install onnxruntime numpy soundfile   # Python 3.10+ 均可
python download_models.py                 # 下载两个量化模型到 models/ (~850 KB)
python test_vad.py
```

模型和测试音频不入库：模型用 `download_models.py` 拉取；测试音频由 `test_vad.py` 首次运行时自动生成（需要 Windows TTS，非 Windows 平台可自行替换 `audio/speech_raw.wav` 为任意 16 kHz 可用的语音 wav）。

> 许可证提示：代码部分可自由使用；两个模型的版权归 snakers4/silero-vad 项目（其许可证对商用有限制，商用前请查阅原仓库 LICENSE）。

## 两个模型的关键差异

| | uint8 (onnx-community) | int8 (sherpa-onnx) |
|---|---|---|
| 模型版本 | Silero VAD **v5** | Silero VAD **v4** |
| 大小 | 639 KB | 208 KB |
| 输入 | `input`(1,N), `state`(2,1,**128**), `sr`(标量) | `x`(1,512), `h`(2,1,**64**), `c`(2,1,64) |
| 状态 | 单个 state 张量 | LSTM 的 h 和 c 分开 |
| 采样率 | 运行时通过 `sr` 传入（16k 用 512 窗，8k 用 256 窗） | 模型内固定 16 kHz，512 窗 |
| 噪声段平均概率 | 0.018 | 0.102 |
| 语音段平均概率 | 0.803 | 0.718 |
| 每窗耗时 (PC) | ~0.39 ms | ~0.92 ms |
| 实时率 RTF (PC) | 0.012 | 0.029 |

两个模型的概率曲线相关系数约 0.86，行为一致；uint8/v5 在噪声抑制和边界干净度上略好。

## 流式调用要点（两个模型通用）

1. 音频必须 **16 kHz、float32、[-1,1]**、按 **512 样本（32 ms）** 一窗送入。
2. 状态张量（v5 的 `state` / v4 的 `h`,`c`）必须在窗口间传递，**不能每窗重置**；只在重新开始一段音频时清零。
3. v5 的 `sr` 输入是 `np.array(16000, dtype=np.int64)` 标量。
4. 输出 `output[0][0]` 即人声概率，常用阈值 0.5；工程上建议配合迟滞（如 >0.6 判开始、<0.35 判结束）+ 最短语音时长/最短静音时长，避免抖动。

## 移植到 WQ7036AC 的建议

### 模型内部结构（onnx 工具检查结果）

- **uint8 (v5)**：顶层只有 `Identity/Equal/If` 三个算子 —— 真正的网络在 `If` 的两个子图里（按 `sr` 是 8k 还是 16k 动态选分支）。**绝大多数嵌入式模型转换工具不支持 If 控制流**，移植前必须先把 sr=16000 固定、折叠分支导出平铺图（或向 onnx-community 要无分支导出）。
- **int8 (v4)**：平铺计算图，无控制流。量化方式为**动态量化**：`ConvInteger`(18) + `DynamicQuantizeLinear`(15) + `DynamicQuantizeLSTM`(2, onnxruntime 私有 contrib 算子)。权重以 int8 存储，激活推理时动态量化。

### 移植路径建议

- 芯片工具链一般不直接吃 ONNX：先确认 WQ 的 SDK 支持的模型格式（常见为厂商私有网络格式或 CMSIS-NN 的 C 数组导出），`Conv` + `LSTM` 的算子支持度是关键。
- 两个模型核心都是 CNN + LSTM 结构，无 Attention/大矩阵乘，计算量很小（每窗毫瓦级），16 位定点或重新定点化通常都能跑，算力不是瓶颈（你已确认够用）。
- **优先以 sherpa 的 int8 (v4) 为蓝本**：体积最小（208 KB）、无控制流、LSTM 状态 64 维（v5 是 128 维，内存减半）、结构版本老且公开资料多（社区有多个 silero-vad v4 的纯 C 移植先例可参考）。
- 两个模型里的量化算子（`DynamicQuantizeLinear`/`ConvInteger`/`DynamicQuantizeLSTM`）都是为 ONNX Runtime CPU 执行设计的，芯片 SDK 大概率不认识：常见做法是**反量化回 float 权重，交给芯片工具链按 int16/int8 重新量化**，精度基本无损（本模型量很轻）。
- 验证方法：把 PC 上 `test_mixed_16k.wav` 的逐窗概率存成基准（可在 test_vad.py 里把 probs 数组 np.save 导出），芯片上跑同样输入逐窗比对，偏差应 <0.1，以此确认移植正确。
