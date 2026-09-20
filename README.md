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
python test_vad.py                        # 合成基准: 白噪声+TTS语音+白噪声
```

分析任意真实音频（mp3/m4a/ogg/wav，自动转 16 kHz）：

```bash
pip install miniaudio                     # mp3 解码; m4a 等格式另需 av (PyAV)
python analyze_audio.py 路径/xxx.mp3      # 命令行版: 时间线 + 统计
python vad_gui.py 路径/xxx.mp3            # GUI 版: 浏览器里看曲线, 点击跳转播放
```

`vad_gui.py` 生成单文件 HTML（音频以 base64 内嵌，完全离线）：上方音频波形、下方
v5/v4 两条人声概率曲线、红色阈值线、绿色检出区间高亮；**悬停看逐点概率，点击任意
位置跳转到该时刻播放**，播放时黄色游标跟随——哪里有人声一看一听便知。
长音频默认截取前 300 秒（`--max-seconds` 可调）。

## 实时演示 (Siri 风格)

```bash
pip install sounddevice websockets
python vad_realtime.py            # 自动打开浏览器, 需要麦克风
python vad_realtime.py --gate v4  # 波形门控改用 v4/int8 概率
```

对着麦克风说话时波形像 Siri 一样跳动，嘈杂声音（非人声）时波形静止；
两个模型的实时概率条 + 最近 12 秒概率历史同步显示。

### 判定标准 (三层)

| 层 | 逻辑 | 参数位置 |
|---|---|---|
| 状态徽章 | 门控模型概率 >= 0.5 判定人声 (Silero 官方推荐阈值) | JS `>= 0.5` |
| 波形门控 | 概率 0.3 以下关闭 / 0.6 以上全开, 中间线性渐变; 快开(~0.1s)慢关(~1s) | JS `(p-0.3)/0.3` |
| 内部辅助 | v4>0.3 时才校准 AGC 增益; 双模型 <0.15 持续 3s 自动复位 v5 状态 | `make_stream` |

### 已验证的重要特性

- **v5/uint8 对输入电平敏感**: 拾音峰值 <0.3 时 v5 概率被压低 (<0.4), 放大后恢复 (>0.98)。
  已内置 v4 门控 AGC (语音期校准增益至 RMS~0.06, 上限 8x, 噪声期冻结)。
  **嵌入式移植必须在 VAD 前保证健康电平**, v4/int8 对电平鲁棒是工程优势。
- **扬声器回放 ≠ 真实测试**: 视频/音频经"扬声器->麦克风"二次拾音后, 电平衰减 +
  系统回声消除(AEC)会系统性压低概率 (实测同一视频直读 87% 人声, 回放拾音 <35% 峰值)。
  评估请用文件直读 (`vad_gui.py`) 或真人近讲。
- **诊断端点**: 运行中访问 `http://127.0.0.1:8766/debug` (逐窗 RMS/概率/回调帧数直方图)
  和 `/debug.wav` (最近 ~13s 音频), 用于排查实时链路问题。

### 工程级 VAD 判定四件套 (嵌入式建议)

起始阈值高 (0.6) + 结束阈值低 (0.35, 迟滞) + 最短语音时长 (0.5s) + 最短静音时长 (0.25s)。
实测 B 站咖啡馆嘈杂素材: v5 裸阈值 25.5% 误触发 -> 加四件套后 13%。

## 自动化测试平台 (CI)

push / PR 自动触发两个 job (均已验证跑通):

| Job | 运行位置 | 内容 |
|---|---|---|
| 算法基准回归 | GitHub 云端 (ubuntu) | 下载模型 -> 对 `test_fixtures/smoke_test.wav` 跑两个模型 -> 与 `baseline_probs.json` 逐窗比对 (容差 0.05, 相关系数>=0.995) |
| 本地 CI 机冒烟 | self-hosted runner (本地 Windows) | 同样的基准回归, 验证本地 runner 链路; 模型优先从本机缓存复制 |
| 固件在环 (HIL) | self-hosted, 手动触发 | **固件到位后启用**: 烧录 -> 下发测试音频 -> 串口回收逐窗概率 -> 与 PC 基准比对 (偏差<=0.1) |

```bash
# 基准回归 (本地手动跑)
python ci/run_tests.py                     # 回归模式, 超限 exit 1
python ci/run_tests.py --update-baseline   # 模型/素材预期变更后重新生成基准并提交

# 固件在环联调 (无硬件时验证比对逻辑)
python ci/firmware_test.py --mock
python ci/firmware_test.py --port COM7     # 固件协议在 ci/firmware_test.py 的 talk_to_firmware() 中实现
```

### 本地 CI 机 (self-hosted runner) 说明

self-hosted job 已跨平台 (Windows/Mac 通用, 步骤统一用 bash; Windows runner 依赖 Git Bash):

- Windows 本机: runner 在 `C:\actions-runner`, 名称 `ci-local`, 标签 `vad-lab`
- 模型缓存 (可选加速): 在 runner 目录的 `.env` 文件配置, 如
  `VAD_MODELS_CACHE=C:\Users\zbj\ZCodeProject\silero-vad-test\models`; 未配置时自动下载
  (已带国内镜像 fallback)
- **接入 Mac CI 机**:
  1. `gh api -X POST repos/haikow/silero-vad-test/actions/runners/registration-token --jq .token`
  2. 按 GitHub 页面指引下载 osx-x64/arm64 runner, `./config.cmd` (Mac 为 `./config.sh`)
     --url https://github.com/haikow/silero-vad-test --token <TOKEN> --labels vad-lab --unattended
  3. 同样在 `.env` 配置 `VAD_MODELS_CACHE` (可选), `./run.sh` 启动
  4. 两台机器同用 `vad-lab` 标签, 空闲者接单; 基准回归与平台无关
- Windows 开机自启 (可选, 需管理员): `schtasks /create /tn GitHubRunner /tr C:\actions-runner\run.cmd /sc onstart /ru system /f`
  (SYSTEM 账户无用户级 Python, 届时 workflow 中需改用绝对 Python 路径)

### 固件到位后的接入步骤

1. 与固件同事约定串口协议 (逐窗发 512 样本 int16, 返回概率), 填入 `ci/firmware_test.py` 的 `talk_to_firmware()`
2. 在 `.github/workflows/firmware.yml` 填入 SDK 烧录命令 (UART0 = GPIO01/GPIO02)
3. 开发板接 CI 机的串口, GitHub 页面手动触发 "固件在环测试" workflow
4. 验收标准已内置: 固件概率 vs PC 基准逐窗偏差 <= 0.1, 相关系数 >= 0.98

模型和测试音频不入库：模型用 `download_models.py` 拉取；`test_vad.py` 的测试音频首次运行时自动生成（需要 Windows TTS，非 Windows 平台可自行替换 `audio/speech_raw.wav`）。

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
