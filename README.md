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

push / PR 自动触发两个 job, 各跑同一套 pytest 用例 (2026-09-22 起 69 条)。
**run 页面可见每个用例的 pass/fail**: 打开某次 Actions run -> 摘要区有按用例的结果表格,
下方 Annotations 逐条列出用例名; `pytest-report-*.xml` 工件含完整报告,
`scenario_metrics.json` 工件是场景指标表 (将来固件 HIL 跑同一组场景对比用)。

| Job | 运行位置 | 内容 |
|---|---|---|
| 算法基准回归 | GitHub 云端 (ubuntu) | 全部用例 (模型现场下载) |
| 本地 CI 机冒烟 | self-hosted runner (`vad-lab`) | 同一套用例, 验证本地链路; 模型优先从本机缓存复制 |
| 固件在环 (HIL) | self-hosted (`vad-lab-mac`), 手动触发 | **固件到位后启用**: 烧录 -> 下发测试音频 -> 串口回收逐窗概率 -> 与 PC 基准比对 (偏差<=0.1) |

### 用例清单 (tests/, 按组)

| 组 | 用例 | 判定 |
|---|---|---|
| 素材完整性 (3) | 模型文件存在 / 素材 16kHz / 基准窗数与素材一致 | 缺东西给人话报错, 不是报错栈 |
| 算法回归 (6, 每模型 x2) | 逐窗最大偏差 / 概率曲线相关系数 / 窗数一致 | 偏差<=0.05, 相关系数>=0.995 (与 `ci/run_tests.py` 同源) |
| 语义行为 (8, 每模型 x4) | 语音段均值>=0.5 / 噪声段均值<=0.2 / 区分度>=0.3 / 检出区间覆盖真实语音>=60% 且漏进噪声<=1.5s | 不止"和基准一样", 还得"像个人声检测器" |
| 流式链路 (5) | 推理确定性(两次一致) / **状态确实在窗间传递**(每窗重置必挂) / HIL mock 验收通过 | 守住嵌入式移植头号事故点 |
| 判定四件套 (10) | `vad_decision.SpeechSegmenter` 规格: 迟滞带持续 / 最短静音关段 / 段尾不含静音 / 最短语音滤毛刺 / 收尾 flush / 连发多段 | 纯函数单测; 将来固件 C 按"同输入同输出"对照移植 |
| 场景矩阵 (20) | 电平矩阵(peak 0.05~0.8) / 合成噪声误报(白粉褐 x2 强度) / SNR 阶梯(20/10/5dB, 按缩放后语音 RMS 定标) / 噪声零误触发集成 | 素材从已提交语音+种子噪声现场合成, 零新增文件; 阈值按 2026-09-21 实测留余量 |
| 咖啡馆 babble (11) | 真实咖啡馆人声噪声: 裸噪声四件套误触发率 + 咖啡馆噪声下 SNR(10/5/0dB) 检出 | int8 误触发 0 段/60s、0dB SNR 检出覆盖 98% 为产品断言; uint8 被 babble 迷惑(9 段/60s)为特征化锁定 |
| 雨声场景 (6) | 雨打树叶+采菌脆响误报 + 雨中轻声旁白(~-10dB SNR)检出/漏检 | 两模型雨声 0 误触发; int8 检出旁白 5 段/覆盖 55%(产品断言), uint8 整体漏检(特征化: 低SNR假阴性) |

场景实测结论 (Mac M3, ORT CPU, 种子固定):

- **电平**: int8 各档 0.73~0.83 纹丝不动; uint8 从 0.78(peak0.8) 掉到 0.46(peak0.05) —— README"v5 电平敏感"结论的自动化锁定
- **合成噪声**: 白/粉/褐两模型均值都 <=0.06, 误报无忧; **真实环境声见下条**
- **SNR(白噪声)**: int8 20/10/5dB 均值 0.93/0.92/0.89, 5dB 下检出覆盖仍 93%; uint8 0.68/0.56/0.34 深噪声渐退
- **真实咖啡馆 babble** (B 站 BV1eW41137G7 1:30-2:30, 60s, `test_fixtures/cafe_noise_16k.wav`):
  裸噪声下 uint8 四件套误触发 **9 段/60s**(13.5s 误判语音), int8 **0 段**;
  咖啡馆噪声 SNR 混合 int8 10/5/0dB = 0.94/0.96/0.92(覆盖 98%+, 0dB 仍稳), uint8 0.32/0.12/0.07(深噪声趴掉)
- **真实雨声** (B 站 BV1tkohB5Eqc, UP 云南山师傅: 90-150s 纯雨声 60s + 25-62s 轻声旁白 37s,
  `test_fixtures/rain_{noise,speech}_16k.wav`): 雨打树叶+采菌脆响两模型均 **0 误触发**;
  雨中轻声旁白(~-10dB SNR, 峰值 0.29) int8 检出 **5 段/覆盖 55%**, uint8 **整体漏检**(均值 0.004,
  放大到峰值 0.8 仍漏 -> 排除电平因果, 系低 SNR 假阴性)
- **uint8(v5) 双向短板证据链闭环**: 咖啡馆误报(假阳性 9 段/60s) + 雨中漏检(假阴性 0 段) +
  合成电平/SNR 阶梯; int8(v4) 两个方向全部稳住 —— 嘈杂/低信号环境可用性分界 = int8, 移植选型最终实锤

本地跑: `pytest tests/ -v` (模型先就位); 新增用例放 `tests/`, fixtures 在 `conftest.py` (推理只算一次, 全组共享)。

```bash
# 基准回归旧入口 (维护基准用, CI 已改走 pytest)
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
  `VAD_MODELS_CACHE=C:\Users\<用户名>\ZCodeProject\silero-vad-test\models`; 未配置时自动下载
  (已带国内镜像 fallback)
- **Mac CI 机 (已接入, 2026-09-21, MacBook Air M3)**: runner 在 `~/actions-runner`, 名称 `vad-lab-mac`,
  标签 `vad-lab,vad-lab-mac` —— `vad-lab` 与 Windows 机共享接单(冒烟), `vad-lab-mac` 供 HIL 固定派单
  (板子只插一台机器, 不能让空闲随机接单)。
  - 常驻: launchd 用户代理 `~/Library/LaunchAgents/com.github.actions.runner.vad-lab-mac.plist`
    (RunAtLoad + KeepAlive, 用 `caffeinate -i` 包裹 `run.sh` 防空闲睡眠, 免 sudo)。
    **合盖仍会睡眠** (MacBook), 合盖当 CI 机用需 `sudo pmset -a disablesleep 1`。
  - `.env`: `VAD_MODELS_CACHE=/Users/a1234/vad-lab/models`(模型缓存, job 零网络拉模型) +
    `PATH=/opt/homebrew/bin:...` (brew `python@3.12 libomp`; 系统无 `python` 命令, workflow 用 venv bootstrap)
  - 排查: `tail ~/actions-runner/runner-service.{out,err}.log`; 重启
    `launchctl kickstart -k gui/$(id -u)/com.github.actions.runner.vad-lab-mac`
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

## XNNC 编译环境(已就绪, 2026-09-20)

`porting/` 目录包含 WQ7036AC(HiFi5 DSP + Neo NPU)的 **XNNC/Cadence NeuroWeave SDK 3.2.2**
编译环境:Dockerfile(Apple Silicon Mac 经 Colima+Rosetta 跑 x86-64,亦适用任意 x86 Linux)、
一键进入脚本、Silero v4 float 的编译配置(流式多输入 + 自动量化 + Neo NPU 卸载)、
校准集生成脚本与踩坑记录。环境与 PC 基准已在同容器内联合验证(相关系数 1.00000)。
**唯一待补**:物奇 SDK 的 Xtensa 核配置(`xtensa_system`),拿到后即可跑通量化→codegen 全流程。
详见 [porting/README.md](porting/README.md)。
