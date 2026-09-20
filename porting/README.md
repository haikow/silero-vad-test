# WQ7036AC · Silero VAD 移植 —— XNNC 编译环境

在 Apple Silicon Mac(M3 / macOS 26 / 24GB)上搭建并**实测验证**的 Cadence NeuroWeave SDK
(XNNC 3.2.2)完整编译环境,用于把 Silero VAD 模型编译为 WQ7036AC 芯片
HiFi5 DSP + Neo NPU 可执行格式。

## 背景

- **目标芯片**: 物奇 WQ7036AC 蓝牙音频 SoC —— 应用核 RISC-V + 蓝牙核 RISC-V +
  **HiFi5 DSP @192MHz(含 Neo NPU 神经网络引擎)**,VAD 跑在 DSP 子系统上。
- **工具链**: Cadence NeuroWeave SDK 3.2.2(XNNC = Xtensa Neural Network Compiler)。
  官方《neuroweave_ug_NW-2.0.0》要求 Linux x86-64 + Python 3.10.7 + GCC 9.3 + LLVM 16 +
  CMake 3.27.7;**无 macOS / arm64 版**(实测全包 303 个主机二进制 100% x86-64 ELF)。
- **Neo NPU 的 ACCEL_LIB 原生支持 Conv/DSC/LSTM/GRU/Sigmoid/Tanh**,覆盖 Silero VAD
  (CNN+LSTM)全部算子,移植路线成立。

## 目录结构

```
porting/
├── README.md                # 本文件
├── docker/Dockerfile        # 编译环境镜像(ubuntu:22.04 + 全部官方依赖钉死)
├── scripts/
│   ├── start-env.sh         # 一键: 启动 VM → 构建镜像 → 进入容器
│   └── xnnc.sh              # 在容器里执行单条命令(非交互)
├── silero/
│   ├── silero_vad.cfg       # XNNC 编译配置(NonImage 多输入 + streaming + Neo 卸载)
│   ├── state.yaml           # 流式 LSTM 状态映射 [h→new_h, c→new_c](UG §4.10)
│   └── make_calibration.py  # 音频 → 校准/验证数据集(流式目录格式)
├── XNNC/                    # (gitignore) NeuroWeave SDK, 见下节
├── .dockerignore
└── .gitignore
```

镜像内布局: XNNC 在 `/opt/XNNC`,Python 虚拟环境 `/opt/xnnc_venv`(按 SDK requirements.txt
固定版本),宿主仓库根挂载为 `/work`。

## 首次搭建

```bash
# 1. 拿到 XNNC_3.2.2_Linux.tar.gz(团队/原厂渠道, Cadence 授权件不入库)
#    解压到 porting/ 下, 使 porting/XNNC/ 存在:
cd porting && tar -xzf /path/to/XNNC_3.2.2_Linux.tar.gz && cd ..

# 2. 安装 docker CLI 与 colima(brew, 国内源已配)
brew install colima docker docker-buildx

# 3. 一键进环境(首次自动建 VM + 镜像, 约 20-40 分钟, 下载依赖约 6GB)
bash porting/scripts/start-env.sh
```

日常使用 / 停止 VM 与非 Mac(x86 Linux)机器的用法见 `start-env.sh` 内注释;
在 x86 Linux 机器上可跳过 VM 直接 `docker build`。

## 验证结果(2026-09-20 实测;Rosetta 与 QEMU 两种模式各跑过一轮)

| 项 | Rosetta(生产态) | QEMU(降级态*) |
|---|---|---|
| 工具链 | Python 3.10.12 / gcc 9.5 / clang 16.0.6 / cmake 3.27.7 ✓ | 完全一致 ✓ |
| XNNC 3.2.2 | xnnc.py 正常,全部 stage 可用 ✓ | 同左 ✓ |
| PC 基准回归 | uint8/int8 全过,相关系数 1.00000 ✓ | 同左 ✓ |
| Silero cfg 行为 | 停在 xtensa_system 边界(等物奇 SDK, 见下)✓ | 同一位置 ✓ |
| numpy 1200² matmul | **0.04s** | 1.45s(慢约 30 倍) |

\* 降级态 = 误用 `colima start`/`colima restart` 后的状态:**功能零差异,只是慢**;
重新用 `scripts/start-env.sh` 启动即回到 Rosetta。Rosetta 只支持 64 位
(CSTUB 用 `build-cstub64`)。

### 已知边界: 等物奇 SDK 的 Xtensa 核配置

XNNC 任何目标编译(含 flt_inference)都在 config_check 停住,要求 `xtensa_system` 下有
`<core>-params` 核参数文件。该文件在物奇 SDK 提供的 Xtensa 安装里
(RJ-2024.3 + patch 916454,WQ7036 专用 HiFi5 核配置),XNNC 包不带。拿到后:

```bash
# 1. 把 Xtensa 安装目录(或整个物奇 SDK)挂进容器, 如 -v /path/xtensa:/opt/xtensa
# 2. 改 silero/silero_vad.cfg: xtensa_system=/opt/xtensa, xtensa_core=<物奇核名>
# 3. 容器内:
cd /work/porting/silero
python3 /opt/XNNC/Scripts/xnnc.py -c silero_vad.cfg --stage flt_inference   # float 基准
python3 /opt/XNNC/Scripts/xnnc.py -c silero_vad.cfg                         # 全流程(量化+codegen)
```

## Silero 编译材料说明

- 蓝本: sherpa 导出的 **silero v4 float ONNX**(平铺图无控制流,LSTM 状态 h/c 已暴露为
  输入;`download_models.py` 之外另需从 sherpa-onnx releases 下载
  `silero_vad.onnx` → `silero/model/silero_vad_v4_float.onnx`,接口见仓库主 README)。
- 不要直接喂量化版模型: `DynamicQuantizeLSTM` 等是 onnxruntime 私有算子,XNNC 不认;
  由 XNNC 按 float 权重 + 校准集重新量化(int8/int16)。
- 校准集生成(容器内,需 numpy/soundfile):
  ```bash
  cd /work/porting/silero
  python3 make_calibration.py /work/test_fixtures/smoke_test.wav dataset
  ```
  建议后续补充更多真实语音/噪声素材(每个 wav 一个 seq,窗间状态传递)。

## colima 注意事项(踩坑记录)

- **VM 必须经 `scripts/start-env.sh`(内部 limactl)启动**:`colima start`/`colima restart`
  会让已存在实例丢掉 rosetta 配置块,amd64 静默退回 QEMU(慢一个数量级);`colima stop` 安全。
- `colima stop` 会删掉 `colima` docker context,脚本已自动重建;limactl 启动后 docker
  服务偶发不自启,脚本有兜底。
- Docker Hub 直连不通的环境:脚本自动往 VM 写镜像加速(1ms.run / daocloud)。
