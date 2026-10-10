# Silero VAD → WQ7036AC ADK 1.4.0.69 补丁包

> **版本 v3.4(2026-10-11)** · 5 个补丁 + 1 条挪库指令 = 完整集成
> **真机验收状态**: 40 窗逐窗概率与 PC 基准(onnxruntime)最大偏差 **≤0.0001**(验收线 0.1),
> 单窗 96.7ms, HIL 自检 10.2s 跑满 40 窗, 无 WDT 复位。
> 在 wq-audio_1.4.0.69 解包树(或贵司同代产品工程)上按编号顺序应用。

## 1. 应用步骤

```bash
# 在源码根目录(含 wq-adk/ 与 wqcore/ 的那一层)执行:
patch -p0 < 0001-silero-vad-source.patch         # Silero 全部源码/权重 + NNLib 头文件
patch -p0 < 0002-rom-symbols-strip.patch         # ROM 符号表剥离 3 个危险符号(必打)
patch -p0 < 0003-defconfig-flash-layout.patch    # defconfig.silero + dcore 分区 300→400 扇区
patch -p0 < 0004-hil-selftest-hook.patch         # [SHIL] 上电自检钩子(量产版可不打)
patch -p0 < 0005-nnlib-include-paths.patch       # SConscript 加 NNLib 头文件路径

# 挪出原厂 WebRTC VAD 闭源库(等价禁用; 移回即回退):
mv wq-adk/components/audio_algorithm/lib/wq_sw_vad \
   wq-adk/components/audio_algorithm/lib_disabled_wq_sw_vad
```

## 2. 构建

```bash
cd wq-adk/examples/glass
scons --defconfig=config/7036AC/defconfig.silero && scons -j16
```

前置: xt-clang(license 环境) + RISC-V GCC 14.2。
**NNLib 不需要内核源码**——SDK 自带同核预编译 `lib/xa_nnlib/libxa_nnlib.a`,
所需的 matmul/matXvec f32 符号全齐, 补丁只接头文件(0001)。

## 3. 各补丁说明

| 补丁 | 内容 | 要点 |
|---|---|---|
| 0001 | Silero 源码 9 文件(算法/权重/5 函数封装/HIL/音频头 + 自编激活源) + NNLib 头文件 47 个 | 算法实现见 §6 性能基线; HIL 输出必须用 **DBGLOG**(printf 在 1.4 是空桩) |
| 0002 | rom_image.ld 剥离 `__recipsf2`/`vec_tanhf`/`vec_reluf` | **必打**。这些 ROM 地址未被官方固件验证(实测 vec_tanhf 坏 SP 崩溃); 剥离后 libgcc/自编源自动顶上 |
| 0003 | defconfig.silero + flash_layout dcore 300→400 扇区 | `CONFIG_VAD_ENABLE`(总开关) + `RING_ALLOCATE_CFG=2`(aud_sv_vad.c 硬性要求) |
| 0004 | entry.c 挂 HIL 钩子 | 必须在 `app_main_entry` **之前**(它启动调度后不返回); 首个合入版建议打上用于对拍, 量产去掉 |
| 0005 | SConscript 加 `-I processor/inc` 与 `-I processor/inc/nnlib` | NNLib 头内部互引需要两个路径 |

## 4. 构建结果验证

1. `xt-nm glass_dcore.elf | grep sv_process`(≈26 个符号)、`grep xa_nn_mat`(matmul/matXvec 已链接)
2. `python3 integration/validate_dcore.py <glass_dcore.bin>` → 段链 PASS
3. 上板: 三核正常启动, 无 WDT 复位; 若打了 0004, 上电即见 `[SHIL] begin windows=40`
4. 串口日志跑 `python3 integration/hil_parse.py <日志> golden_prob.f32`
   → **40 窗全部 ≤0.1 为 PASS**(我方实测 0.0001, 贵司合入版应得到同量级结果)

## 5. 性能基线(v3.4 真机实测)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| STFT | 9.7ms | 加窗 256 点基 2 FFT(纯 C); 不依赖任何 FFT 库 |
| first_layer | ~30ms | matmul SIMD + dwconv 标量; 权重 flash 流量为主 |
| encoder | ~15ms | 同上 |
| LSTM×2 | ~36ms | matXvec SIMD; R 已暂存 TCM, W 留 flash(128KB≈23ms)为主要残余 |
| **全窗** | **~97ms** | 40 窗 HIL 全程 10.2s(含 100ms 帧间节拍), 不触发 WDT |

**真麦实时线 32ms/窗的最后一公里**: 把 LSTM W(128KB) + first/enc 权重(75KB) 暂存进
TCM 即可到 **~25-30ms**, 但当前 TCM 运行时余量仅 ~69KB(preset 占用后)——
需要贵司协助 **preset 内存重配释放 ~140KB TCM**(如 preset 块挪 PSRAM/缩容)。
我方已备齐全部数据(布局定案/内存预算表/逐阶段实测), 沟通即可实施。

## 6. 接口契约(厂商自查清单, 不需要我方源码)

我方交付的 `processor/inc/wq_sw_vad.h` 与贵司原厂 `lib/wq_sw_vad/inc/wq_sw_vad.h`
**逐字一致**(已 diff 验证), 上层调用点零改动:

```c
int   wq_get_sw_vad_hd_size(void);                    /* 句柄字节数(几十字节) */
int   wq_get_sw_vad_scratch_size(void);               /* 返回 0(大缓冲为静态区) */
void *wq_sw_vad_init(void *vad_hd, void *pscratch);   /* 返回 vad_hd */
int   wq_sw_vad_process(void *vad_hd, char *in_data, unsigned int in_len);
      /* in_data = 单通道 int16 PCM, in_len = 字节数(320 样本 = 640 字节/帧);
         内部攒满 512 样本推理一次(Silero v4 窗长);
         返回 1 = 所在窗概率≥阈值(命中), 0 = 未命中 —— 与原厂语义一致 */
void  wq_get_sw_vad_lib_version(char *version_string);
```

- **迟滞层零改动**: 贵司 `processor/src/sw_vad.c` 用返回值做 continue_hit/stop_cnt
  迟滞, 语义与原厂完全一致
- **阈值可调**: `processor/inc/wq_silero_vad.h` 的 `WQ_SILERO_THRESHOLD`(默认 0.5);
  逐窗概率有 `[SVs]7-done p=` 日志可观测(量产可关)
- **include**: 挪库后 `wq_sw_vad.h` 由 `processor/inc` 提供(0005 保证)

## 7. 贵司工程与本补丁基线(1.4.0.69)不同时

1. **0001 纯新增文件**(56 个)任何树都能打; 0002-0005 每个仅几行, patch 打不上时
   按 reject 手工对齐即可
2. **贵司产品 defconfig 必须包含**(已验证组合):
   ```
   CONFIG_VAD_ENABLE=y              # VAD 总开关(自动 select AUDIO_VAD_ENABLE 等)
   CONFIG_RING_ALLOCATE_CFG=2       # 或 3, aud_sv_vad.c 硬性要求
   CONFIG_XT_TOOLCHAIN_XT_CLANG=y   # dcore 必须 xt-clang(GCC 软浮点跑不动)
   CONFIG_RISCV_TOOLCHAIN_GCC_VERSION_14_2_0=y
   ```
3. **flash_layout**: dcore 分区 ≥400 扇区(1.6MB; dcore 镜像 ~1.27MB, 300 扇区不够)
4. **回传物**: 编好的 wpk + 触发录音的方式(测试命令/模式)——真麦验证需要贵司产品
   工程的麦克风/录音通路, 我方验证固件仅含 HIL golden 喂数
5. **禁止项**: 基线必须 glass-stereo-dac 同代(ADK 1.4.x), 打在 ai.recorder/1.3 上
   必然 rpc 断言循环; ro_cfg.bin 用目标板原厂的

## 附: 版本记录

| 版本 | 日期 | 要点 |
|---|---|---|
| v1 | 09-30 | 标量版初稿 |
| v2 | 10-09 | NNLib 版 + HIL 钩子位置修正 + DBGLOG(有 conv1d 布局/漏 ReLU 两个致错 bug, 勿用) |
| v3 | 10-11 | 数值闭环版(40 窗 ≤0.0001): conv1d 参数重排 / 补 4 处中间 ReLU / 音频 ×32768 |
| **v3.4** | **10-11** | **性能版(96.7ms)**: FFT 替代 STFT 卷积 / LSTM matXvec+R 暂存 / 1x1 matmul / HIL 喂狗 |
