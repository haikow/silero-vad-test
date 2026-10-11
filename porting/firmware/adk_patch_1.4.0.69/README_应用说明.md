# Silero VAD → WQ7036AC ADK 1.4.0.69 补丁包

> **版本 v3.6(int8 量化最终交付版, 2026-10-11)** · 5 个补丁 + 1 条挪库指令 = 完整集成
> **真机验收状态**(40 窗 HIL 自检, golden 音频): **判定翻转 0/40**(阈值 0.5 口径),
> 逐窗概率与 PC 基准最大偏差 0.0438(验收线 0.1), **单窗 31ms(真麦实时线 32ms 达成)**,
> HIL 全程 7.7s, 无 WDT 复位; wpk 1.67MB(dcore 权重 609KB→86KB)。
> 在 wq-audio_1.4.0.69 解包树(或贵司同代产品工程)上按编号顺序应用。

## 1. 应用步骤

```bash
# 在源码根目录(含 wq-adk/ 与 wqcore/ 的那一层)执行:
patch -p0 < 0001-silero-vad-source.patch         # Silero 全部源码/权重(int8) + NNLib 头文件
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

前置: xt-clang(license 环境, **Docker 必须 `--mac-address 00:15:5d:2b:42:e6`**, 漏掉的症状
是链接阶段 FlexNet -5,147) + RISC-V GCC 14.2。
**NNLib 不需要内核源码**——SDK 自带同核预编译 `lib/xa_nnlib/libxa_nnlib.a`, 所需符号
(`xa_nn_matXvec_8x8_32`/`xa_nn_matmul_f32xf32_f32` 等)全齐, 补丁只接头文件(0001)。

## 3. 各补丁说明

| 补丁 | 内容 | 要点 |
|---|---|---|
| 0001 | Silero 源码 10 文件(算法/float+int8 权重/5 函数封装/HIL/音频头 + 自编激活源) + NNLib 头文件 47 个 | 算法实现见 §5; HIL 输出必须用 **DBGLOG**(printf 在 1.4 是空桩) |
| 0002 | rom_image.ld 剥离 `__recipsf2`/`vec_tanhf`/`vec_reluf` | **必打**。这些 ROM 地址未被官方固件验证(实测 vec_tanhf 坏 SP 崩溃); 剥离后 libgcc/自编源自动顶上 |
| 0003 | defconfig.silero + flash_layout dcore 300→400 扇区 | `CONFIG_VAD_ENABLE`(总开关) + `RING_ALLOCATE_CFG=2`(aud_sv_vad.c 硬性要求) |
| 0004 | entry.c 挂 HIL 钩子 | 必须在 `app_main_entry` **之前**(它启动调度后不返回); 首个合入版建议打上用于对拍, 量产去掉 |
| 0005 | SConscript 加 `-I processor/inc` 与 `-I processor/inc/nnlib` | NNLib 头内部互引需要两个路径 |

## 4. 构建结果验证

1. `xt-nm glass_dcore.elf | grep sv_process`(≈26 个符号)、`grep matXvec_8x8_32`(int8 LSTM 内核已链接)
2. `python3 integration/validate_dcore.py <glass_dcore.bin>` → 段链 PASS
3. 上板: 三核正常启动, 无 WDT 复位; 若打了 0004, 上电即见 `[SHIL] begin windows=40`
4. 串口日志跑 `python3 integration/hil_parse.py <日志> golden_prob.f32`
   → **int8 验收口径: 判定翻转 0/40(阈值 0.5)且逐窗偏差 <0.25**(我方实测最大 0.0438;
   注意 int8 口径与 float 时代的逐窗 ≤0.0001 不同——量化预期如此, 判定层零翻转)

## 5. 性能基线(v3.6 int8 真机实测)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| STFT | 4.6ms | 双实数帧合批 FFT(4 次复 FFT 代 8 次, 纯 C) |
| 归一化 | 2.8ms | 快速 log/sqrt(误差有界 ~3.5e-6, `SV_NORM_FAST` 宏可回退 libm) |
| first_layer | ~10ms | f32 matmul + 权重已暂存 TCM(无 flash 流量) |
| encoder | ~3.5ms | 同上 |
| LSTM×2 | ~9.7ms | **int8**(matXvec_8x8_32, int32 出口 + per-row 反量化), 权重已暂存 TCM |
| **全窗** | **~31ms** | 40 窗 HIL 全程 7.7s, 不触发 WDT; **真麦实时线 32ms 已达成, 无需 preset 重配** |

**真麦耗时预估**: 纯算法 ~30ms 与数据无关; 量产时 VAD 与 AANC/空间音频/编码等同核
并发, 墙钟视产品功能组合可能为 30-40ms。音频池 240ms(`WQ_SW_VAD_SAVE_PKT_NUM=12`,
sw_vad.h)可吸收突发; 若真麦实测持续均值 >32ms, 对策=调大该宏(12→24 仅 +7.7KB)。
历史教训备查: .a 的量化内核(matXvec_8x16_32 等)为通用慢速版, **只适合每窗一次调用**
(LSTM 即此形态); 逐时间步场景(1x1 卷积)int8 化实测不可行(已回退, 见指南 §47)。

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
- **内存**: VAD 需 TCM 堆 ~140KB(启动时一次性暂存权重, 之后零 flash 流量)+ HIL 自检
  临时句柄; 已在 preset 占用后的运行时余量(197KB)内验证
- **include**: 挪库后 `wq_sw_vad.h` 由 `processor/inc` 提供(0005 保证)

## 7. 贵司工程与本补丁基线(1.4.0.69)不同时

1. **0001 纯新增文件**(57 个)任何树都能打; 0002-0005 每个仅几行, patch 打不上时
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
| v3 | 10-11 | float 数值闭环版(40 窗 ≤0.0001): conv1d 参数重排 / 补 4 处中间 ReLU / 音频 ×32768 |
| v3.4 | 10-11 | float 性能版(96.7ms): FFT 替代 STFT 卷积 / LSTM matXvec+R 暂存 / 1x1 matmul / HIL 喂狗 |
| v3.5 | 10-11 | int8 化: LSTM matXvec_8x8_32 + dw int8 + 1x1 float 暂存(basis 退役, wpk -357KB) |
| **v3.6** | **10-11** | **int8 定稿(31ms)**: 双帧合批 FFT / norm 快速近似(SV_NORM_FAST 可回退) / 门控 float 修复; 1x1 int8 尝试失败已回退(.a 量化内核慢速版, 见指南 §47) |
