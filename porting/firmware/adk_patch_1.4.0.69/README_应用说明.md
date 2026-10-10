# Silero VAD → WQ7036AC ADK 1.4.0.69 补丁包(厂商 patch 交付版 v2)

> 2026-10-11 v3(数值闭环终版): **II/KK 包实测 40 窗全部 |板端-PC|≤0.0001**(验收线 0.1),
> 全窗 ~160ms(其中 STFT 8ms/KK)。五个补丁 + 一条挪库指令 = 完整集成。
> 在 wq-audio_1.4.0.69 原始解包树上按编号顺序应用。
>
> **v3 相对 v2 的实质修正**(v2 有两个致错 bug, 不能用):
> 1. conv1d kernel 布局: NNLib 实际按 [OC][ICW_pad×KH] 读 kernel, v2 直传 [OC][KH]
>    导致 oc≥129 越界读权重区(GG 包结构性零通道实锤) → 参数重排 [OC][KH=64][ICW=4]
> 2. dwconv5 补内嵌 ReLU: ONNX 每块 dw→ReLU→pw, v2 漏了 → 静音窗 p 偏高 ~0.3
> 另: 1x1/残差/LSTM/decoder 全标量化(NNLib matmul 同款 HWC 布局坑)、sv_basis() TCM
> 暂存 STFT kernel、HIL 音频 ×32768 无损量化、0002 增剥 vec_tanhf/vec_reluf

## 应用步骤

```bash
# 在 wq-audio 源码根目录(含 wq-adk/ 与 wqcore/ 的那一层)执行:
patch -p0 < 0001-silero-vad-source.patch         # Silero 源码+NNLib头(算法/权重/5函数封装/HIL/NNLib全套头)
patch -p0 < 0002-rom-symbols-strip.patch # ROM 符号表剥离 __recipsf2(必打)
patch -p0 < 0003-defconfig-flash-layout.patch     # defconfig.silero + dcore 分区 300→400 扇区
patch -p0 < 0004-hil-selftest-hook.patch          # [SHIL] 自检钩子(量产可不打)
patch -p0 < 0005-nnlib-include-paths.patch        # SConscript 加 NNLib 头文件 -I 路径

# 挪出原厂 WebRTC VAD 闭源库(等价禁用; 移回即回退):
mv wq-adk/components/audio_algorithm/lib/wq_sw_vad \
   wq-adk/components/audio_algorithm/lib_disabled_wq_sw_vad
```

## 构建

```bash
cd wq-adk/examples/glass
scons --defconfig=config/7036AC/defconfig.silero && scons -j16
```

前置: xt-clang(license 环境) + **RISC-V GCC 14.2**(wq-toolchain-dl GitLab 有)。
**NNLib 内核不需要源码**——SDK 自带同核预编译 `lib/xa_nnlib/libxa_nnlib.a`, 所需
9 个 f32 符号全齐(conv1d_std/matmul/matXvec/dot_prod/激活/elm_add), 补丁只接头文件。

## 各补丁说明

| 补丁 | 内容 | 要点 |
|---|---|---|
| 0001 | Silero 源码 9 文件(算法/权重/5函数封装/HIL/音频头+自编激活源) + **NNLib 全套头文件**(47 个) | silero_vad.c **v3.4**: STFT=加窗 256 点基2 FFT(basis[0]=窗/行j=cos/行129+j=-sin, **无 conv1d**); LSTM 门=matXvec SIMD 一次调用 + R 暂存 TCM(W 留 flash, 带回退); 1x1 卷积=matmul SIMD(输入[T][IC]/输出[T][OC] 前后转置); dwconv5 内嵌 ReLU; 残差/decoder 标量; HIL 输出用 **DBGLOG**(printf 在 1.4 是空桩!), PACE 100ms + 显式喂狗 |
| 0002 | rom_image.ld 删 `__recipsf2`/`vec_tanhf`/`vec_reluf`(1.0 前者, 2.0 三者) | **必打**。ROM 地址从未被官方固件验证(实测 vec_tanhf 坏 SP 崩溃); 剥离后 libgcc/自编源自动顶上(不要写 shim 返回 1.0f/x, 会撞 multiple definition) |
| 0003 | defconfig.silero + flash_layout dcore 300→400 | 总开关 CONFIG_VAD_ENABLE(自动 select AUDIO_VAD_ENABLE 等)+ RING_ALLOCATE_CFG=2(aud_sv_vad.c 硬性要求) |
| 0004 | entry.c 挂 HIL 钩子 | **必须在 app_main_entry 之前**(它启动调度后不返回); 上电 [SHIL] 40 窗概率输出 |
| 0005 | SConscript 加 `-I processor/inc` 和 `-I processor/inc/nnlib` | NNLib 头的内部互引需要两个路径 |

## 验证清单

1. `xt-nm glass_dcore.elf | grep sv_process`(≈26 个)、`grep xa_nn_conv1d`(NNLib conv1d 链接)
2. `python3 integration/validate_dcore.py <glass_dcore.bin>` 段链 PASS
3. 上板: 三核正常 + 无 WDT2 崩溃 + `[SVs]7-done p=` 40 行
4. `python3 integration/hil_parse.py <日志> golden_prob.f32` → **40 窗全部 ≤0.1**(实测 0.0001)

## 性能基线(KK 实测, 供优化对照)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| STFT | **8ms** | kernel 264KB 已暂存 TCM 堆; 若回退 flash(data_xip) 为 400ms |
| first_layer | ~42ms | 标量 pw1x1(已 k 外提); flash 读 50KB |
| encoder | ~23ms | 同上 |
| LSTM×2 | ~80ms | **下一个优化点**: W/R 256KB flash 流读; 可 TCM 暂存(堆余量不足, 需取舍) |
| 全窗 | ~160ms | 40 窗 HIL 全程 12.8s(含 100ms 帧间 pacing), 不触发 WDT2 |

## 注意

- 基线必须 glass-stereo-dac 同代(ADK 1.4.x); 打在 ai.recorder/1.3 上必然 rpc 断言循环
- ro_cfg.bin 用目标板原厂的
- v1→v2 变更: 标量版→NNLib 版 / HIL 钩子挪到 app_main_entry 前 / printf→DBGLOG /
  新增 0005(SConscript include) / 0001 含 NNLib 头文件

## 接口契约(厂商自查清单 —— 不需要我们提供源码即可对接)

我们交付的 `processor/inc/wq_sw_vad.h` 与贵司原厂 `lib/wq_sw_vad/inc/wq_sw_vad.h`
**逐字一致**(已 diff 验证), 上层调用点零改动。五函数契约:

```c
int   wq_get_sw_vad_hd_size(void);                    /* 句柄字节数(小, 几十字节) */
int   wq_get_sw_vad_scratch_size(void);               /* 返回 0(大缓冲为静态区) */
void *wq_sw_vad_init(void *vad_hd, void *pscratch);   /* 返回 vad_hd */
int   wq_sw_vad_process(void *vad_hd, char *in_data, unsigned int in_len);
      /* in_data=单通道 int16 PCM, in_len=字节数(320样本=640字节/帧);
         内部攒满 512 样本推理一次(Silero v4 窗长);
         返回 1=所在窗概率≥WQ_SILERO_THRESHOLD(命中), 0=未命中 */
void  wq_get_sw_vad_lib_version(char *version_string);
```

- **返回值语义**: 贵司 `processor/src/sw_vad.c` 把返回值当"本帧命中"做迟滞
  (continue_hit_cnt/continue_stop_cnt)——语义与原厂一致, 该层零改动
- **阈值调节**: `processor/inc/wq_silero_vad.h` 的 `WQ_SILERO_THRESHOLD`(默认 0.5);
  逐窗概率另有 `[SVs]7-done p=` 日志可观测(量产可关)
- **include**: 挪库后 `wq_sw_vad.h` 由 `processor/inc` 提供(0005 的 -I 保证)

## 贵司工程与本补丁基线(1.4.0.69)不同时

1. **0001 纯新增文件**(56 个), 任何树都能打; 0002/0003/0004/0005 每个仅几行,
   patch 打不上时按 reject 手工对齐即可
2. **贵司产品 defconfig 必须包含**(已验证组合):
   ```
   CONFIG_VAD_ENABLE=y              # VAD 总开关(自动 select AUDIO_VAD_ENABLE 等)
   CONFIG_RING_ALLOCATE_CFG=2       # 或 3, aud_sv_vad.c 硬性要求
   CONFIG_XT_TOOLCHAIN_XT_CLANG=y   # dcore 必须 xt-clang
   CONFIG_RISCV_TOOLCHAIN_GCC_VERSION_14_2_0=y
   ```
3. **flash_layout**: dcore 分区 ≥400 扇区(1.6MB; dcore 镜像 ~1.27MB, 300 扇区不够)
4. 构建产物请回传 wpk + **触发录音的方式**(测试命令/模式) —— 真麦验证需要
   贵司产品工程里的麦克风/录音通路, 我方验证固件仅含 HIL golden 喂数
5. 首个合入版建议**先打 0004**(上电自动跑 [SHIL] 40 窗自检, 与我方已验证的
   ≤0.0001 直接对拍, 确认合入无误); 量产版去掉 0004 即纯产品启动
