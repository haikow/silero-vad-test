# Silero VAD → WQ7036AC ADK 1.4.0.69 补丁包(厂商 patch 交付版 v2)

> 2026-10-09 v2(修正版): NNLib 加速版实现 + 钩子位置修正 + DBGLOG 输出。
> 五个补丁 + 一条挪库指令 = Silero VAD 完整集成(NNLib 内核加速, 预期单窗 <5ms)。
> 在 wq-audio_1.4.0.69 原始解包树上按编号顺序应用。

## 应用步骤

```bash
# 在 wq-audio 源码根目录(含 wq-adk/ 与 wqcore/ 的那一层)执行:
patch -p0 < 0001-silero-vad-source.patch         # Silero 源码+NNLib头(算法/权重/5函数封装/HIL/NNLib全套头)
patch -p0 < 0002-rom-symbols-strip-recipsf2.patch # ROM 符号表剥离 __recipsf2(必打)
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
| 0001 | Silero 源码 5 文件 + **NNLib 全套头文件**(nnlib/目录+NatureDSP/ndsp 头+state 头) | silero_vad.c 为 NNLib 映射版: STFT=conv1d_std_f32, 1x1卷积=matmul(CHW 路径), LSTM 门=matXvec, decoder=dot_prod, 激活=vec_sigmoid/tanh**_f32_f32**(32_32 是定点别用), dwconv5 保留标量; conv1d 必须给 bias(零数组)和 p_scratch; HIL 输出用 **DBGLOG**(printf 在 1.4 是空桩!), PACE 100ms 帧间 yield |
| 0002 | rom_image.ld 删 `__recipsf2` 行(1.0/2.0 两版) | **必打**。xt-clang 的 float 除法引用它; ROM 地址从未被官方固件验证, 剥离后 libgcc 自带实现自动顶上(不要写 shim 返回 1.0f/x, 会撞 multiple definition) |
| 0003 | defconfig.silero + flash_layout dcore 300→400 | 总开关 CONFIG_VAD_ENABLE(自动 select AUDIO_VAD_ENABLE 等)+ RING_ALLOCATE_CFG=2(aud_sv_vad.c 硬性要求) |
| 0004 | entry.c 挂 HIL 钩子 | **必须在 app_main_entry 之前**(它启动调度后不返回); 上电 [SHIL] 40 窗概率输出 |
| 0005 | SConscript 加 `-I processor/inc` 和 `-I processor/inc/nnlib` | NNLib 头的内部互引需要两个路径 |

## 验证清单

1. `xt-nm glass_dcore.elf | grep sv_process`(≈26 个)、`grep xa_nn`(NNLib 内核链接)
2. `python3 validate_dcore.py <glass_dcore.bin>` 段链 PASS
3. 上板: 三核正常 + 无 WDT2 崩溃 + `[SHIL]` 40 窗 → hil_parse.py ≤0.1 验收

## 注意

- 基线必须 glass-stereo-dac 同代(ADK 1.4.x); 打在 ai.recorder/1.3 上必然 rpc 断言循环
- ro_cfg.bin 用目标板原厂的
- v1→v2 变更: 标量版→NNLib 版 / HIL 钩子挪到 app_main_entry 前 / printf→DBGLOG /
  新增 0005(SConscript include) / 0001 含 NNLib 头文件
