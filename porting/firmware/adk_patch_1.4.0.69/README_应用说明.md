# Silero VAD → WQ7036AC ADK 1.4.0.69 补丁包(厂商 patch 交付版)

> 2026-10-08。四个补丁 + 一条挪库指令 = Silero VAD(glass-stereo-dac/7036AC 正确基线)完整集成。
> 在 wq-audio_1.4.0.69 原始解包树上应用(与打补丁顺序无关, 但建议按编号)。

## 应用步骤

```bash
# 在 wq-audio 源码根目录(含 wq-adk/ 与 wqcore/ 的那一层)执行:
patch -p0 < 0001-silero-vad-source.patch        # Silero 源码 9 文件(算法+权重+5函数封装+HIL)
patch -p0 < 0002-rom-symbols-strip-recipsf2.patch # ROM 符号表剥离 __recipsf2(必打, 见下)
patch -p0 < 0003-defconfig-flash-layout.patch    # 新 defconfig.silero + dcore 分区 300→400 扇区
patch -p0 < 0004-hil-selftest-hook.patch         # [SHIL] 自检钩子(测试用, 量产可不打)

# 挪出原厂 WebRTC VAD 闭源库(等价禁用; 移回即回退):
mv wq-adk/components/audio_algorithm/lib/wq_sw_vad \
   wq-adk/components/audio_algorithm/lib_disabled_wq_sw_vad
```

## 构建

```bash
cd wq-adk/examples/glass
scons --defconfig=config/7036AC/defconfig.silero && scons -j16
# 产物: build/7036AC/glass-silero/*.wpk
```

前置: xt-clang(license 环境) + **RISC-V GCC 14.2**(1.4 必须, 10.2 会在 riscv_tools.py 失败;
wq-toolchain-dl GitLab 有) + dcore 镜像约 1.2MB(400 扇区分区)。

## 各补丁说明

| 补丁 | 内容 | 为什么 |
|---|---|---|
| 0001 | processor/src|inc 新增 9 文件 | Silero v4 纯 C99 float 实现; 原厂 sw_vad.c 的 5 函数接口(drop-in), processor 层零改动 |
| 0002 | rom_syms/{1.0,2.0}/xt/rom_image.ld 删 `__recipsf2` 行 | **必打**。xt-clang 对 float 除法发射 __recipsf2; 该 ROM 地址(0x2615f940)从未被任何官方固件验证过, 链上去会零日志死机。剥离后 libgcc 自带实现(ieee754divs.o)自动顶上——不要自己写 shim 返回 1.0f/x(会拉 libgcc 除法撞 multiple definition) |
| 0003 | defconfig.silero(新) + flash_layout.json | 总开关是 CONFIG_VAD_ENABLE(自动 select AUDIO_VAD_ENABLE/RECORDSV/RECORD_BASE), 附 RING_ALLOCATE_CFG=2(aud_sv_vad.c 硬性要求); 权重 601KB 入像需扩 dcore 分区 |
| 0004 | entry.c 挂 wq_silero_hil_selftest | 上电 [SHIL] begin→40 窗概率→end, 配 hil_parse.py 数值验收(≤0.1); CONFIG_AUDIO_VAD_ENABLE 门控, 量产可不打或移除 |

## 验证清单

1. 链接产物符号: `xt-nm glass_dcore.elf | grep sv_process`(应 25 个)、`grep WebRtc` 只剩
   WebRtcSpl_Rand*/Aec 曲线(公共工具, 非 VAD 本体)
2. 镜像段链: `python3 validate_dcore.py <wpk里的glass_dcore.bin>` 应 PASS
3. 上板: 三核正常 + `[SHIL]` 40 窗 → 日志发回跑 hil_parse.py(≤0.1 验收)

## 注意

- 基线必须是 glass-stereo-dac 同代(ADK 1.4.x); 若在 ai.recorder/1.3.0.398 上打此补丁
  **必然 rpc 断言循环**(分区/ro_cfg/核间不兼容, 2026-10 实证)
- ro_cfg.bin 用贵司目标板原厂的(随包的 ro_cfg 与板级硬件绑定)
- 已知无需处理: 权重文件 SV_WSEC 段属性已置空(AC 布局 XIP rodata 空间足够);
  is_defined→defined 已改; -Wmissing-prototypes 原型已加
