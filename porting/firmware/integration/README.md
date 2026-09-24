# wq-adk 固件集成指南(Silero VAD 替换原厂 WebRTC VAD)

> 2026-09-22 实测:wq-audio SDK 1.3.0.398 + glass/7036AC 例子,容器内 GCC 构建全通过,
> 产出可烧录 wpk;镜像内 Silero 符号就位、WebRTC VAD 符号清零、processor 层零改动。
> 2026-09-23 补:同源码 **7036AX**(stereo.dac 基线)构建通过,wpk 已产出——见下文 7036AX 一节。

## SDK 树改动清单(相对原版 wq-audio 1.3.0.398)

1. **算法源码放置**(编译进 audio_alg 模块,SConscript 自动扫 processor/src/*.c):
   - `components/audio_algorithm/processor/src/` += `silero_vad.c` `silero_vad_weights.c` `wq_silero_vad.c` `wq_gcc_shim.c`(GCC 构建垫片,xt-clang 构建不需要)
   - `components/audio_algorithm/processor/inc/` += `silero_vad.h` `silero_vad_consts.h` `wq_silero_vad.h` `wq_sw_vad.h`(接口头,内容同原 lib 头)
2. **停用原厂预编译库**:`lib/wq_sw_vad` → `lib_disabled/wq_sw_vad`(移出 SConscript 扫描;还原即回退)
3. **defconfig**: `examples/glass/config/7036AC/defconfig.silero`(本目录 `config/7036AC/` 附)——基于 ai.recorder
   + `CONFIG_VAD_ENABLE=y` + `CONFIG_RING_ALLOCATE_CFG=2` + `CONFIG_PIPELINE_STREAM_MAX_NUM=20`
   + `CONFIG_XT_TOOLCHAIN_GCC=y`(免 license;生产用 xt-clang 时改回并去掉 shim)
4. **flash 布局**: `config/7036AC/flash_layout.json` dcore 分区 300 → 400 扇区
   (权重 623KB 进镜像后 dcore 1.43MB,原 300 扇区 1.17MB 不够)
5. **GCC12 豁免**(仅 GCC 构建需要): `wqcore/tools/SCons/xtensa_tools.py` 三处 -Werror 后追加
   `-Wno-error=stringop-overflow/maybe-uninitialized/array-bounds`(旧 SDK 代码的 GCC12 误报)

## 构建命令(容器内,工具链挂 /opt/wqcore)

```bash
# 工具链: RI-2020.4 + xtensa-wuqi-elf-gcc 12.2 + riscv64 10.2(wqcore/toolchain/)
docker run --platform linux/amd64 -v <sdk>:/work -v <wqcore>:/opt/wqcore \
  -e WQCORE_TOOLCHAIN_PATH=/opt/wqcore/toolchain \
  -w /work/wq-audio/wq-adk/examples/glass wq-build:latest bash -c \
  'scons --defconfig=config/7036AC/defconfig.silero && scons -j16'
# 产物: build/7036AC/silero/*.wpk(烧录包) + {acore,bcore,dcore}/*.elf
```

## 7036AX 构建(2026-09-23,板子实为 7036AX 后新增)

第一轮烧录失败根因即芯片型号:板子是 **7036AX**(bbb 系列),固件编成了 7036AC。
重编要点:

1. **defconfig 极简基线**:`config/7036AX/defconfig.silero`(本目录附)——直接基于
   原版 `defconfig.stereo.dac` 复制,只改 `CONFIG_XT_TOOLCHAIN_GCC=y` 一处
   (VAD 等开关 stereo.dac 原本就开)。**不要**照搬 7036AC 那份(裁剪项会引发依赖级联)。
2. **DRAM 溢出 84KB**:623KB 权重 + stereo.dac 代码 > 768KB DRAM。
   解法:权重数组链到 **`.icache.literal` 段**(1MB icache 区,基本空闲)——
   `silero_vad_weights.c` 里以 `SV_WSEC` 宏实现,**仅 `__XTENSA__` 目标生效**
   (桌面 clang/gcc 编译自动去掉属性,`make test` 不受影响,基准已复跑 PASS)。
   注意 SDK 内集成的副本用的是裸 attribute,仓库这份是带宏保护版,重新集成时二选一即可。
3. **flash 布局**:`config/7036AX/flash_layout.json` dcore 300 → 400 扇区(同 7036AC)。
4. **清理重编**:改段属性后必须 `rm -rf build/7036AX/silero` 全清,否则旧缓存把
   已停用的 WebRTC 符号链回来。
5. 构建命令同下,把 defconfig 路径换成 `config/7036AX/defconfig.silero` 即可。

产物:`glass-stereo-dac-7036AX-0.0.0.0.wpk`(2,221,352B,
MD5 c2b9db46d99aa8b5597a1730116969f7),测试包在 `~/Desktop/silero-vad-固件测试包/`,
也挂在 GitHub release。烧录排错与硬件接线见同目录 **[烧录硬件指南.md](烧录硬件指南.md)**。

## 关键体积/布局数据(实测)

| 核 | 镜像 | 分区占用 |
|---|---|---|
| dcore(含 Silero) | 1,429,528B | 87.25%(400 扇区) |
| bcore | 489,860B | 42.71% |
| acore | 599,208B | 73.15% |

## GCC vs xt-clang 注意

- GCC 12.2 为**软浮点**构建(不用 HiFi5 FPU):功能验证/联调可用;**实时性不足**,
  且 Kconfig 声明 GCC 不支持 HiFi 系指令——生产固件请用 xt-clang(license 环境),
  本集成对编译器透明(纯 C99),defconfig 改回 `CONFIG_XT_TOOLCHAIN_XT_CLANG=y` 即可。
- 链接 XCC 预编译库需 `wq_gcc_shim.c`(_Assert/__recipsf2/__ieee754_sqrtf 垫片);
  xt-clang 构建下不需要(其自带运行时),该文件可保留无害。

## 下一步(上板)

1. wpk 经 PCBA 烧录工具写入开发板(wq-debug-tools/Beetle);
2. VAD 事件经 app_customer_vad 的 BLE 协议(0x09FF)或 dump 通路验证;
3. HIL:UART 逐窗概率(wq_silero_last_prob)vs PC 基准,验收线 0.1。


## 给物奇固件团队的交接说明(xt-clang + license 生产版)

**你们环境里只需三步,源码零适配:**

1. **拷文件**(从本仓库 porting/firmware/ 取):
   - `silero_vad.c` `silero_vad_weights.c` `wq_silero_vad.c` → SDK `components/audio_algorithm/processor/src/`
   - `silero_vad.h` `silero_vad_consts.h` `wq_silero_vad.h` `wq_sw_vad.h` → `processor/inc/`
2. **挪库**:`lib/wq_sw_vad` 整目录移出 `lib/`(如 `lib_disabled/`;移回即回退原厂 VAD)
3. **配置**:在你们现有 defconfig 上加 `CONFIG_VAD_ENABLE=y`(**保持
   `CONFIG_XT_TOOLCHAIN_XT_CLANG=y` 不动**);`flash_layout.json` 的 dcore 分区
   300→400 扇区(权重 623KB 进镜像后需要);然后正常 `scons`。

**注意**:本仓库 integration/ 目录里的 `wq_gcc_shim.c` 和 xtensa_tools.py 的
-Wno-error 补丁**你们不需要**——那是我们无 license 的 GCC 联调环境专用;
你们的 defconfig 也不用换 `defconfig.silero`(GCC 版专用,现位于 `config/7036AC|AX/`),
在**自己的 defconfig 上加 VAD 开关即可**。验证:编出来的 dcore elf 里
`nm | grep sv_process` 有符号、`grep WebRtcVad` 为零即替换成功。

**性能**:xt-clang 会用 HiFi5 FPU 编我们的纯 C float 代码(网络仅 ~70 万 MAC/窗,
余量很大)。若将来要省功耗再考虑 int16 版(见 ../int16_wip/,目前 92% 场景一致率,
WIP 状态)。

## wpk 产物(本目录 wpk/)

| 文件 | 目标芯片 | MD5 | 说明 |
|---|---|---|---|
| glass-stereo-dac-7036AX-0.0.0.0.wpk | **7036AX**(当前目标板) | c2b9db46d99aa8b5597a1730116969f7 | stereo.dac 基线, GCC 联调版, 含 HIL 自检 |
| ai-recorder-7036AC-0.0.0.0.wpk | 7036AC | 65cac9c20ce957a808bde25591eb4732 | ai.recorder 基线, 同上 |

> 注:原计划挂 GitHub release,当前网络 uploads.github.com 不通,wpk 直接入私有库
> (一次性 ~4.3MB,可接受;后续高频迭代版建议本地留档、不重复入库)。

## 诊断双包(2026-09-24,WDT 崩溃对照实验,详见 ../移植指南.md §7)

| 文件 | 用途 | MD5 |
|---|---|---|
| ai-recorder-stub-7036AC-0.0.0.0.wpk | A包:推理置空,隔离计算耗时变量 | d444ce52aa9fe23d48fb58c4fff03bc7 |
| ai-recorder-paced-7036AC-0.0.0.0.wpk | B包:推理照常+每帧延时800ms,完成40窗数值验收 | f1c1b959f46aa08aa68f030981860899 |
