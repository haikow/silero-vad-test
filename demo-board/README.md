# 厂商 Demo 板线（ChuangQi DemoBroad / AI_Record_BT_base）

> 本分支 = **厂商 demo 板固件线**。`main` 分支 = 1.4.0.69 老基线（自家 glass-stereo-dac 台架）开发线。
> 两条线的 Silero 核心代码已按各自平台分叉（demo 板含 DFS 持有 + 录音态降载），不要互相覆盖。

## 这条线是什么

在方案商的 **AI_Record_BT_base（feature-ChuangQi_DemoBroad）** 工程上，把原厂预编译库
`lib/wq_sw_vad.a`（轻量能量 VAD）替换为我们的 **Silero v4 int8 源码实现**，实现
"说话 → TVAD 硬件触发 → Silero 确认 → 自动录音"。目标板 WQ7036AC demo 板
（stereo.dac.spi_nand 配置，ADK 1.4.0.69 同代）。

## 快速上手（三种角色）

**只想编译出固件包** → 看 [docs/编译指南.md](docs/编译指南.md)，从解压到 wpk 共 5 条命令。

**想了解改了厂商哪些文件** → `overlay/` 目录即全部差异（保持厂商 zip 内相对路径），
[setup.sh](setup.sh) 一键还原出可编译的完整工程树：

```bash
./setup.sh /path/to/AI_Record_BT_base-feature-ChuangQi_DemoBroad.zip ~/demo_work
# 产物 ~/demo_work/wq-audio/wq-adk/examples/glass 即编译入口
```

**想看每个固件包改了什么、哪版能用** → [CHANGELOG.md](CHANGELOG.md)（CCC→CCCv5，含 md5 与验证状态）。

**想看后续怎么走（稳定性/功耗实测/零功耗优化）** → [docs/优化路线图.md](docs/优化路线图.md)。

**A/B 实测后定了新方向（原厂基础优化）** → [docs/原厂基础上的优化方向.md](docs/原厂基础上的优化方向.md)（2026-10-20：两版差距不大 → 场景 profile/复核窗/TVAD 门限/回归集 + 踩坑约束）。

## 改动清单（overlay = 对厂商 zip 的全部差异）

| 类型 | 文件 | 内容 |
|---|---|---|
| 改 | `audio_algorithm/SConscript` | 加 processor/inc 与 nnlib 头文件搜索路径 |
| 改 | `audio_service/vad/acore/aud_sv_vad.c` | 原厂代码 -Werror=jump-misses-init 内联修复 |
| 改 | `entry/dcore/entry.c` | HIL 自测钩子（现 `#if 0` 停用，见踩坑速查 #2） |
| 改 | `examples/glass/config/7036AC/defconfig.record` | 加 `CONFIG_VAD_ENABLE=y` + `CONFIG_RING_ALLOCATE_CFG=2` |
| 禁用 | `audio_algorithm/lib/wq_sw_vad` | 原厂预编译 VAD 库改名 `lib_disabled_wq_sw_vad`（不删，可回退） |
| 新增 | `processor/{inc,src}/` 共 50+ 文件 | Silero v4 int8 核心 + 权重 + NNLib 头 + HIL + 接口层 |

**defconfig 其它选项零改动**（分区表原厂原样；两段式 VAD 唤醒框架、opus 录音均为原厂原生）。

## 交付物状态（2026-10-16）

- ✅ Silero 数值验收（40 窗 0/40 判定翻转，与 PC 金标一致）
- ✅ 真麦检测：说话 p≥0.88 实时响应，DFS 192M 持有
- ✅ 触发链路打通：`hit:1` → 本地录音自动启动（CCCv4 实测）
- 🔶 长录音稳定性：录音态降载修复（CCCv5）待板验——见 CHANGELOG 与踩坑速查 #3

## 深水区文档（在 main 分支）

移植史/方法论/全部坑：`git switch main` 后看 `porting/firmware/`（快速上手 / 移植指南 /
踩坑总结 / 厂商交接指南）。本分支 docs 只保留 demo 板日常需要的部分。
