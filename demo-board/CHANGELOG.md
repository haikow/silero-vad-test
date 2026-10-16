# 固件包变更史（demo 板线）

> 包位置: 本地 `~/Desktop/烧录工具箱/`。md5 用于核对拿到的是不是同一版。

| 包 | md5(前8) | 内容 | 验证状态 |
|---|---|---|---|
| CCC | 95c1000d | 原厂分区零改动 + Silero int8 集成（首版 demo 板包） | [SHIL] 0/40 翻转过；真麦跑挂（=DFS 降频, 引出 CCCv2） |
| CCCv2 | dad936e3 | + init 时 `dfs_request(192M)` 持有（修复 #1 池溢出） | 真麦实时 p=0.88，但 `hit:0` 录音不触发（引出 CCCv3） |
| CCCv3 | af3ce691 | + 返回值契约修复（逐帧回报最近窗判定） | **勿用**：HIL×开机自启 VAD 竞态 → m_ring 断言死机循环（引出 CCCv4） |
| CCCv4 | 6be9653f | 停用 HIL 钩子（修复 #2 开机死机） | **触发链路实测打通**: hit:1 → 本地录音启动、文件创建；新问题: 录音 16s WDT2（算力超支） |
| **CCCv5** | **3d2dc902** | **录音态降载（命中后每 3 窗推理 1 窗）+ [SVs] 打印节流** | ✅ **板验通过(2026-10-16, 09:44 日志)**: 14.7min 零崩溃/WDT2; 52 候选窗→17 hit→7 次完整录音启停(最长 168s), `hit:1`→RECORDING→干净退出; [SVs] 0.36 行/s(原 31); 异常行仅 7×release repeated error+7×destory fail(转场良性) |
| **CCCv6** | **90ce03ae** | **降载常驻不退出**（v5 的"2s 静音退出"在句间停顿>2s 时中途回全速→录音期 run too long 44ms×146 次贴地运行）+ REC_SKIP 3→6（Silero 均摊 3.2ms/帧, 预算余量 45%） | **当前版本, 待板验**（验收: 录音期 run too long 显著减少且无连续 44ms 段; 触发/录音行为不回退） |

## 对应代码里程碑（本分支 overlay 的演进）

- 2026-10-14: 集成方式定为"直接代码拷贝"（非 patch），defconfig 仅加 `CONFIG_VAD_ENABLE=y`/`RING_ALLOCATE_CFG=2`，分区原厂原样
- 2026-10-15 上午: DFS 192M 持有（`sv_dfs_hold`）
- 2026-10-15 下午: 返回值契约修复——上层 `sw_vad.c` 迟滞按**连续调用**计数（continue_hit_cnt=10），
  库必须**逐帧**回报最近完成窗的命中态（此前仅窗完成帧返回 1，连续 1 最多 2 次，数学上不可能确认）
- 2026-10-15 晚: entry.c HIL 停用（#2 死机）
- 2026-10-16: 录音态降载（`rec_phase`/`REC_SKIP=3`/2s 静音退出）+ [SVs] 打印节流（状态变化+每 32 窗）

## 上报厂商待办

若 CCCv5 板验仍复现长录音 WDT2，向吴礼云/张福霖提交:
`DCORE WDT2, PC=0x262291ee(opus compute_mdcts), VAD-hit 本地 opus 录音 ~16s`+
完整日志与 crash dump，问题: 原厂 GVAD 触发本地 opus 录音路径是否验证过、
dump 解码工具能否给出调用栈。
