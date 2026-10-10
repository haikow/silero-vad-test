#!/usr/bin/env python3
"""HIL 自环测试结果解析: 串口日志 -> 逐窗概率 -> 与 PC 基准比对
用法: python3 hil_parse.py <串口日志文件> [golden_prob.f32]
验收: 最大偏差 <= 0.1(HIL 标准)

2026-10-10 v2(HH 包起):
- 首选 [SVs]7-done p=NNNN 行(每窗一条, ×10^4 整数, 1:1 对齐 golden);
  按 [SHIL] begin 分轮, 取最长一轮
- 回退 [SHIL] win=k p= 行(注意: 320样本帧喂入时只在每 2560 样本对齐打印,
  [SHIL] win=k 实际是 golden 窗 5k+4 的概率, 仅采样 8/40 窗)
"""
import re, sys, struct

log = open(sys.argv[1], errors="replace").read()
golden_path = sys.argv[2] if len(sys.argv) > 2 else "golden_prob.f32"

# ---- 方法1: [SVs]7-done p=NNNN (全窗, 首选) ----
# 按 [SHIL] begin 分轮; begin 缺失(mid-run 抓取)时用稀疏点相位锚定:
# [SHIL] win=k 打印前最近的 [SVs]7 = golden[5k+4]
lines = log.splitlines()
runs, cur, phase = [], [], []
last_svs = -1
for line in lines:
    m = re.search(r"\[SVs\]7-done p=(\d+)", line)
    if m:
        cur.append(int(m.group(1)) / 10000.0)
        last_svs = len(cur) - 1
        continue
    if "[SHIL] begin" in line:
        if cur: runs.append((cur, 0))
        cur, phase = [], []
        continue
    m = re.search(r"\[SHIL\] win=(\d+) p=(\d+\.\d+)", line)
    if m and cur:
        phase.append((last_svs, int(m.group(1))))
if cur: runs.append((cur, phase))

# 取最长轮 + 求相位
best = max(runs, key=lambda r: len(r[0])) if runs else ([], [])
probs, ph = best
if len(probs) >= 10:
    off = 0
    if ph:
        idx, k = ph[0]                 # 稀疏点前最近 SVs 序号 = golden[5k+4]
        if idx >= 0:
            off = (5 * k + 4) - idx
            if off < 0: off = 0
    end = min(len(probs), 40 - off)
    g = struct.unpack(f"<{end}f", open(golden_path, "rb").read()[off*4:(off+end)*4])
    maxd = max(abs(a-b) for a, b in zip(probs[:end], g))
    print(f"[SVs]7-done 模式: {len(runs)} 轮, 最长 {len(probs)} 窗, 相位偏移={off}(golden[{off}]起)" if off else
          f"[SVs]7-done 模式: {len(runs)} 轮, 最长 {len(probs)} 窗, 从 golden[0]")
    print(f"最大偏差 {maxd:.4f}")
    for i in range(end):
        a, b = probs[i], g[i]
        flag = "✗" if abs(a-b) > 0.1 else " "
        print(f"  win={i+off:2d} 板={a:.4f} PC={b:.4f} d={a-b:+.4f} {flag}")
    print("PASS (<=0.1)" if maxd <= 0.1 else "FAIL (>0.1)")
    sys.exit(0 if maxd <= 0.1 else 1)

# ---- 方法2: [SHIL] win=k (回退, 采样 8/40 窗, 映射 golden[5k+4]) ----
probs = [(int(m.group(1)), float(m.group(2)))
         for m in re.finditer(r"\[SHIL\] win=(\d+) p=(\d+\.\d+)", log)]
if not probs:
    sys.exit("未找到 [SVs]7-done 或 [SHIL] 输出")
gall = open(golden_path, "rb").read()
maxd, rows = 0.0, []
for k, p in probs:
    gi = 5 * k + 4          # 320样本帧喂入: win=k 打印点=第 5(k+1) 窗完成时
    g = struct.unpack("<f", gall[gi*4:gi*4+4])[0] if gi*4+4 <= len(gall) else None
    if g is None: continue
    d = abs(p - g); maxd = max(maxd, d)
    rows.append((k, gi, p, g, d))
print(f"[SHIL] 模式(采样窗): {len(rows)} 点 vs PC 基准(golden[5k+4])")
for k, gi, p, g, d in rows:
    print(f"  win={k} -> golden[{gi}] 板={p:.4f} PC={g:.4f} d={p-g:+.4f} {'✗' if d>0.1 else ' '}")
print(f"最大偏差 {maxd:.4f}")
print("PASS (<=0.1)" if maxd <= 0.1 else "FAIL (>0.1)")
sys.exit(0 if maxd <= 0.1 else 1)
