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
# 按 [SHIL] begin 分轮(时间顺序扫描原始日志行)
runs, cur = [], []
for line in log.splitlines():
    if "[SHIL] begin" in line:
        if cur: runs.append(cur)
        cur = []
    m = re.search(r"\[SVs\]7-done p=(\d+)", line)
    if m:
        cur.append(int(m.group(1)) / 10000.0)
if cur: runs.append(cur)

best = max(runs, key=len) if runs else []
if len(best) >= 10:
    g = struct.unpack(f"<{len(best)}f", open(golden_path, "rb").read()[:len(best)*4])
    maxd = max(abs(a-b) for a, b in zip(best, g))
    print(f"[SVs]7-done 模式: {len(runs)} 轮, 取最长 {len(best)} 窗 vs PC 基准")
    print(f"最大偏差 {maxd:.4f}")
    for i, (a, b) in enumerate(zip(best, g)):
        flag = "✗" if abs(a-b) > 0.1 else " "
        print(f"  win={i:2d} 板={a:.4f} PC={b:.4f} d={a-b:+.4f} {flag}")
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
