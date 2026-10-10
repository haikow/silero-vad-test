#!/usr/bin/env python3
"""gg_compare.py —— 板端 [GG] 日志 vs PC 金标对照
用法: python3 gg_compare.py <板端log文件>
从日志提取 [GG]x=.. stft=.. .. .. mag=.. feat=.. 行, 与 gg_golden_stft.npy + golden_x.f32 对照
"""
import sys, re
import numpy as np

LOG = sys.argv[1]
GOLD = "gg_golden_stft.npy"   # [N][3]: stft00, stft0_129, stft6_129 (x1e5)
BASE = "/Users/a1234/.zcode/workspace/default/wq7036-silero-vad/firmware-silero"

g = np.load(f"{BASE}/{GOLD}")
X = np.fromfile(f"{BASE}/golden_x.f32", np.float32).reshape(-1, 512)

pat = re.compile(r"\[GG\]x=(-?\d+) stft=(-?\d+) (-?\d+) (-?\d+) mag=(-?\d+) feat=(-?\d+)")
rows = []
for line in open(LOG, errors="ignore"):
    mm = pat.search(line)
    if mm:
        rows.append([int(v) for v in mm.groups()])
if not rows:
    print("未找到 [GG] 行"); sys.exit(1)

print(f"板端 {len(rows)} 窗, PC 金标 {len(g)} 窗")
print("win | x0板/PC     | stft00板/PC       | d_stft00 | stft129板/PC | mag00板/PC        | 判定")
bad = 0
for i, r in enumerate(rows):
    if i >= len(g):
        break
    x0, s00, s129, s6129, mag, feat = r
    g00, g129, g6129 = g[i]
    gx0 = int(X[i][0] * 100000)
    d00 = s00 - g00
    ok = (d00 == 0 and s129 == 0 and mag == abs(g00))
    if not ok: bad += 1
    mark = "✓" if ok else "✗"
    print(f"{i:3d} | {x0:7d}/{gx0:<7d} | {s00:9d}/{g00:<9d} | {d00:9d} | {s129:4d}/{g129:<4d} | {mag:9d}/{abs(g00):<9d} | {mark}")
print(f"\n不匹配窗数: {bad}/{min(len(rows),len(g))}")
print("结论: stft00 全对=conv1d正确,偏差来自后级; stft00 错=conv1d非零输入有bug")
