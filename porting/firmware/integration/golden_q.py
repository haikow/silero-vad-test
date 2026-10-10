#!/usr/bin/env python3
"""golden_q.py —— 用板端同款 int16 截断量化输入生成 PC 金标(隔离固件数学 vs 输入量化)
板端输入: xq = trunc_toward_zero(x*32768)/32768 (HIL 音频 int16 存储)
对比: 板端 [SVs]7-done p 序列 vs 原始金标 vs 量化金标
"""
import sys, re, struct
import numpy as np
import onnxruntime as ort

# m.onnx 由 gen_weights.py 的同源模型(silero_vad_v4 float onnx)放入本目录
# golden_x/golden_prob.f32 在上级目录 porting/firmware/
BASE = __file__.rsplit("/", 1)[0] + "/.."
X = np.fromfile(f"{BASE}/golden_x.f32", np.float32).reshape(-1, 1, 512)
G = np.fromfile(f"{BASE}/golden_prob.f32", np.float32)

# 板端量化: trunc toward zero
Xq = (np.trunc(X * 32768.0) / 32768.0).astype(np.float32)

sess = ort.InferenceSession(f"{BASE}/m.onnx", providers=["CPUExecutionProvider"])
h = np.zeros((2, 1, 64), np.float32); c = np.zeros((2, 1, 64), np.float32)
pq = []
for i in range(len(Xq)):
    out, h, c = sess.run(["prob", "new_h", "new_c"], {"x": Xq[i], "h": h, "c": c})
    pq.append(float(out[0].reshape(-1)[0]))
Pq = np.array(pq, np.float32)
Pq.tofile(f"{BASE}/golden_q_prob.f32")

# 板端 p(HH 日志第二轮, 从窗0起)
board = None
if len(sys.argv) > 1:
    log = open(sys.argv[1], errors="replace").read()
    runs, cur = [], []
    for line in log.splitlines():
        if "[SHIL] begin" in line:
            if cur: runs.append(cur)
            cur = []
        m = re.search(r"\[SVs\]7-done p=(\d+)", line)
        if m: cur.append(int(m.group(1)) / 10000.0)
    if cur: runs.append(cur)
    board = max(runs, key=len) if runs else None  # 最长轮=从窗0起的完整轮

print(f"{'win':>3} | {'板端':>7} | {'原始金标':>7} {'d原':>7} | {'量化金标':>7} {'d量':>7}")
worst_o = worst_q = 0.0
for i in range(len(board) if board is not None else len(Pq)):
    b = board[i] if board is not None else float("nan")
    do = abs(b - G[i]); dq = abs(b - Pq[i])
    worst_o = max(worst_o, do); worst_q = max(worst_q, dq)
    print(f"{i:3d} | {b:7.4f} | {G[i]:7.4f} {do:7.4f} | {Pq[i]:7.4f} {dq:7.4f}")
print(f"\n最大偏差: vs原始={worst_o:.4f}  vs量化金标={worst_q:.4f}")
print("结论: d量≈0 → 固件数学精确, 偏差纯属 int16 输入量化 → 换用 golden_q 作 HIL 基准即闭环")
