#!/usr/bin/env python3
"""gen_golden.py —— 用 onnxruntime float 模型生成黄金测试向量
用法: python3 gen_golden.py <model.onnx> <音频x目录或wav...>  [窗口数]
产出: golden_x.f32 (N*512 float 输入窗), golden_prob.f32 (N float 逐窗概率)
"""
import sys
import glob
import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL = sys.argv[1]
SRCS = sys.argv[2:-1] if len(sys.argv) > 3 else sys.argv[2:3]
NWIN = int(sys.argv[-1]) if len(sys.argv) > 3 and sys.argv[-1].isdigit() else 40

audio = []
for s in SRCS:
    p = None
    if s.endswith(".wav"):
        p = [s]
    else:
        p = sorted(glob.glob(s + "/*.npy"))[:NWIN]
        if p:  # 已切窗的 npy
            X = np.stack([np.load(f) for f in p]).astype(np.float32)
            break
    for f in p:
        a, sr = sf.read(f, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        assert sr == 16000
        audio.append(a)
else:
    a = np.concatenate(audio)
    X = np.stack([a[i:i+512] for i in range(0, len(a)-512, 512)][:NWIN]).astype(np.float32)
    X = X.reshape(len(X), 1, 512)

h = np.zeros((2,1,64), np.float32); c = np.zeros((2,1,64), np.float32)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
probs = []
for i in range(len(X)):
    out, h, c = sess.run(["prob","new_h","new_c"], {"x": X[i], "h": h, "c": c})
    probs.append(float(out[0]))
P = np.array(probs, np.float32)
X.astype(np.float32).tofile("golden_x.f32")
P.tofile("golden_prob.f32")
print(f"golden: {len(P)} 窗, 前5概率 {np.round(P[:5],4).tolist()}")
