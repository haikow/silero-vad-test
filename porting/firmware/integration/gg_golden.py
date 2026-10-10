#!/usr/bin/env python3
"""gg_golden.py —— PC 金标: 与固件 [GG] dump 相同字段的逐窗对照表
复现固件 sv_process 前段: reflect(96,96) -> conv1d(STFT, s64) -> mag -> feat
dump 字段: x[0], stft[0][0], stft[0][129], stft[6][129], mag[0][0], feat[0][0] (全部 x1e5 取整)
用法: python3 gg_golden.py <m.onnx> <golden_x.f32> [窗数]
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

MODEL = sys.argv[1] if len(sys.argv) > 1 else "m.onnx"
XFILE = sys.argv[2] if len(sys.argv) > 2 else "golden_x.f32"
NWIN = int(sys.argv[3]) if len(sys.argv) > 3 else 40

m = onnx.load(MODEL)
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
W = inits["feature_extractor.forward_basis_buffer"]      # [258,1,256]
W = W.reshape(W.shape[0], W.shape[-1]).astype(np.float32)  # [258][256] OC x KH
print(f"forward_basis: {W.shape}", file=sys.stderr)

X = np.fromfile(XFILE, np.float32).reshape(-1, 512)
N = min(NWIN, len(X))

def i5(v):  # 与固件 (int)(v*100000) 一致(向零截断)
    return int(v * 100000)

print("win | x[0]        | stft[0][0]  | stft[0][129] | stft[6][129] | mag[0][0]  | feat[0][0]")
rows = []
for w in range(N):
    x = X[w]
    pad = np.concatenate([x[96:0:-1], x, x[510:414:-1]]).astype(np.float32)  # 96+512+96=704
    assert len(pad) == 704
    # conv1d: out[t][oc] = sum_kh W[oc][kh]*pad[t*64+kh], t=0..7
    stft = np.zeros((8, 258), np.float32)
    for t in range(8):
        seg = pad[t * 64: t * 64 + 256]
        stft[t] = W @ seg                                   # [258]
    mag00 = np.sqrt(stft[0, 0] ** 2 + stft[0, 129] ** 2)    # mag[f=0][t=0]
    r = (w, i5(x[0]), i5(stft[0, 0]), i5(stft[0, 129]), i5(stft[6, 129]), i5(mag00), i5(mag00))
    rows.append(r)
    print(f"{w:3d} | {r[1]:12d} | {r[2]:12d} | {r[3]:12d} | {r[4]:12d} | {r[5]:10d} | {r[6]:10d}")

np.save("gg_golden_stft.npy", np.array([r[2:5] for r in rows]))
print("已存 gg_golden_stft.npy", file=sys.stderr)
