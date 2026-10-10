#!/usr/bin/env python3
"""int8 量化仿真器(C 路线量化标定) — 逐层模拟板上 int8 计算, 评估判定一致性。

两级模式:
  W8A32: 仅权重量化(per-row scale), 激活 float —— 隔离权重量化误差
  W8A8 : 激活也量化(静态标定 scale, golden 40 窗标定 + 5% 余量) —— 板上真实形态
用法: .venv/bin/python porting/firmware/integration/quant_sim.py [W8A32|W8A8]
基线: porting/firmware/{m.onnx, golden_x.f32, golden_prob.f32}
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import warnings
from onnx import numpy_helper

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[3]   # 仓库根
m = onnx.load(BASE / "porting/firmware/m.onnx")
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                W[n.output[0]] = numpy_helper.to_array(a.t)
g = lambda name: W[name].astype(np.float32).ravel()
X = np.fromfile(BASE / "porting/firmware/golden_x.f32", np.float32).reshape(-1, 512)
G = np.fromfile(BASE / "porting/firmware/golden_prob.f32", np.float32)

MODE = sys.argv[1] if len(sys.argv) > 1 else "W8A8"

# ---------- 权重 per-row(per-output-channel) int8 ----------
def q8row(w2d):
    s = np.abs(w2d).max(axis=1) / 127.0
    s[s == 0] = 1e-8
    return np.clip(np.round(w2d / s[:, None]), -127, 127), s

WLIST = ["first_layer.0.proj.weight", "first_layer.0.pw_conv.0.weight",
         "encoder.3.0.proj.weight", "encoder.3.0.pw_conv.0.weight",
         "encoder.7.0.pw_conv.0.weight", "encoder.11.0.proj.weight",
         "encoder.11.0.pw_conv.0.weight", "onnx::Conv_349", "onnx::Conv_352",
         "onnx::Conv_355", "onnx::Conv_358", "onnx::LSTM_398", "onnx::LSTM_399",
         "onnx::LSTM_418", "onnx::LSTM_419", "first_layer.0.dw_conv.0.weight",
         "encoder.3.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.weight",
         "encoder.11.0.dw_conv.0.weight"]
IC_MAP = {"first_layer.0.proj.weight": 258, "first_layer.0.pw_conv.0.weight": 258,
          "first_layer.0.dw_conv.0.weight": 5, "encoder.3.0.proj.weight": 16,
          "encoder.3.0.pw_conv.0.weight": 16, "encoder.3.0.dw_conv.0.weight": 5,
          "encoder.7.0.pw_conv.0.weight": 32, "encoder.7.0.dw_conv.0.weight": 5,
          "encoder.11.0.proj.weight": 32, "encoder.11.0.pw_conv.0.weight": 32,
          "encoder.11.0.dw_conv.0.weight": 5, "onnx::Conv_349": 16, "onnx::Conv_352": 32,
          "onnx::Conv_355": 32, "onnx::Conv_358": 64,
          "onnx::LSTM_398": 64, "onnx::LSTM_399": 64, "onnx::LSTM_418": 64, "onnx::LSTM_419": 64}
QW = {}
for k in WLIST:
    QW[k] = q8row(g(k).reshape(-1, IC_MAP[k]))
def qg(k):
    q, s = QW[k]
    return (q.astype(np.float32) * s[:, None]).ravel()

# ---------- 激活静态标定 ----------
ACTS = ["feat", "proj", "dwt1", "pw1", "act0", "e0", "pj1", "w1", "act1", "e2",
        "act2", "e4", "pj3", "w3", "act3", "e6", "h1", "h2"]
rng = {a: [np.inf, -np.inf] for a in ACTS}
def obs(a, v):
    v = np.asarray(v, np.float32)
    rng[a][0] = min(rng[a][0], float(v.min()))
    rng[a][1] = max(rng[a][1], float(v.max()))
SCALE = {}
def qs(a, v):
    s = SCALE[a]
    return np.clip(np.round(np.asarray(v, np.float32) / s), -127, 127).astype(np.int32), s

def pw1x1f(wn, bn, x, IC, OC, T, relu):
    out = g(wn).reshape(OC, IC) @ x + g(bn)[:, None]
    return np.maximum(out, 0) if relu else out
def dwf(wn, bn, x, C, T):
    xp = np.zeros((C, T + 4), np.float32); xp[:, 2:2 + T] = x
    out = np.zeros((C, T), np.float32)
    for c in range(C):
        acc = np.full(T, g(bn)[c], np.float32)
        for k in range(5):
            acc += g(wn)[c * 5 + k] * xp[c, k:k + T]
        out[c] = np.maximum(acc, 0)
    return out
def lstmf(wn, rn, bn, x, h, c):
    gt = (g(bn)[:256] + g(bn)[256:]) + g(wn).reshape(256, -1) @ x + g(rn).reshape(256, -1) @ h
    i = 1 / (1 + np.exp(-gt[0:64])); o = 1 / (1 + np.exp(-gt[64:128]))
    f = 1 / (1 + np.exp(-gt[128:192])); gg = np.tanh(gt[192:256])
    c = f * c + i * gg
    return (o * np.tanh(c)).astype(np.float32), c.astype(np.float32)

fb = g("feature_extractor.forward_basis_buffer").reshape(258, 256)
filt = g("adaptive_normalization.filter_").flatten()
def feats_of(x):
    pad = np.concatenate([x[96:0:-1], x, x[510:414:-1]]).astype(np.float32)
    stft = np.stack([fb @ pad[t*64:t*64+256] for t in range(8)])
    mag = np.sqrt(stft[:, :129]**2 + stft[:, 129:]**2).T
    lg = np.log(mag * 1048576.0 + 1.0).astype(np.float32)
    mt = lg.mean(axis=0, dtype=np.float32)
    mp = np.array([mt[3],mt[2],mt[1],mt[0],mt[1],mt[2],mt[3],mt[4],mt[5],mt[6],mt[7],mt[6],mt[5],mt[4]], np.float32)
    base = np.float32(0)
    for k in range(8):
        acc = np.float32(0)
        for j in range(7):
            acc = np.float32(acc + filt[j] * mp[k+j])
        base = np.float32(base + acc)
    base = np.float32(base / 8.0)
    return np.concatenate([mag.astype(np.float32), (lg - base).astype(np.float32)])

# ---- 第一遍: 标定 ----
h1 = c1 = h2 = c2 = np.zeros(64, np.float32)
for wI in range(40):
    feat = feats_of(X[wI]); obs("feat", feat)
    proj = pw1x1f("first_layer.0.proj.weight", "first_layer.0.proj.bias", feat, 258, 16, 8, 0); obs("proj", proj)
    dwt = dwf("first_layer.0.dw_conv.0.weight", "first_layer.0.dw_conv.0.bias", feat, 258, 8); obs("dwt1", dwt)
    pw = pw1x1f("first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias", dwt, 258, 16, 8, 0); obs("pw1", pw)
    act0 = np.maximum(pw + proj, 0); obs("act0", act0)
    e0 = np.maximum(g("onnx::Conv_349").reshape(16, 16) @ act0 + g("onnx::Conv_350")[:, None], 0)[:, 0::2]; obs("e0", e0)
    pj1 = pw1x1f("encoder.3.0.proj.weight", "encoder.3.0.proj.bias", e0, 16, 32, 4, 0); obs("pj1", pj1)
    d1 = dwf("encoder.3.0.dw_conv.0.weight", "encoder.3.0.dw_conv.0.bias", e0, 16, 4); obs("dwt1", d1)
    w1 = pw1x1f("encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias", d1, 16, 32, 4, 0); obs("w1", w1)
    act1 = np.maximum(w1 + pj1, 0); obs("act1", act1)
    e2 = np.maximum(g("onnx::Conv_352").reshape(32, 32) @ act1 + g("onnx::Conv_353")[:, None], 0)[:, 0::2]; obs("e2", e2)
    d2 = dwf("encoder.7.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.bias", e2, 32, 2); obs("dwt1", d2)
    w2 = pw1x1f("encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias", d2, 32, 32, 2, 0); obs("w1", w2)
    act2 = np.maximum(w2 + e2, 0); obs("act2", act2)
    e4 = np.maximum(g("onnx::Conv_355").reshape(32, 32) @ act2 + g("onnx::Conv_356")[:, None], 0)[:, 0::2]; obs("e4", e4)
    pj3 = pw1x1f("encoder.11.0.proj.weight", "encoder.11.0.proj.bias", e4, 32, 64, 1, 0); obs("pj3", pj3)
    d3 = dwf("encoder.11.0.dw_conv.0.weight", "encoder.11.0.dw_conv.0.bias", e4, 32, 1); obs("dwt1", d3)
    w3 = pw1x1f("encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias", d3, 32, 64, 1, 0); obs("w3", w3)
    act3 = np.maximum(w3 + pj3, 0); obs("act3", act3)
    e6 = np.maximum(g("onnx::Conv_358").reshape(64, 64) @ act3 + g("onnx::Conv_359")[:, None], 0); obs("e6", e6)
    h1o, c1 = lstmf("onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400", e6[:, 0], h1, c1); obs("h1", h1o)
    h2o, c2 = lstmf("onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420", h1o, h2, c2); obs("h2", h2o)
    h1, h2 = h1o, h2o
for a in ACTS:
    lo, hi = rng[a]
    SCALE[a] = max(abs(lo), abs(hi)) * 1.05 / 127.0 or 1e-8

# ---- 第二遍: 量化仿真 ----
def matq(wn, bn, xq, sx, IC, OC, T):
    wq, sw = QW[wn]
    acc = wq.astype(np.int32) @ xq.astype(np.int32).reshape(IC, T)
    return acc.astype(np.float32) * (sw[:, None] * sx) + g(bn)[:, None]
def dwconv5q(wn, bn, x, C, T):
    xp = np.zeros((C, T + 4), np.float32); xp[:, 2:2 + T] = x
    out = np.zeros((C, T), np.float32)
    for c in range(C):
        acc = np.full(T, g(bn)[c], np.float32)
        for k in range(5):
            acc += qg(wn)[c * 5 + k] * xp[c, k:k + T]
        out[c] = np.maximum(acc, 0)
    return out
def lstmq(wn, rn, bn, x, h, c):
    xq = np.clip(np.round(x / SCALE["e6"]), -127, 127).astype(np.int32)
    hq = np.clip(np.round(h / SCALE["h1"]), -127, 127).astype(np.int32)
    gt = (g(bn)[:256] + g(bn)[256:]) \
        + (QW[wn][0].astype(np.int32) @ xq) * (QW[wn][1] * SCALE["e6"]) \
        + (QW[rn][0].astype(np.int32) @ hq) * (QW[rn][1] * SCALE["h1"])
    i = 1 / (1 + np.exp(-gt[0:64])); o = 1 / (1 + np.exp(-gt[64:128]))
    f = 1 / (1 + np.exp(-gt[128:192])); gg = np.tanh(gt[192:256])
    c = f * c + i * gg
    return (o * np.tanh(c)).astype(np.float32), c.astype(np.float32)

h1 = c1 = h2 = c2 = np.zeros(64, np.float32)
dev = flips = 0.0
worst_win = -1
for wI in range(40):
    feat = feats_of(X[wI])
    if MODE == "W8A32":
        proj = pw1x1f("first_layer.0.proj.weight", "first_layer.0.proj.bias", feat, 258, 16, 8, 0)
    else:
        fq, fs = qs("feat", feat)
        proj = matq("first_layer.0.proj.weight", "first_layer.0.proj.bias", fq, fs, 258, 16, 8)
    dwt = dwconv5q("first_layer.0.dw_conv.0.weight", "first_layer.0.dw_conv.0.bias", feat, 258, 8)
    if MODE == "W8A32":
        pw = pw1x1f("first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias", dwt, 258, 16, 8, 0)
    else:
        pw = matq("first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias", dwt, 1.0, 258, 16, 8)
    act0 = np.maximum(pw + proj, 0)
    if MODE == "W8A8":
        a0q, a0s = qs("act0", act0)
        e0 = np.maximum(matq("onnx::Conv_349", "onnx::Conv_350", a0q, a0s, 16, 16, 8), 0)[:, 0::2]
    else:
        e0 = np.maximum(qg("onnx::Conv_349").reshape(16, 16) @ act0 + g("onnx::Conv_350")[:, None], 0)[:, 0::2]
    if MODE == "W8A8": e0q, e0s = qs("e0", e0)
    pj1 = matq("encoder.3.0.proj.weight", "encoder.3.0.proj.bias", e0q, e0s, 16, 32, 4) if MODE == "W8A8" \
        else pw1x1f("encoder.3.0.proj.weight", "encoder.3.0.proj.bias", e0, 16, 32, 4, 0)
    d1 = dwconv5q("encoder.3.0.dw_conv.0.weight", "encoder.3.0.dw_conv.0.bias", e0, 16, 4)
    w1 = matq("encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias", d1, 1.0, 16, 32, 4) if MODE == "W8A8" \
        else pw1x1f("encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias", d1, 16, 32, 4, 0)
    act1 = np.maximum(w1 + pj1, 0)
    if MODE == "W8A8":
        a1q, a1s = qs("act1", act1)
        e2 = np.maximum(matq("onnx::Conv_352", "onnx::Conv_353", a1q, a1s, 32, 32, 4), 0)[:, 0::2]
    else:
        e2 = np.maximum(qg("onnx::Conv_352").reshape(32, 32) @ act1 + g("onnx::Conv_353")[:, None], 0)[:, 0::2]
    if MODE == "W8A8": e2q, e2s = qs("e2", e2)
    d2 = dwconv5q("encoder.7.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.bias", e2, 32, 2)
    w2 = matq("encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias", d2, 1.0, 32, 32, 2) if MODE == "W8A8" \
        else pw1x1f("encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias", d2, 32, 32, 2, 0)
    act2 = np.maximum(w2 + e2, 0)
    if MODE == "W8A8":
        a2q, a2s = qs("act2", act2)
        e4 = np.maximum(matq("onnx::Conv_355", "onnx::Conv_356", a2q, a2s, 32, 32, 2), 0)[:, 0::2]
    else:
        e4 = np.maximum(qg("onnx::Conv_355").reshape(32, 32) @ act2 + g("onnx::Conv_356")[:, None], 0)[:, 0::2]
    if MODE == "W8A8": e4q, e4s = qs("e4", e4)
    pj3 = matq("encoder.11.0.proj.weight", "encoder.11.0.proj.bias", e4q, e4s, 32, 64, 1) if MODE == "W8A8" \
        else pw1x1f("encoder.11.0.proj.weight", "encoder.11.0.proj.bias", e4, 32, 64, 1, 0)
    d3 = dwconv5q("encoder.11.0.dw_conv.0.weight", "encoder.11.0.dw_conv.0.bias", e4, 32, 1)
    w3 = matq("encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias", d3, 1.0, 32, 64, 1) if MODE == "W8A8" \
        else pw1x1f("encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias", d3, 32, 64, 1, 0)
    act3 = np.maximum(w3 + pj3, 0)
    if MODE == "W8A8":
        a3q, a3s = qs("act3", act3)
        e6 = np.maximum(matq("onnx::Conv_358", "onnx::Conv_359", a3q, a3s, 64, 64, 1), 0)
    else:
        e6 = np.maximum(qg("onnx::Conv_358").reshape(64, 64) @ act3 + g("onnx::Conv_359")[:, None], 0)
    if MODE == "W8A8":
        h1, c1 = lstmq("onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400", e6[:, 0].copy(), h1, c1)
        h2, c2 = lstmq("onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420", h1, h2, c2)
    else:
        h1, c1 = lstmf("onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400", e6[:, 0], h1, c1)
        h2, c2 = lstmf("onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420", h1, h2, c2)
    dot = float(np.maximum(h2, 0) @ g("decoder.decoder.1.weight")) + float(g("decoder.decoder.1.bias")[0])
    p = 1 / (1 + np.exp(-dot))
    d = abs(p - G[wI])
    if d > dev: dev, worst_win = d, wI
    if (p >= 0.5) != (G[wI] >= 0.5): flips += 1

print(f"{MODE}: 最大偏差 {dev:.4f}(窗{worst_win})  判定翻转 {flips}/40")
print(f"权重体积: float {sum(g(k).nbytes for k in WLIST)/1024:.0f}KB → int8 {sum(QW[k][0].nbytes for k in WLIST)/1024:.0f}KB + scales")
