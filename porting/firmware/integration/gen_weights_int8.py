#!/usr/bin/env python3
"""gen_weights_int8.py —— C 路线: 生成 int8 权重 C 文件(LSTM per-row 统一 scale + dw + 窗)
用法: .venv/bin/python porting/firmware/integration/gen_weights_int8.py <输出.c>
语义(与 silero_vad.c SS 实现对齐):
  gate_f[i] = acc32[i]·s_c[i] + bsum[i],  acc32 = matXvec_8x8_32(Wq,Rq, x8,h8)
  其中 Wq[i][k]=round(W[i][k]/s_w[i]), s_w[i]=s_c[i]/s_x;  Rq 同理除 s_h
  x8 = round(e6/s_x), h8 = round(h/s_h)   (per-tensor, 标定自 golden 40 窗, 5% 余量)
  dw: out[c] = (Σ_k dwq[c][k]·x)·s_dc[c] + b[c]
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import warnings
from onnx import numpy_helper

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[3]   # 仓库根
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "porting/firmware/silero_vad_weights_int8.c"

m = onnx.load(BASE / "porting/firmware/m.onnx")
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                W[n.output[0]] = numpy_helper.to_array(a.t)
g = lambda k: W[k].astype(np.float32).ravel()
X = np.fromfile(BASE / "porting/firmware/golden_x.f32", np.float32).reshape(-1, 512)

# ---- e6/h 范围标定(float 参考流水线, 与 quant_sim 同口径) ----
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
e6rng = [np.inf, -np.inf]; hrng = [np.inf, -np.inf]
A16 = {}   # 1x1 int16 激活标定: 张量名 -> [lo,hi]
h1 = c1 = h2 = c2 = np.zeros(64, np.float32)
for wI in range(40):
    x = X[wI]
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
    feat = np.concatenate([mag.astype(np.float32), (lg - base).astype(np.float32)])
    proj = pw1x1f("first_layer.0.proj.weight", "first_layer.0.proj.bias", feat, 258, 16, 8, 0)
    dwt = dwf("first_layer.0.dw_conv.0.weight", "first_layer.0.dw_conv.0.bias", feat, 258, 8)
    pw = pw1x1f("first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias", dwt, 258, 16, 8, 0)
    act0 = np.maximum(pw + proj, 0)
    e0 = np.maximum(g("onnx::Conv_349").reshape(16, 16) @ act0 + g("onnx::Conv_350")[:, None], 0)[:, 0::2]
    pj1 = pw1x1f("encoder.3.0.proj.weight", "encoder.3.0.proj.bias", e0, 16, 32, 4, 0)
    d1 = dwf("encoder.3.0.dw_conv.0.weight", "encoder.3.0.dw_conv.0.bias", e0, 16, 4)
    w1 = pw1x1f("encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias", d1, 16, 32, 4, 0)
    act1 = np.maximum(w1 + pj1, 0)
    e2 = np.maximum(g("onnx::Conv_352").reshape(32, 32) @ act1 + g("onnx::Conv_353")[:, None], 0)[:, 0::2]
    d2 = dwf("encoder.7.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.bias", e2, 32, 2)
    w2 = pw1x1f("encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias", d2, 32, 32, 2, 0)
    act2 = np.maximum(w2 + e2, 0)
    e4 = np.maximum(g("onnx::Conv_355").reshape(32, 32) @ act2 + g("onnx::Conv_356")[:, None], 0)[:, 0::2]
    pj3 = pw1x1f("encoder.11.0.proj.weight", "encoder.11.0.proj.bias", e4, 32, 64, 1, 0)
    d3 = dwf("encoder.11.0.dw_conv.0.weight", "encoder.11.0.dw_conv.0.bias", e4, 32, 1)
    w3 = pw1x1f("encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias", d3, 32, 64, 1, 0)
    act3 = np.maximum(w3 + pj3, 0)
    e6 = np.maximum(g("onnx::Conv_358").reshape(64, 64) @ act3 + g("onnx::Conv_359")[:, None], 0)
    for nm, v in [("feat", feat), ("dwt", dwt), ("act0", act0), ("e0", e0),
                  ("d1", d1), ("act1", act1), ("e2", e2), ("d2", d2),
                  ("act2", act2), ("e4", e4), ("d3", d3), ("act3", act3)]:
        lo, hi = A16.setdefault(nm, [np.inf, -np.inf])
        A16[nm] = [min(lo, float(np.min(v))), max(hi, float(np.max(v)))]
    e6rng[0] = min(e6rng[0], float(e6.min())); e6rng[1] = max(e6rng[1], float(e6.max()))
    h1o, c1 = lstmf("onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400", e6[:, 0], h1, c1)
    h2o, c2 = lstmf("onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420", h1o, h2, c2)
    hrng[0] = min(hrng[0], float(h1o.min()), float(h2o.min()))
    hrng[1] = max(hrng[1], float(h1o.max()), float(h2o.max()))
    h1, h2 = h1o, h2o

S_X = max(abs(e6rng[0]), abs(e6rng[1])) * 1.05 / 127.0    # e6 int8 标定
S_H = max(abs(hrng[0]), abs(hrng[1])) * 1.05 / 127.0      # h int8 标定
print(f"标定: e6 范围 {e6rng} → s_x={S_X:.6f}; h 范围 {hrng} → s_h={S_H:.6f}")

# ---- LSTM per-row 统一 scale ----
def lstm_rows(k, rows=256, cols=64):
    return g(k).reshape(rows, cols)
def emit_lstmq(name, Wk, Rk):
    Wm = lstm_rows(Wk); Rm = lstm_rows(Rk)
    s_c = 1.05 * np.maximum(Wm.max(axis=1) * S_X, Rm.max(axis=1) * S_H) / 127.0
    s_c[s_c == 0] = 1e-12
    Wq = np.clip(np.round(Wm / (s_c[:, None] / S_X)), -127, 127).astype(np.int8)
    Rq = np.clip(np.round(Rm / (s_c[:, None] / S_H)), -127, 127).astype(np.int8)
    return Wq, Rq, s_c
W1q, R1q, sc1 = emit_lstmq("L1", "onnx::LSTM_398", "onnx::LSTM_399")
W2q, R2q, sc2 = emit_lstmq("L2", "onnx::LSTM_418", "onnx::LSTM_419")

# ---- dw per-channel int8 ----
def emit_dw(name, k, C):
    w = g(k).reshape(C, 5)
    s_dc = 1.05 * w.max(axis=1) / 127.0; s_dc[s_dc == 0] = 1e-12
    q = np.clip(np.round(w / s_dc[:, None]), -127, 127).astype(np.int8)
    return q, s_dc
DW = {n: emit_dw(n, k, c) for n, k, c in [
    ("dw1", "first_layer.0.dw_conv.0.weight", 258), ("dw2", "encoder.3.0.dw_conv.0.weight", 16),
    ("dw3", "encoder.7.0.dw_conv.0.weight", 32), ("dw4", "encoder.11.0.dw_conv.0.weight", 32)]}

# ---- 窗 = basis 行 0 ----
win = g("feature_extractor.forward_basis_buffer").reshape(258, 256)[0]

# ---- 发射 C 文件 ----
def carr(name, arr, typ):
    v = ",".join(str(int(x)) for x in np.asarray(arr).ravel())
    return f"const {typ} {name}[{arr.size}] = {{\n{v}\n}};\n"
def cflt(name, arr):
    def fmt(x):
        t = f"{x:.9g}"
        if not any(ch in t for ch in ".eE"):   # "0"→"0.0", "5"→"5.0" (否则 "0f"/"5f" 非法 C)
            t += ".0"
        return t + "f"
    v = ",".join(fmt(x) for x in np.asarray(arr).ravel())
    return f"const float {name}[{np.asarray(arr).size}] = {{\n{v}\n}};\n"

with open(OUT, "w") as f:
    f.write("""/* 自动生成: gen_weights_int8.py (C 路线 int8 权重, 勿手改)
   语义: gate_f[i]=acc32[i]*s_c[i]+bsum[i]; acc32=matXvec_8x8_32(Wq,Rq,x8,h8,NULL,256,64,64,64,64,0,0)
   x8=round(e6/s_x), h8=round(h/s_h); dw: out[c]=(Σ dwq[c][k]*x)*s_dc[c]+b[c] */
#include <stdint.h>
""")
    f.write(carr("svq_lstm_w1", W1q, "int8_t") + "\n")
    f.write(carr("svq_lstm_r1", R1q, "int8_t") + "\n")
    f.write(cflt("svq_lstm_sc1", sc1) + "\n")
    f.write(carr("svq_lstm_w2", W2q, "int8_t") + "\n")
    f.write(carr("svq_lstm_r2", R2q, "int8_t") + "\n")
    f.write(cflt("svq_lstm_sc2", sc2) + "\n")
    for n, (q, sdc) in DW.items():
        f.write(carr(f"svq_{n}", q, "int8_t") + "\n")
        f.write(cflt(f"svq_{n}_sc", sdc) + "\n")
    f.write(cflt("sv_win", win) + "\n")
    f.write(f"const float svq_s_x = {S_X:.8g}f;\n")
    f.write(f"const float svq_s_h = {S_H:.8g}f;\n")

    # ---- 1x1 层 int8 权重(per-row) + int16 激活 scale(XX 包: matXvec_8x16_32) ----
    f.write("/* ---- 1x1 层 int8(XX): out[c][t]=acc32[c]*sw[c]*s16 + b[c]; "
            "mat2=零列技巧(cols2=1) ---- */\n")
    P1 = {"p1": ("first_layer.0.proj.weight", 16), "w1": ("first_layer.0.pw_conv.0.weight", 16),
          "c349": ("onnx::Conv_349", 16), "ep1": ("encoder.3.0.proj.weight", 32),
          "ew1": ("encoder.3.0.pw_conv.0.weight", 32), "c352": ("onnx::Conv_352", 32),
          "ew2": ("encoder.7.0.pw_conv.0.weight", 32), "c355": ("onnx::Conv_355", 32),
          "ep3": ("encoder.11.0.proj.weight", 64), "ew3": ("encoder.11.0.pw_conv.0.weight", 64),
          "c358": ("onnx::Conv_358", 64)}
    tot1x1 = 0
    for nm, (k, rows) in P1.items():
        w2d = g(k).reshape(rows, -1)
        s_w = 1.05 * w2d.max(axis=1) / 127.0; s_w[s_w == 0] = 1e-12
        q = np.clip(np.round(w2d / s_w[:, None]), -127, 127).astype(np.int8)
        tot1x1 += q.nbytes
        f.write(carr(f"svq1_{nm}", q, "int8_t") + "\n")
        f.write(cflt(f"svq1_{nm}_sc", s_w) + "\n")
    for nm in ["feat", "dwt", "act0", "e0", "d1", "act1", "e2", "d2",
               "act2", "e4", "d3", "act3"]:
        lo, hi = A16[nm]
        s16 = max(abs(lo), abs(hi)) * 1.05 / 32767.0 or 1e-9
        f.write(f"const float svq1_s_{nm.replace('.', '_')} = {s16:.8g}f;\n")
    f.write("const int8_t svq1_zmat8[64] = {0};\n")     # 零 mat2([rows×1])
    f.write("const int16_t svq1_zvec16[1] = {0};\n")   # 零 vec2
    print(f"1x1 int8 权重: {tot1x1/1024:.1f}KB")
tot = (W1q.nbytes + R1q.nbytes + W2q.nbytes + R2q.nbytes + sum(q.nbytes for q, _ in DW.values()))
print(f"已生成 {OUT}: int8 权重 {tot/1024:.1f}KB + scales + 窗 1KB")
