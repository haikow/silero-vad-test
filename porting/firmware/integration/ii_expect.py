#!/usr/bin/env python3
"""ii_expect.py —— II 包全窗预期表: numpy-C(含ReLU修复) 流水线 × int16截断输入 → 预测板端 p
与 onnxruntime golden 对照, 给出判读线
"""
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

# m.onnx 由 gen_weights.py 的同源模型(silero_vad_v4 float onnx)放入本目录
# golden_x/golden_prob.f32 在上级目录 porting/firmware/
BASE = __file__.rsplit("/", 1)[0] + "/.."
m = onnx.load(f"{BASE}/m.onnx")
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                W[n.output[0]] = numpy_helper.to_array(a.t)
g = lambda name: W[name].astype(np.float32).ravel()

X = np.fromfile(f"{BASE}/golden_x.f32", np.float32).reshape(-1, 512)
Xq = (np.trunc(X * 32768.0) / 32768.0).astype(np.float32)  # 板端 int16 截断
G = np.fromfile(f"{BASE}/golden_prob.f32", np.float32)
fb = g("feature_extractor.forward_basis_buffer").reshape(258, 256)
filt = g("adaptive_normalization.filter_").flatten()

def pw1x1(w, b, x, IC, OC, T, relu):
    out = w.reshape(OC, IC) @ x + b[:, None]
    return np.maximum(out, 0) if relu else out

def dwconv5(w, b, x, C, T):
    xp = np.zeros((C, T + 4), np.float32); xp[:, 2:2+T] = x
    out = np.zeros((C, T), np.float32)
    for c in range(C):
        acc = np.full(T, b[c], np.float32)
        for k in range(5):
            acc += w[c*5+k] * xp[c, k:k+T]
        out[c] = np.maximum(acc, 0)      # II 修复: ReLU(dw)
    return out

def lstm_step(Wm, R, bsum, x, h, c):
    gates = bsum + Wm.reshape(256, -1) @ x + R.reshape(256, -1) @ h
    i = 1/(1+np.exp(-gates[0:64])); o = 1/(1+np.exp(-gates[64:128]))
    f = 1/(1+np.exp(-gates[128:192])); gg = np.tanh(gates[192:256])
    c = f*c + i*gg
    return (o*np.tanh(c)).astype(np.float32), c.astype(np.float32)

h1 = np.zeros(64, np.float32); c1 = np.zeros(64, np.float32)
h2 = np.zeros(64, np.float32); c2 = np.zeros(64, np.float32)
pred = []
for w in range(len(Xq)):
    x = Xq[w]
    pad = np.concatenate([x[96:0:-1], x, x[510:414:-1]]).astype(np.float32)
    stft = np.stack([fb @ pad[t*64:t*64+256] for t in range(8)])   # [8][258]
    mag = np.sqrt(stft[:, :129]**2 + stft[:, 129:]**2).T           # [129][8]
    lg = np.log(mag * 1048576.0 + 1.0).astype(np.float32)
    mt = lg.mean(axis=0)
    mp = np.array([mt[3],mt[2],mt[1],mt[0],mt[1],mt[2],mt[3],mt[4],mt[5],mt[6],mt[7],mt[6],mt[5],mt[4]], np.float32)
    base = sum(filt[j]*mp[k+j] for k in range(8) for j in range(7)) / 8.0
    feat = np.concatenate([mag, lg - base], axis=0)

    proj = pw1x1(g("first_layer.0.proj.weight"), g("first_layer.0.proj.bias"), feat, 258, 16, 8, 0)
    dwt = dwconv5(g("first_layer.0.dw_conv.0.weight"), g("first_layer.0.dw_conv.0.bias"), feat, 258, 8)
    pw = pw1x1(g("first_layer.0.pw_conv.0.weight"), g("first_layer.0.pw_conv.0.bias"), dwt, 258, 16, 8, 0)
    act0 = np.maximum(pw + proj, 0)
    e0 = pw1x1(g("onnx::Conv_349"), g("onnx::Conv_350"), act0, 16, 16, 8, 1)[:, 0::2]

    pj1 = pw1x1(g("encoder.3.0.proj.weight"), g("encoder.3.0.proj.bias"), e0, 16, 32, 4, 0)
    d1 = dwconv5(g("encoder.3.0.dw_conv.0.weight"), g("encoder.3.0.dw_conv.0.bias"), e0, 16, 4)
    w1 = pw1x1(g("encoder.3.0.pw_conv.0.weight"), g("encoder.3.0.pw_conv.0.bias"), d1, 16, 32, 4, 0)
    act1 = np.maximum(w1 + pj1, 0)
    e2 = pw1x1(g("onnx::Conv_352"), g("onnx::Conv_353"), act1, 32, 32, 4, 1)[:, 0::2]

    d2 = dwconv5(g("encoder.7.0.dw_conv.0.weight"), g("encoder.7.0.dw_conv.0.bias"), e2, 32, 2)
    w2 = pw1x1(g("encoder.7.0.pw_conv.0.weight"), g("encoder.7.0.pw_conv.0.bias"), d2, 32, 32, 2, 0)
    act2 = np.maximum(w2 + e2, 0)
    e4 = pw1x1(g("onnx::Conv_355"), g("onnx::Conv_356"), act2, 32, 32, 2, 1)[:, 0::2]

    pj3 = pw1x1(g("encoder.11.0.proj.weight"), g("encoder.11.0.proj.bias"), e4, 32, 64, 1, 0)
    d3 = dwconv5(g("encoder.11.0.dw_conv.0.weight"), g("encoder.11.0.dw_conv.0.bias"), e4, 32, 1)
    w3 = pw1x1(g("encoder.11.0.pw_conv.0.weight"), g("encoder.11.0.pw_conv.0.bias"), d3, 32, 64, 1, 0)
    act3 = np.maximum(w3 + pj3, 0)
    e6 = pw1x1(g("onnx::Conv_358"), g("onnx::Conv_359"), act3, 64, 64, 1, 1)

    h1, c1 = lstm_step(g("onnx::LSTM_398"), g("onnx::LSTM_399"),
                       g("onnx::LSTM_400")[:256]+g("onnx::LSTM_400")[256:], e6[:, 0], h1, c1)
    h2, c2 = lstm_step(g("onnx::LSTM_418"), g("onnx::LSTM_419"),
                       g("onnx::LSTM_420")[:256]+g("onnx::LSTM_420")[256:], h1, h2, c2)
    dot = float(np.maximum(h2, 0) @ g("decoder.decoder.1.weight")) + float(g("decoder.decoder.1.bias")[0])
    pred.append(1/(1+np.exp(-dot)))

P = np.array(pred, np.float32)
P.tofile(f"{BASE}/ii_expect_prob.f32")
print("win | II预期(板端语义+量化输入) | onnx金标 | 差")
for i in range(len(P)):
    print(f"{i:3d} | {P[i]:.4f} | {G[i]:.4f} | {P[i]-G[i]:+.4f}")
print(f"最大|预期-金标| = {np.abs(P-G).max():.4f}  (int16量化+累加序的影响)")
