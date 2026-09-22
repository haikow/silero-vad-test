#!/usr/bin/env python3
"""gen_weights_fixed.py —— 从 float ONNX 生成 int16 定点权重 + 全部量化常数
用法(需 onnx): python3 gen_weights_fixed.py <silero_vad_v4_float.onnx>
产出: silero_vad_fixed_weights.c/h

量化方案:
- 权重 int16 逐行(输出通道)缩放; 激活 int16 定点(逐层 Q 格式, 由标定范围决定)
- conv/lstm 累加 int64, requant 用 (accum x M)>>31, M=int32 Q31 乘数(逐行)
- bias 折成累加单位 int32
- 非线性: sigmoid 查表(输入 Q8, 输出 Q15), tanh 用 2*sig(2x)-1
- STFT: 基 int16 逐张量; 幅度用整数开方; log 用 ln 分解 + 512 项 LUT
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

MODEL = sys.argv[1] if len(sys.argv) > 1 else "silero_vad_v4_float.onnx"
m = onnx.load(MODEL)
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

# ---- 标定范围(2026-09-22, 5325 窗场景语料实测 absmax) ----
CAL = dict(mag=38.1471, act0=54.62, e0=25.73, act1=25.43, e2=15.56,
           act2=23.62, e4=14.50, act3=15.91, e6=19.40, gate=58.20, pre=6.99)
# ---- 逐层 Q 格式(激活 = 值 x 2^Q) ----
Q = dict(pcm=15, feat=9, act0=8, e0=9, act1=9, e2=10, act2=9, e4=10,
         act3=10, e6=10, gate=8, lg=11, c=11, h=14, prob=15)

def q31(x):
    v = int(round(float(x) * (1 << 31)))
    assert -(1 << 31) <= v < (1 << 31), f"M 溢出: {x}"
    return v

out_h = []
out_c = []

def emit_w_int16(name, arr2d_rows):
    """逐行 int16 权重: 返回 (int16 扁平数组, 每行缩放 s[rows])"""
    rows = np.atleast_2d(arr2d_rows)
    n, k = rows.shape
    flat = np.zeros(n * k, np.int16)
    s = np.zeros(n, np.float64)
    for r in range(n):
        s[r] = max(np.abs(rows[r]).max(), 1e-12) / 32767.0
        flat[r*k:(r+1)*k] = np.clip(np.round(rows[r] / s[r]), -32768, 32767)
    out_c.append(f"const short {name}[{n*k}] = {{ /* {rows.shape} 逐行缩放 */")
    out_c.append(",\n".join(", ".join(str(int(v)) for v in flat[i:i+12]) for i in range(0, n*k, 12)))
    out_c.append("};")
    return s

def emit_i32(name, arr):
    a = np.asarray(arr, np.int64)
    out_c.append(f"const int {name}[{len(a)}] = {{")
    out_c.append(",\n".join(", ".join(str(int(v)) for v in a[i:i+12]) for i in range(0, len(a), 12)))
    out_c.append("};")

def emit_i64(name, v):
    out_c.append(f"const long long {name} = {int(v)}LL;")

def emit_def(name, v):
    out_h.append(f"#define {name} {v}")

# ============ STFT 基(逐张量) ============
basis = W["feature_extractor.forward_basis_buffer"].reshape(258, 256)
s_b = np.abs(basis).max() / 32767.0
bq = np.clip(np.round(basis / s_b), -32768, 32767).astype(np.int16)
out_c.append(f"const short svq_basis[258*256] = {{ /* scale {s_b:.10g} */")
out_c.append(",\n".join(", ".join(str(int(v)) for v in bq.ravel()[i:i+12]) for i in range(0, bq.size, 12)))
out_c.append("};")
# mag_q 的实际值缩放: STFT 累加右移 SVQ_STFT_SHIFT 位后存储, 防止 int32 溢出
STFT_SHIFT = 8
S_m = s_b * 2.0 ** (-15 + STFT_SHIFT)
emit_def("SVQ_Sm_Q31", q31(S_m))                 # mag_true = mag_q * S_m
emit_def("SVQ_MAG2FEAT_M", q31(S_m * 2.0 ** Q["feat"]))  # featM = mag_q*此值>>31
# log: lg = ln(A*mag_q+1), A = S_m*2^20
A = S_m * 2.0 ** 20
emit_def("SVQ_LNC", int(round(np.log(A) * 2 ** Q["lg"])))  # ln(A) Q11 (可为负)
emit_def("SVQ_STFT_SHIFT", STFT_SHIFT)
emit_i64("SVQ_INV_A", int(round(1.0 / A)))
# log LUT: ln(1+m), 512 级中点采样(消系统性向下偏), C 侧线性插值
lut = np.round(np.log1p((np.arange(513) + 0.5 - 0.5) / 512.0) * 2 ** Q["lg"]).astype(np.int32)
lut = np.round(np.log1p((np.arange(513)) / 512.0) * 2 ** Q["lg"]).astype(np.int32)  # 513 点: 0..512
emit_i32("svq_ln_lut", lut)
# ln2 Q11
emit_def("SVQ_LN2_Q11", int(round(np.log(2.0) * 2 ** Q["lg"])))

# ============ 归一化 filter_(逐张量) ============
filt = W["adaptive_normalization.filter_"].ravel()
s_f = np.abs(filt).max() / 32767.0
fq = np.clip(np.round(filt / s_f), -32768, 32767).astype(np.int16)
out_c.append(f"const short svq_filter7[7] = {{ {', '.join(str(int(v)) for v in fq)} }}; /* scale {s_f:.10g} */")
# m 序列 Q11 -> conv7 输出仍 Q11: out = accum * s_f
emit_def("SVQ_FILT_M", q31(s_f))

# ============ conv 层(逐行) ============
# (weights 2D reshape, bias, 输入Q, 输出Q, 前缀名)
def conv_layer(wname, bname, Qin, Qout, prefix):
    w = W[wname]
    b = W[bname].ravel()
    if w.ndim == 3: w = w.reshape(w.shape[0], -1)
    s = emit_w_int16(f"svq_{prefix}_w", w)
    n = w.shape[0]
    # bias 累加单位: b / (s*2^-Qin)
    bq = np.round(b / (s * 2.0 ** -Qin)).astype(np.int64)
    emit_i32(f"svq_{prefix}_b", bq)
    # M = s * 2^(Qout-Qin)
    M = np.array([q31(s[r] * 2.0 ** (Qout - Qin)) for r in range(n)])
    emit_i32(f"svq_{prefix}_M", M)

# first_layer 残差: pw 输出 + proj 输出相加后过 relu, 两支都要到同一 Q
# proj 支: 输出直接 requant 到 act0 前的 "pw 同格式" —— 让两支同 Q:
# 重做: proj 输出到 Q=Q['act0']-? 残差 add 发生在 relu 前, 取 Qres = Q['act0'] 带饱和余量
# 简化: proj 输出 Q 与 pw 输出 Q 相同 (设 Qres=Q['act0']), relu 后饱和到 int16
conv_layer("first_layer.0.dw_conv.0.weight", "first_layer.0.dw_conv.0.bias", Q["feat"], Q["act0"], "fl_dw")
conv_layer("first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias", Q["act0"], Q["act0"], "fl_pw")
conv_layer("first_layer.0.proj.weight", "first_layer.0.proj.bias", Q["feat"], Q["act0"], "fl_proj")

conv_layer("onnx::Conv_349", "onnx::Conv_350", Q["act0"], Q["e0"], "enc0")
conv_layer("encoder.3.0.proj.weight", "encoder.3.0.proj.bias", Q["e0"], Q["act1"], "en3_proj")
conv_layer("encoder.3.0.dw_conv.0.weight", "encoder.3.0.dw_conv.0.bias", Q["e0"], Q["act1"], "en3_dw")
conv_layer("encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias", Q["act1"], Q["act1"], "en3_pw")
conv_layer("onnx::Conv_352", "onnx::Conv_353", Q["act1"], Q["e2"], "enc4")
conv_layer("encoder.7.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.bias", Q["e2"], Q["act2"], "en7_dw")
conv_layer("encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias", Q["act2"], Q["act2"], "en7_pw")
conv_layer("onnx::Conv_355", "onnx::Conv_356", Q["act2"], Q["e4"], "enc8")
conv_layer("encoder.11.0.proj.weight", "encoder.11.0.proj.bias", Q["e4"], Q["act3"], "en11_proj")
conv_layer("encoder.11.0.dw_conv.0.weight", "encoder.11.0.dw_conv.0.bias", Q["e4"], Q["act3"], "en11_dw")
conv_layer("encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias", Q["act3"], Q["act3"], "en11_pw")
conv_layer("onnx::Conv_358", "onnx::Conv_359", Q["act3"], Q["e6"], "enc12")

# ============ LSTM(逐行, W/R 共用行缩放) ============
def lstm_layer(wname, rname, bname, Qin_x, prefix):
    Wm = W[wname][0]           # (256, in)
    Rm = W[rname][0]           # (256, 64)
    Bf = W[bname].reshape(-1)  # (512,) = Wb+Rb
    n = 256
    s = np.zeros(n)
    wq = np.zeros_like(Wm, np.int16)
    rq = np.zeros_like(Rm, np.int16)
    for r in range(n):
        s[r] = max(np.abs(Wm[r]).max(), np.abs(Rm[r]).max(), 1e-12) / 32767.0
        wq[r] = np.clip(np.round(Wm[r] / s[r]), -32768, 32767)
        rq[r] = np.clip(np.round(Rm[r] / s[r]), -32768, 32767)
    out_c.append(f"const short svq_{prefix}_w[{Wm.size}] = {{ /* W {Wm.shape} */")
    out_c.append(",\n".join(", ".join(str(int(v)) for v in wq.ravel()[i:i+12]) for i in range(0, Wm.size, 12)))
    out_c.append("};")
    out_c.append(f"const short svq_{prefix}_r[{Rm.size}] = {{ /* R {Rm.shape} */")
    out_c.append(",\n".join(", ".join(str(int(v)) for v in rq.ravel()[i:i+12]) for i in range(0, Rm.size, 12)))
    out_c.append("};")
    # bias: (Wb+Rb) 累加单位, 输入 = x(Qin_x) 与 h(Q14) 混合 -> 用 x 的单位近似:
    # 精确做法: bias 折两份不必要; 直接 b/(s*2^-Qin) (h 支的 M 单位差异由行缩放吸收近似)
    bq = np.round((Bf[:n] + Bf[n:]) / (s * 2.0 ** -Qin_x)).astype(np.int64)
    emit_i32(f"svq_{prefix}_b", bq)
    M = np.array([q31(s[r] * 2.0 ** (Q["gate"] - Qin_x)) for r in range(n)])
    emit_i32(f"svq_{prefix}_M", M)
    # h 反馈支: h_q14 -> 与 x 单位 (2^Qin) 差 (Q["h"]-Qin_x) 位: 在 C 里 h<<diff 或 >>diff
    emit_def(f"SVQ_{prefix.upper()}_HSHIFT", Q["h"] - Qin_x)

lstm_layer("onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400", Q["e6"], "l1")
lstm_layer("onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420", Q["h"], "l2")

# ============ decoder(单行) ============
dw = W["decoder.decoder.1.weight"].ravel()
db = W["decoder.decoder.1.bias"].ravel()
s_d = np.abs(dw).max() / 32767.0
dwq = np.clip(np.round(dw / s_d), -32768, 32767).astype(np.int16)
out_c.append(f"const short svq_dec_w[64] = {{ {', '.join(str(int(v)) for v in dwq)} }}; /* scale {s_d:.10g} */")
emit_def("SVQ_DEC_B", int(round(db[0] / (s_d * 2.0 ** -Q["h"]))))
emit_def("SVQ_DEC_M", q31(s_d * 2.0 ** (Q["gate"] - Q["h"])))  # pre-sigmoid requant 到 Q8 进 LUT

# ============ sigmoid LUT(Q8 输入 -> Q15 输出, 范围 ±16) ============
xs = np.arange(-4096, 4096) / 256.0
sig = np.round(1.0 / (1.0 + np.exp(-xs)) * 32767.0).astype(np.int16)
out_c.append(f"const short svq_sig_lut[{len(sig)}] = {{ /* Q8 in, Q15 out, [-16,16) */")
out_c.append(",\n".join(", ".join(str(int(v)) for v in sig[i:i+12]) for i in range(0, len(sig), 12)))
out_c.append("};")

# ============ 汇总头 ============
for k, v in Q.items():
    emit_def(f"SVQ_Q_{k.upper()}", v)
emit_def("SVQ_C_CLAMP", 16 * (1 << Q["c"]))
emit_def("SVQ_PROB_HALF", 1 << 14)

hdr = """/* silero_vad_fixed.h —— 自动生成的量化常数(勿手改) */
#ifndef SILERO_VAD_FIXED_H
#define SILERO_VAD_FIXED_H
"""
hdr += "\n".join(out_h) + "\n#endif\n"
open("silero_vad_fixed_consts.h", "w").write(hdr)

body = "/* silero_vad_fixed_weights.c —— 自动生成(勿手改), 由 gen_weights_fixed.py 重新生成 */\n#include \"silero_vad_fixed_consts.h\"\n\n" + "\n".join(out_c) + "\n"
open("silero_vad_fixed_weights.c", "w").write(body)
n16 = sum(1 for _ in []) # 占位
print("生成完成: silero_vad_fixed_consts.h + silero_vad_fixed_weights.c")
print(f"权重 int16 总量约 {(258*256 + 7 + 2*(16*258+258*5+16*258) + 16*16+16*5+32*16 + 32*32+32*5+32*32 + 64*32+32*5+64*32 + 2*(256*64*2))/1:.0f} shorts = {(258*256 + 2*(16*258+258*5+16*258) + 16*16+16*5+32*16 + 32*32+32*5+32*32 + 64*32+32*5+64*32 + 2*(256*64*2))*2/1024:.0f} KB (不含 LUT)")
