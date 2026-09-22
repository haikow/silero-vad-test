#!/usr/bin/env python3
"""gen_weights.py —— 从 silero_vad_v4_float.onnx 生成 C 权重数组
用法(在含 onnx 包的环境): python3 gen_weights.py <model.onnx>
产出: silero_vad_weights.c, silero_vad_consts.h
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

MODEL = sys.argv[1] if len(sys.argv) > 1 else "silero_vad_v4_float.onnx"
m = onnx.load(MODEL)
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
consts = {}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                consts[n.output[0]] = numpy_helper.to_array(a.t)

def c_name(n):
    return (n.replace(".", "_").replace("[", "_").replace("]", "")
             .replace("/", "_").replace(":", "_")
             .replace("onnx__", "").replace("__", "_"))

def fmt(x):
    s = f"{x:.8g}"
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"

ORDER = [
    "feature_extractor.forward_basis_buffer", "adaptive_normalization.filter_",
    "first_layer.0.proj.weight", "first_layer.0.proj.bias",
    "first_layer.0.dw_conv.0.weight", "first_layer.0.dw_conv.0.bias",
    "first_layer.0.pw_conv.0.weight", "first_layer.0.pw_conv.0.bias",
    "onnx::Conv_349", "onnx::Conv_350",
    "encoder.3.0.proj.weight", "encoder.3.0.proj.bias",
    "encoder.3.0.dw_conv.0.weight", "encoder.3.0.dw_conv.0.bias",
    "encoder.3.0.pw_conv.0.weight", "encoder.3.0.pw_conv.0.bias",
    "onnx::Conv_352", "onnx::Conv_353",
    "encoder.7.0.dw_conv.0.weight", "encoder.7.0.dw_conv.0.bias",
    "encoder.7.0.pw_conv.0.weight", "encoder.7.0.pw_conv.0.bias",
    "onnx::Conv_355", "onnx::Conv_356",
    "encoder.11.0.proj.weight", "encoder.11.0.proj.bias",
    "encoder.11.0.dw_conv.0.weight", "encoder.11.0.dw_conv.0.bias",
    "encoder.11.0.pw_conv.0.weight", "encoder.11.0.pw_conv.0.bias",
    "onnx::Conv_358", "onnx::Conv_359",
    "onnx::LSTM_398", "onnx::LSTM_399", "onnx::LSTM_400",
    "onnx::LSTM_418", "onnx::LSTM_419", "onnx::LSTM_420",
    "decoder.decoder.1.weight", "decoder.decoder.1.bias",
]

lines = ["// 自动生成: silero v4 float ONNX 权重 (勿手改, 由 gen_weights.py 重新生成)", ""]
total = 0
for n in ORDER:
    arr = np.ascontiguousarray(inits[n], dtype=np.float32)
    flat = arr.ravel()
    total += flat.size
    lines.append(f"/* {n} {list(arr.shape)} */")
    lines.append(f"const float sv_{c_name(n)}[{flat.size}] = {{")
    vals = [fmt(x) for x in flat]
    lines.append(",\n".join(", ".join(vals[i:i+8]) for i in range(0, len(vals), 8)))
    lines.append("};")
    lines.append("")
lines.append(f"/* total: {total} floats = {total*4} bytes */")
open("silero_vad_weights.c", "w").write("\n".join(lines))
print(f"silero_vad_weights.c: {total} floats = {total*4/1024:.0f} KB")

with open("silero_vad_consts.h", "w") as f:
    f.write("// 归一化常数(ONNX Constant 节点提取)\n")
    for key, macro, desc in [
            ("/adaptive_normalization/Constant_output_0", "SV_NORM_MUL", "幅度缩放(log 前)"),
            ("/adaptive_normalization/Constant_1_output_0", "SV_NORM_ADD", "加法偏移(log 前)")]:
        v = np.atleast_1d(consts[key]).astype(float)[0]
        f.write(f"#define {macro} {v:.9g}f  /* {desc} */\n")
print("silero_vad_consts.h 完成")
