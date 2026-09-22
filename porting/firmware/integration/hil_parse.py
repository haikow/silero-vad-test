#!/usr/bin/env python3
"""HIL 自环测试结果解析: 串口日志 -> 逐窗概率 -> 与 PC 基准比对
用法: python3 hil_parse.py <串口日志文件或已保存的 [SHIL] 文本> [golden_prob.f32]
验收: 最大偏差 <= 0.1(HIL 标准), 每行 [SHIL] win=NN p=X.XXXX
"""
import re, sys, struct

log = open(sys.argv[1], errors="replace").read()
probs = [float(m.group(1)) for m in re.finditer(r"\[SHIL\] win=\d+ p=(\d+\.\d+)", log)]
if not probs:
    sys.exit("未找到 [SHIL] 输出——检查调试串口(波特率见固件 dbglog 配置)")
golden_path = sys.argv[2] if len(sys.argv) > 2 else "golden_prob.f32"
g = struct.unpack(f"<{len(probs)}f", open(golden_path, "rb").read()[:len(probs)*4])
maxd = max(abs(a-b) for a, b in zip(probs, g))
print(f"固件 {len(probs)} 窗 vs PC 基准: 最大偏差 {maxd:.4f}")
print("PASS (<=0.1)" if maxd <= 0.1 else "FAIL (>0.1)")
sys.exit(0 if maxd <= 0.1 else 1)
