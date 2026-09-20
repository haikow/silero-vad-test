# -*- coding: utf-8 -*-
"""
固件在环 (HIL) 测试骨架: WQ7036AC 到货后填充串口协议
当前状态: 协议占位, 与 PC 基准的比对逻辑已就绪

预期协议 (与固件同事约定后修改此处):
  PC -> 固件: 逐窗发送 512 样本 int16 音频帧
  固件 -> PC: 每窗回一个人声概率 (文本行 "prob=0.873" 或二进制)
比对: 固件概率曲线 vs test_fixtures/baseline_probs.json (PC 基准)
验收: 逐窗偏差 <= 0.1, 相关系数 >= 0.98
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from test_vad import SAMPLE_RATE, WINDOW  # noqa: E402

FIXTURE = BASE / "test_fixtures" / "smoke_test.wav"
BASELINE = BASE / "test_fixtures" / "baseline_probs.json"
RESULT = BASE / "hil_result.json"
FW_TOLERANCE = 0.10   # 固件定点化后与 PC 浮点的合理偏差
FW_MIN_CORR = 0.98


def list_ports():
    try:
        from serial.tools import list_ports
        ports = [f"{p.device} - {p.description}" for p in list_ports.comports()]
        print("可用串口:\n  " + ("\n  ".join(ports) if ports else "(无)"))
        return [p.device for p in list_ports.comports()]
    except ImportError:
        print("需要 pyserial: pip install pyserial")
        return []


def talk_to_firmware(port: str, wins):
    """与固件交互, 返回逐窗概率列表。协议确定后替换本函数实现。"""
    import serial  # noqa: F401
    raise SystemExit(
        "固件串口协议未定义。拿到 SDK 后在此实现:\n"
        "  1. serial.Serial(port, baudrate=?) 打开串口\n"
        "  2. 逐窗发送 wins[i].astype('<i2').tobytes()\n"
        "  3. 读取每窗返回的概率, 收集为 list[float] 返回")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="", help="开发板串口, 留空则列出可用串口")
    ap.add_argument("--mock", action="store_true",
                    help="用 PC 模型输出模拟固件返回 (联调比对逻辑)")
    a = ap.parse_args()

    if not a.port and not a.mock:
        list_ports()
        print("\n用法: python ci/firmware_test.py --port COM7 (或 --mock 联调)")
        return 1

    audio, sr = sf.read(FIXTURE, dtype="float32")
    assert sr == SAMPLE_RATE
    wins = [audio[i:i + WINDOW] for i in range(0, len(audio) - WINDOW, WINDOW)]

    if a.mock:  # 联调模式: 用 PC 上 v4 输出模拟固件 (模拟定点化偏差)
        from test_vad import MODELS, VadRunner
        r = VadRunner(MODELS["int8  (sherpa-onnx, 208KB)"])
        fw_probs = [r.process(w) for w in wins]
    else:
        fw_probs = talk_to_firmware(a.port, wins)

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    ref = np.array(base["int8"])          # 固件蓝本为 v4/int8
    fw = np.array(fw_probs)
    n = min(len(ref), len(fw))
    dev = np.abs(fw[:n] - ref[:n]).max()
    corr = np.corrcoef(fw[:n], ref[:n])[0, 1]
    ok = dev <= FW_TOLERANCE and corr >= FW_MIN_CORR
    RESULT.write_text(json.dumps({
        "n_windows": n, "max_dev": round(float(dev), 4),
        "corr": round(float(corr), 5), "tolerance": FW_TOLERANCE,
        "pass": bool(ok)}, indent=1), encoding="utf-8")
    print(f"[{'PASS' if ok else 'FAIL'}] 固件 vs PC基准: 最大偏差 {dev:.4f} "
          f"(容差 {FW_TOLERANCE}), 相关系数 {corr:.5f}")
    print(f"报告: {RESULT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
