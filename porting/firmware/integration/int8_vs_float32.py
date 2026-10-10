#!/usr/bin/env python3
"""int8 vs float32 全语料判定一致性对比(报告见 int8对比报告.md)
用法: .venv/bin/python porting/firmware/integration/int8_vs_float32.py
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parents[3]   # 仓库根(porting/firmware/integration/)
sys.path.insert(0, str(BASE))
from test_vad import VadRunner, WINDOW, SAMPLE_RATE      # noqa: E402
from vad_decision import SpeechSegmenter                 # noqa: E402

FLOAT_M = str(BASE / "porting/firmware/m.onnx")
INT8_M = str(BASE / "models/silero_vad_int8.onnx")

def segs(model, wav):
    a, sr = sf.read(wav, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    assert sr == SAMPLE_RATE
    r = VadRunner(model)
    probs = [float(r.process(a[i:i+WINDOW])) for i in range(0, len(a)-WINDOW+1, WINDOW)]
    s = SpeechSegmenter()
    for p in probs: s.process(p)
    s.flush()
    return list(s.final_segments()), probs

def main():
    wavs = sorted((BASE/"test_fixtures").glob("*.wav"))
    tot_flips = tot_frames = 0
    print(f"{'fixture':34s} {'窗数':>5} {'概率max差':>9} {'门控翻转':>8} {'段数f/i8':>8}")
    for w in wavs:
        sf_seg, pf = segs(FLOAT_M, w)
        s8_seg, p8 = segs(INT8_M, w)
        pf, p8 = np.array(pf), np.array(p8)
        dmax = float(np.abs(pf-p8).max())
        gatef = np.array([any(st <= i*0.032 < en for st, en in sf_seg) for i in range(len(pf))])
        gate8 = np.array([any(st <= i*0.032 < en for st, en in s8_seg) for i in range(len(p8))])
        flips = int((gatef != gate8).sum())
        tot_flips += flips; tot_frames += len(pf)
        print(f"{w.name:34s} {len(pf):5d} {dmax:9.4f} {flips:8d} {len(sf_seg):4d}/{len(s8_seg):<4d}")
    print(f"\n合计: {tot_frames} 窗, 门控翻转 {tot_flips} ({100*tot_flips/tot_frames:.2f}%)")

if __name__ == "__main__":
    main()
