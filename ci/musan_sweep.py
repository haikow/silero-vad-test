# -*- coding: utf-8 -*-
"""MUSAN 噪声全库扫描 (本地校准工具, 不依赖 CI): 量化各引擎在真实噪声上的误报分布

对 ~/vad-lab/musan_extract/musan/noise/free-sound 抽样(每 STRIDE 个取 1 个, 截 60s),
跑 WebRTC(0-3)/FireRed/Silero(int8,uint8), 统计: 触发过≥1段的文件占比 / 帧级误报率均值 /
中位 / p95 / 总四件套段数。用于: ①挑 fixture(误报分布的代表点) ②README 发布分布数据。
CI 上没有全库, CI 用的是从这里挑出的 test_fixtures/musan_*.wav。

用法: python ci/musan_sweep.py [--stride 8] [--max-seconds 60] [--json out.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

from vad_decision import SpeechSegmenter  # noqa: E402

MUSAN_NOISE = Path.home() / "vad-lab" / "musan_extract" / "musan" / "noise" / "free-sound"


def engines():
    import numpy as _np
    from test_vad import WINDOW, VadRunner
    from ci.compare_vad import ENGINES

    def _silero_stream(r, a):
        r.reset()
        return _np.array([r.process(a[i:i + WINDOW]) for i in range(0, len(a) - WINDOW, WINDOW)])

    out = list(ENGINES)

    def make_silero(rr):
        def run(a):
            return _silero_stream(rr, a), 0.032, 0.032, 0.5, 0.6, 0.35
        return run

    for key in ("int8", "uint8"):
        r = VadRunner(str(BASE / "models" / f"silero_vad_{key}.onnx"))
        out.append((f"silero-{key}", make_silero(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-seconds", type=int, default=60)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(MUSAN_NOISE.glob("*.wav"))[::a.stride]
    print(f"扫描 {len(files)}/{len(list(MUSAN_NOISE.glob('*.wav')))} 个噪声文件, 截 {a.max-seconds if hasattr(a,'max-seconds') else a.max_seconds}s")
    eng = engines()
    stats = {name: {"trig_files": 0, "fps": [], "segments": 0} for name, _ in eng}

    for idx, f in enumerate(files):
        audio, sr = sf.read(f, dtype="float32", frames=a.max_seconds * 16000)
        assert sr == 16000
        for name, run in eng:
            probs, shift_s, _flen, thr, start, end = run(audio)
            s = SpeechSegmenter(start=start, end=end, window_s=shift_s)
            s.feed(probs)
            n_seg = len(s.final_segments())
            fp = float((probs >= thr).mean())
            stats[name]["fps"].append(fp)
            stats[name]["segments"] += n_seg
            if n_seg:
                stats[name]["trig_files"] += 1
        if (idx + 1) % 20 == 0:
            print(f"  ... {idx+1}/{len(files)}")

    print(f"\n{'引擎':<12}{'触发文件占比':>10}{'帧误报均值':>9}{'中位':>8}{'p95':>8}{'总段数':>8}")
    summary = {}
    for name, _ in eng:
        st = stats[name]
        fps = st["fps"]
        summary[name] = {
            "n_files": len(files),
            "trig_file_pct": round(st["trig_files"] / len(files), 3),
            "frame_fp_mean": round(statistics.mean(fps), 4),
            "frame_fp_median": round(statistics.median(fps), 4),
            "frame_fp_p95": round(sorted(fps)[int(len(fps) * 0.95) - 1], 4),
            "total_segments": st["segments"],
        }
        s = summary[name]
        print(f"{name:<12}{s['trig_file_pct']:>9.0%}{s['frame_fp_mean']:>9.3f}"
              f"{s['frame_fp_median']:>8.3f}{s['frame_fp_p95']:>8.3f}{s['total_segments']:>8d}")

    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
