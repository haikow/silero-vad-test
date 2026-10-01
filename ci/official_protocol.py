# -*- coding: utf-8 -*-
"""FLEURS-VAD 式官方协议复现: 帧级 AUC-ROC / F1 / FAR / Miss + 工作点扫描

官方基准 (FireRedTeam/FireRedVAD README): FLEURS-VAD-102 = FLEURS test 抽 9443 文件,
二值标注 speech/silence —— 朗读语音 + 句间停顿, 专业录音电平。标注集 "coming soon"
未开源, 故用 LibriSpeech test-clean 自建同形态素材 (fleurs_like_spk_{a,b}: 每句 RMS
归一 0.12, 1s 句间停顿填 -45dBFS 麦克风底噪, 拼接边界即逐帧真值, 见 *.gt.json)。

指标: AUC-ROC (概率型) / F1 / FAR / Miss。每引擎两行:
  default = 官方默认工作点 (FireRed/Silero 阈值 0.5, WebRTC 激进度 0~3 全列)
  best    = 该引擎阈值扫描 (0.50~0.90 步 0.05) 的最佳 F1 工作点 —— 回答"基线该怎么设"

用法: python ci/official_protocol.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

SESSIONS = ["fleurs_like_spk_a", "fleurs_like_spk_b"]
SWEEP = np.arange(0.50, 0.91, 0.05)


def auc(labels, scores) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def frame_labels(gt, shift_s, flen_s, n):
    centers = np.arange(n) * shift_s + flen_s / 2
    lab = np.zeros(n, np.int8)
    for s, e in gt["speech"]:
        lab[(centers >= s) & (centers < e)] = 1
    return lab


def metrics(lab, pred):
    tp = int(((lab == 1) & (pred == 1)).sum())
    fp = int(((lab == 0) & (pred == 1)).sum())
    fn = int(((lab == 1) & (pred == 0)).sum())
    tn = int(((lab == 0) & (pred == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    miss = fn / (fn + tp) if (fn + tp) else 0.0
    return f1, far, miss


def evaluate_sessions():
    """计算两个 fleurs_like 会话的全引擎指标, 返回 {stem: {engine: {...}}}"""
    from vad_firered import FireRedRunner
    from test_vad import WINDOW, VadRunner

    def silero_engine(key):
        r = VadRunner(str(BASE / "models" / f"silero_vad_{key}.onnx"))

        def run(a):
            r.reset()
            return (np.array([r.process(a[i:i + WINDOW])
                              for i in range(0, len(a) - WINDOW, WINDOW)]),
                    WINDOW / 16000, WINDOW / 16000)
        return run

    prob_engines = [
        ("FireRed", lambda a: (FireRedRunner().probs(a), 0.01, 0.025)),
        ("Silero-int8", silero_engine("int8")),
        ("Silero-uint8", silero_engine("uint8")),
    ]
    import webrtcvad
    def wrtc(ag):
        def run(a):
            v = webrtcvad.Vad(ag)
            pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
            n = len(a) // 480
            dec = np.array([v.is_speech(pcm[i * 960:(i + 1) * 960], 16000)
                            for i in range(n)], np.int8)
            return dec, 0.03, 0.03
        return run
    bin_engines = [(f"WRTC-{ag}", wrtc(ag)) for ag in (0, 1, 2, 3)]

    agg = {}
    for stem in SESSIONS:
        gt = json.loads((BASE / "test_fixtures" / f"{stem}.gt.json").read_text())
        audio, sr = sf.read(BASE / "test_fixtures" / f"{stem}.wav", dtype="float32")
        assert sr == 16000
        rows = {}
        for name, run in prob_engines:
            probs, shift_s, flen_s = run(audio)
            lab = frame_labels(gt, shift_s, flen_s, len(probs))
            a = auc(lab, probs.astype(np.float64))
            f1d, fard, missd = metrics(lab, (probs >= 0.5).astype(np.int8))
            best = max((((float(t),) + metrics(lab, (probs >= t).astype(np.int8)))
                        for t in SWEEP), key=lambda r: r[1])
            rows[name] = {"auc": round(a, 4), "default_thr": 0.5,
                          "default": [round(f1d, 4), round(fard, 4), round(missd, 4)],
                          "best_thr": round(best[0], 2),
                          "best": [round(best[1], 4), round(best[2], 4), round(best[3], 4)]}
        for name, run in bin_engines:
            probs, shift_s, flen_s = run(audio)
            lab = frame_labels(gt, shift_s, flen_s, len(probs))
            f1, far, miss = metrics(lab, probs)
            rows[name] = {"auc": None, "default_thr": name.split("-")[1],
                          "default": [round(f1, 4), round(far, 4), round(miss, 4)]}
        agg[stem] = rows
    return agg


def main():
    print("=" * 100)
    print("FLEURS-VAD 式官方协议复现 (朗读+停顿, 语音RMS 0.12, 底噪-45dBFS; 帧级 F1/FAR/Miss)")
    print("官方参考(FLEURS-VAD-102): FireRed F1=97.6/FAR=2.7/Miss=3.6 | Silero F1=96.0/FAR=9.4/Miss=4.0 | WebRTC F1=52.3")
    print("=" * 100)
    agg = evaluate_sessions()
    for stem, rows in agg.items():
        silence = json.loads((BASE / "test_fixtures" / f"{stem}.gt.json").read_text())
        print(f"\n--- {stem} ---")
        print(f"{'引擎':<22}{'AUC':>7}  {'default(F1/FAR/Miss)':>22}  {'best-F1 工作点':>22}")
        for name, r in rows.items():
            f1d, fard, missd = r["default"]
            a = r["auc"]
            if r.get("best"):
                bt = r["best_thr"]; bf1, bfar, bmiss = r["best"]
                tail = f"thr={bt:.2f}: {bf1:.1%}/{bfar:.1%}/{bmiss:.1%}"
            else:
                tail = "(0/1 判定, 无阈值扫描)"
            print(f"{name:<22}{('-' if a is None else f'{a:.4f}'):>7}  "
                  f"{f1d:>7.1%}/{fard:>6.1%}/{missd:>6.1%}    {tail}")

    out = BASE / "scenario_metrics.json"
    try:
        d = json.loads(out.read_text(encoding="utf-8"))
    except Exception:
        d = {"scenarios": {}}
    d["scenarios"]["fleurs_like"] = agg
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n指标已并入 {out.name} (scenarios.fleurs_like)")


if __name__ == "__main__":
    main()
