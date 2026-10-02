# -*- coding: utf-8 -*-
"""VAD 引擎终评评分卡: 从全场景实测数据聚合出分类得分与总分

口径 (与 ci/compare_vad.py 完全一致的素材与阈值):
  检出分 = 17 个语音场景的 >=阈值 覆盖率均值 ×100
    (smoke语音区, 电平×4, 白噪SNR×4, 咖啡馆SNR×3, LibriSpeech×2, 混响×2, 雨中旁白)
  误报分 = (1 - 帧级误报率均值) ×100, 5 组
    (smoke噪声区, 咖啡馆裸babble, 雨声, MUSAN噪声6件均值, MUSAN音乐2件均值)
  总分 = (检出分 + 误报分) / 2  —— 唤醒类前端的五五开; 权重可按产品调整

用法: python ci/vad_scorecard.py
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

from ci.compare_vad import ENGINES, scenarios, rate_in  # noqa: E402


def main():
    from ci.musan_sweep import engines  # ENGINES + silero-int8/uint8
    all_engines = engines()
    scen = scenarios()

    det_names = [n for n, a, r, k in scen if k == "rate"]
    metrics = json.loads((BASE / "scenario_metrics.json").read_text(encoding="utf-8"))["scenarios"]

    rows = {}
    for name, run in all_engines:
        det, fa = {}, {}
        for sname, audio, region, kind in scen:
            probs, shift_s, flen_s, thr, st, en = run(audio)
            if kind == "rate":
                det[sname] = rate_in(probs, shift_s, flen_s, thr, *region)
            else:
                fa[sname] = float((probs >= thr).mean())
        # MUSAN 从指标表取 (与 tests/test_musan.py 同源)
        fa["MUSAN噪声×6"] = metrics["musan/noise"][name]["fp_mean"]
        fa["MUSAN音乐×2"] = metrics["musan/music"][name]["fp_mean"]

        det_score = float(np.mean(list(det.values()))) * 100
        fa_score = (1 - float(np.mean(list(fa.values())))) * 100
        rows[name] = {"det": det_score, "fa": fa_score,
                      "total": (det_score + fa_score) / 2,
                      "worst_det": min(det, key=det.get),
                      "worst_det_val": min(det.values()),
                      "worst_fa": max(fa, key=fa.get),
                      "worst_fa_val": max(fa.values())}

    print("=" * 110)
    print(f"VAD 引擎终评评分卡  (检出 {len(det_names)} 场景 + 误报 5 组; 素材与阈值与 compare_vad 同口径)")
    print("检出分: 17 场景覆盖率均值 | 误报分: (1-帧误报率均值)×100, 含 MUSAN | 总分: 五五开")
    print("=" * 110)
    print(f"{'引擎':<14}{'检出分':>7}{'误报分':>7}{'总分':>7}   最弱检出场景 / 最差误报场景")
    print("-" * 110)
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{name:<14}{r['det']:>6.1f}{r['fa']:>7.1f}{r['total']:>7.1f}   "
              f"{r['worst_det']}({r['worst_det_val']:.0%}) / {r['worst_fa']}({r['worst_fa_val']:.0%})")
    print("\n注: FSMN=离线段输出(非流式帧级); FireRed 阈值 0.6(概率软校准); WebRTC/rVAD=二值判定。"
          "权重可按产品调整(如近讲查准优先 → 提高误报分权重)。")

    out = BASE / "scenario_metrics.json"
    d = json.loads(out.read_text(encoding="utf-8"))
    d["scenarios"]["scorecard"] = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                                       for kk, vv in v.items()} for k, v in rows.items()}
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"评分已并入 {out.name} (scenarios.scorecard)")


if __name__ == "__main__":
    main()
