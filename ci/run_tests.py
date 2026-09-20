# -*- coding: utf-8 -*-
"""
CI 基准回归测试: 对固定素材跑两个量化模型, 与基准概率逐窗比对
用法:
  python ci/run_tests.py                    # 回归模式, 偏差超限 exit 1
  python ci/run_tests.py --update-baseline  # 重新生成基准 (模型/素材变更后)
用途: 算法回归 (云端 Actions) + 未来固件移植结果的比对基准
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from test_vad import MODELS, SAMPLE_RATE, WINDOW, VadRunner  # noqa: E402

FIXTURE = BASE / "test_fixtures" / "smoke_test.wav"
BASELINE = BASE / "test_fixtures" / "baseline_probs.json"
TOLERANCE = 0.05      # 逐窗概率容差 (覆盖 ORT 版本间微小数值差异)
MIN_CORR = 0.995      # 曲线相关系数下限


def infer(audio):
    out = {}
    for name, path in MODELS.items():
        key = "uint8" if "uint8" in name else "int8"
        r = VadRunner(path)
        out[key] = [round(float(r.process(audio[i:i + WINDOW])), 4)
                    for i in range(0, len(audio) - WINDOW, WINDOW)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="重新生成基准文件")
    a = ap.parse_args()

    audio, sr = sf.read(FIXTURE, dtype="float32")
    assert sr == SAMPLE_RATE, f"素材采样率 {sr} != {SAMPLE_RATE}"
    print(f"素材: {FIXTURE.name}, {len(audio)/SAMPLE_RATE:.1f}s, "
          f"{len(audio)//WINDOW} 窗")

    if not all((BASE / p).exists() for p in MODELS.values()):
        print("模型缺失, 先运行: python download_models.py")
        return 1
    probs = infer(audio)

    if a.update_baseline or not BASELINE.exists():
        BASELINE.write_text(json.dumps(
            {"sr": SAMPLE_RATE, "window": WINDOW, "tolerance": TOLERANCE,
             **probs}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"基准已写入: {BASELINE}")
        return 0

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    failed = False
    for key in ("uint8", "int8"):
        p, b = np.array(probs[key]), np.array(base[key])
        if len(p) != len(b):
            print(f"[FAIL] {key}: 窗数变化 {len(b)} -> {len(p)} "
                  "(素材或窗口配置变了? 需 --update-baseline)")
            failed = True
            continue
        max_dev = np.abs(p - b).max()
        corr = np.corrcoef(p, b)[0, 1]
        n_over = int((np.abs(p - b) > TOLERANCE).sum())
        status = "PASS" if (max_dev <= TOLERANCE and corr >= MIN_CORR) else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"[{status}] {key}: 最大偏差 {max_dev:.4f} (容差 {TOLERANCE}), "
              f"超限窗 {n_over}/{len(p)}, 相关系数 {corr:.5f}")
    if failed:
        print("\n回归失败: 如属预期变更, 用 --update-baseline 更新基准后提交")
        return 1
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
