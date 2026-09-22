# -*- coding: utf-8 -*-
"""把 scenario_metrics.json 渲染成 markdown, 写入 $GITHUB_STEP_SUMMARY (run 页摘要区)

用法: python ci/render_metrics.py            # 打印到 stdout
      python ci/render_metrics.py --summary  # 追加写入 GITHUB_STEP_SUMMARY
本地: python ci/render_metrics.py 随时看最新指标表
"""
import json
import os
import sys
from pathlib import Path

METRICS = Path(__file__).resolve().parent.parent / "scenario_metrics.json"


def fmt(v):
    """按指标形状渲染: 裸浮点 / mean+triggers / 雨旁白含段数 / mean+coverage"""
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    if "mean" in v and "segments" in v:
        return (f"{v['mean']:.3f} / {v['segments']} 段 / "
                f"覆盖 {v['coverage']:.0%}")
    if "mean" in v and "triggers" in v:
        return f"{v['mean']:.3f} / {v['triggers']} 段"
    if "mean" in v and "coverage" in v:
        return f"{v['mean']:.3f} / {v['coverage']:.0%}"
    return str(v)


def sort_key(k, prefix):
    """数字后缀的项按数值降序 (SNR 20/10/5), 其余字母序"""
    digits = "".join(c for c in k[len(prefix):] if c.isdigit() or c == ".")
    try:
        return (0, -float(digits))
    except ValueError:
        return (1, 0)


def quick_compare(d):
    """头部速览: 两个模型拉开差距的关键指标 (缺失则跳过该行)"""
    rows = [
        ("咖啡馆误触发 (段/60s)", "cafe/raw", lambda v: f"{v['triggers']} 段 / {v['false_speech_s']:.1f}s"),
        ("雨中轻声旁白检出", "rain/speech", lambda v: f"{v['segments']} 段 / 覆盖 {v['coverage']:.0%}"),
        ("咖啡馆 0dB SNR 检出", "cafe/snr/0dB", lambda v: f"{v['mean']:.2f} / {v['coverage']:.0%}"),
        ("白噪 5dB SNR 检出", "snr/5dB", lambda v: f"{v['mean']:.2f} / {v['coverage']:.0%}"),
        ("低电平 (peak 0.05) 语音均值", "level/peak=0.05", lambda v: fmt(v)),
    ]
    lines = ["### 关键对比速览 (uint8/v5 vs int8/v4)", "",
             "| 指标 | uint8 (v5) | int8 (v4) |", "|---|---|---|"]
    for label, key, f in rows:
        if key in d and all(k in d[key] for k in ("uint8", "int8")):
            lines.append(f"| {label} | {f(d[key]['uint8'])} | {f(d[key]['int8'])} |")
    return lines


GROUPS = [
    ("电平矩阵 (语音段均值, peak=)", "level/peak=", "电平"),
    ("合成噪声误报 (概率均值)", "noise/", "噪声"),
    ("白噪 SNR 阶梯 (均值 / 覆盖)", "snr/", "SNR"),
    ("真实咖啡馆 (均值 / 误触发 或 均值 / 覆盖)", "cafe/", "场景"),
    ("真实雨声", "rain/", "场景"),
]


def render():
    d = json.loads(METRICS.read_text(encoding="utf-8"))["scenarios"]
    out = ["## VAD 场景指标总览", "",
           "逐窗断言见上方用例区; 本表为各场景实测值 (阈值详情在 tests/ 对应文件头)。", ""]
    out += quick_compare(d) + [""]
    for title, prefix, first_col in GROUPS:
        keys = sorted((k for k in d if k.startswith(prefix)),
                      key=lambda k: sort_key(k, prefix))
        if not keys:
            continue
        out += [f"### {title}", "", f"| {first_col} | uint8 (v5) | int8 (v4) |", "|---|---|---|"]
        for k in keys:
            name = k[len(prefix):] or k
            out.append(f"| {name} | {fmt(d[k]['uint8'])} | {fmt(d[k]['int8'])} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    text = render()
    if "--summary" in sys.argv and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print("已写入 run 页摘要")
    else:
        print(text)
