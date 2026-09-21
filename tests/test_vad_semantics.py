# -*- coding: utf-8 -*-
"""VAD 语义行为: 不止"和基准一样", 还得"像个人声检测器"

阈值依据 README 实测值留足余量:
  uint8 噪声均值 0.018 / 语音均值 0.803 ; int8 噪声均值 0.102 / 语音均值 0.718
"""
import numpy as np
import pytest

from test_vad import THRESHOLD, WINDOW, SAMPLE_RATE
from test_vad import segments_from_probs

KEYS = ["uint8", "int8"]
SPEECH_MIN_MEAN = 0.5     # 语音段平均概率下限 (实测 0.72~0.80)
NOISE_MAX_MEAN = 0.2      # 噪声段平均概率上限 (实测 0.02~0.10)
MIN_SEPARATION = 0.3      # 语音均值-噪声均值 下限 (实测 0.62~0.79)
OVERLAP_MIN_RATIO = 0.6   # 检出语音与真实语音区间的重覆盖率下限
LEAK_MAX_SECONDS = 1.5    # 检出语音漏到纯噪声区的总时长上限 (边界过渡 + 判定迟滞)


@pytest.mark.parametrize("key", KEYS)
def test_speech_segment_mean_prob(key, probs, speech_mask):
    p = np.asarray(probs[key])[speech_mask]
    assert p.mean() >= SPEECH_MIN_MEAN, \
        f"{key}: 语音段平均概率 {p.mean():.4f} < {SPEECH_MIN_MEAN}, 人声检不出"


@pytest.mark.parametrize("key", KEYS)
def test_noise_segment_mean_prob(key, probs, speech_mask):
    p = np.asarray(probs[key])[~speech_mask]
    assert p.mean() <= NOISE_MAX_MEAN, \
        f"{key}: 噪声段平均概率 {p.mean():.4f} > {NOISE_MAX_MEAN}, 静音误报"


@pytest.mark.parametrize("key", KEYS)
def test_speech_noise_separation(key, probs, speech_mask):
    p = np.asarray(probs[key])
    sep = p[speech_mask].mean() - p[~speech_mask].mean()
    assert sep >= MIN_SEPARATION, \
        f"{key}: 区分度 {sep:.4f} < {MIN_SEPARATION}, 概率曲线拉不开"


@pytest.mark.parametrize("key", KEYS)
def test_detected_speech_overlaps_ground_truth(key, probs, audio, speech_mask):
    """阈值 0.5 检出的语音区间应覆盖大部分真实语音区, 且不明显漏进噪声区"""
    a, _ = audio
    n = len(probs[key])
    times = np.arange(n) * WINDOW / SAMPLE_RATE
    segs = segments_from_probs(probs[key], times, THRESHOLD)
    assert segs, f"{key}: 阈值 {THRESHOLD} 下未检出任何语音区间"

    truth = (speech_mask.nonzero()[0][0] * WINDOW / SAMPLE_RATE,
             speech_mask.nonzero()[0][-1] * WINDOW / SAMPLE_RATE + WINDOW / SAMPLE_RATE)
    overlap = sum(max(0.0, min(e, truth[1]) - max(s, truth[0])) for s, e in segs)
    leak = sum(max(0.0, max(0.0, truth[0] - s)) + max(0.0, e - truth[1]) for s, e in segs)
    ratio = overlap / (truth[1] - truth[0])
    assert ratio >= OVERLAP_MIN_RATIO, \
        f"{key}: 检出区间仅覆盖真实语音 {ratio:.0%} < {OVERLAP_MIN_RATIO:.0%}, 漏检"
    assert leak <= LEAK_MAX_SECONDS, \
        f"{key}: 检出区间漏进噪声区 {leak:.2f}s > {LEAK_MAX_SECONDS}s, 误检"
