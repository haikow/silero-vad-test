# -*- coding: utf-8 -*-
"""判定四件套规格单测: 每条用例锁定 vad_decision.SpeechSegmenter 的一条语义

窗口 32ms => 16 窗=0.512s(>=0.5 最短语音), 8 窗=0.256s(>=0.25 最短静音), 7 窗=0.224s(不够)
"""
import numpy as np

from vad_decision import SpeechSegmenter

W = 512 / 16000


def seg(probs, **kw):
    s = SpeechSegmenter(**kw)
    s.feed(probs)
    return s


def test_start_requires_high_threshold():
    """0.6 起始阈值: 全程 0.55 (介于 0.35~0.6) 不进入语音"""
    s = seg([0.55] * 200)
    assert s.final_segments() == []


def test_hysteresis_band_sustains_speech():
    """进入语音后, 迟滞带 (0.35~0.6) 内的概率仍算语音持续, 不掉出"""
    probs = [0.9] * 10 + [0.45] * 60   # 起始后 1.92s 都在 0.45
    s = SpeechSegmenter()
    live = s.feed(probs)
    assert live[0] and all(live)          # 始终处于语音
    assert len(s.final_segments()) == 1  # 收尾判成一个长段


def test_short_silence_does_not_close():
    """语音中静音不足 0.25s (7 窗) 后恢复: 合并为一个段, 不拆成两段"""
    probs = [0.9] * 40 + [0.1] * 7 + [0.9] * 20
    assert len(seg(probs).final_segments()) == 1


def test_sustained_silence_closes_segment():
    """静音满 0.25s (8 窗) 关闭当前段, 后续语音开新段"""
    probs = [0.9] * 40 + [0.1] * 8 + [0.9] * 40
    segs = seg(probs).final_segments()
    assert len(segs) == 2


def test_segment_end_excludes_silence_tail():
    """段结束时刻回退到静音起点, 不吞静音尾"""
    probs = [0.9] * 40 + [0.1] * 8       # 语音 [0, 40W), 静音 8 窗
    segs = seg(probs).final_segments()
    assert len(segs) == 1
    assert abs(segs[0][1] - 40 * W) < 1e-9


def test_blip_below_min_speech_discarded():
    """低于最短语音 (5 窗=0.16s 的孤立尖峰) 的候选段丢弃 => 0 误触发"""
    probs = [0.1] * 20 + [0.95] * 5 + [0.1] * 40
    assert seg(probs).final_segments() == []


def test_half_second_segment_kept():
    """满 16 窗 (0.512s >= 0.5s) 的段保留"""
    probs = [0.1] * 10 + [0.95] * 16 + [0.1] * 20
    segs = seg(probs).final_segments()
    assert len(segs) == 1
    assert abs(segs[0][0] - 10 * W) < 1e-9


def test_flush_closes_open_segment_at_stream_end():
    """语音持续到流结束: 收尾时判定成立(不要求出现静音)"""
    probs = [0.1] * 10 + [0.9] * 30
    assert len(seg(probs).final_segments()) == 1


def test_repeated_bursts_yield_segments():
    """三段独立短促语音 (各 ~0.7s, 间隔 >0.25s) => 恰好三个段"""
    burst = [0.9] * 22
    gap = [0.05] * 12
    probs = burst + gap + burst + gap + burst
    assert len(seg(probs).final_segments()) == 3


def test_all_silence_never_fires():
    s = SpeechSegmenter()
    live = s.feed(np.zeros(300))
    assert not any(live) and s.final_segments() == []
