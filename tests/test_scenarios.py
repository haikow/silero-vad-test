# -*- coding: utf-8 -*-
"""场景矩阵: 不同声学条件下的 VAD 表现 (阈值按 2026-09-21 实测留余量标定)

三条曲线 fixture 在 conftest (session 级, 只推理一次):
  level_curve  电平矩阵   noise_table  噪声误报   snr_curve  SNR 阶梯
实测依据 (Mac M3, onnxruntime CPU, 种子固定; SNR 按缩放后语音 RMS 定标):
  level: int8 各档 0.727~0.828 / uint8 0.782(peak0.8) -> 0.461(peak0.05)
  noise: 全部 0.001~0.059
  snr:   int8 0.934/0.921/0.885 @20/10/5dB (检出覆盖 95%/94%/93%)
         uint8 0.676/0.556/0.336 (71%/57%/30%) —— 深噪声下渐退, int8 全面占优
"""
import pytest

from vad_decision import SpeechSegmenter
from scenario_audio import noise_clip, stream_probs

# ---- 电平矩阵 ----
INT8_ROBUST_MIN = 0.70    # int8 各档实测下限 0.727 (电平鲁棒 = 嵌入式拾音余量大)
UINT8_DROP_MIN = 0.25     # uint8 高低电平差实测 0.321 (README"低电平塌"结论)
UINT8_LOW_MAX = 0.55      # uint8 @peak0.05 实测 0.461

# ---- 噪声误报 ----
NOISE_MAX_MEAN = 0.15     # 6 组实测最大 0.059, 余量充足

# ---- SNR 阶梯 (嘈杂环境产品底线: int8 在 5dB SNR 仍 93% 检出) ----
SNR_INT8_20DB_MIN = 0.90  # 实测 0.934
SNR_INT8_10DB_MIN = 0.88  # 实测 0.921
SNR_INT8_5DB_MIN = 0.82   # 实测 0.885
SNR_UINT8_20DB_MIN = 0.55 # 实测 0.676 (干净环境底线)
SNR_UINT8_5DB_MAX = 0.45  # 实测 0.336 (特征化: 深噪声渐退; 模型更换时此断言报警)
SNR_INT8_ADVANTAGE_MIN = 0.30  # 实测 5dB 差 0.549: 嘈杂环境 int8 显著占优


@pytest.mark.parametrize("peak", [0.05, 0.1, 0.3, 0.8])
def test_level_int8_robust_across_levels(peak, level_curve):
    """int8 (移植蓝本) 全电平档位语音段均值 >= 0.70: 拾音电平不用精调"""
    m = level_curve[peak]["int8"]
    assert m >= INT8_ROBUST_MIN, f"int8 @peak={peak}: {m:.3f} < {INT8_ROBUST_MIN}"


def test_level_uint8_degrades_at_low_level(level_curve):
    """uint8 (v5) 低电平显著衰减 —— README 手工结论的自动化锁定"""
    drop = level_curve[0.8]["uint8"] - level_curve[0.05]["uint8"]
    low = level_curve[0.05]["uint8"]
    assert drop >= UINT8_DROP_MIN, \
        f"uint8 高低电平差 {drop:.3f} < {UINT8_DROP_MIN}, 电平敏感特性变了"
    assert low <= UINT8_LOW_MAX, f"uint8 @peak=0.05 均值 {low:.3f} > {UINT8_LOW_MAX}"


@pytest.mark.parametrize("kind,rms", [
    ("white", 0.05), ("white", 0.15),
    ("pink", 0.05), ("pink", 0.15),
    ("brown", 0.05), ("brown", 0.15),
])
def test_noise_false_alarm_mean(kind, rms, noise_table):
    """合成有色噪声全程平均概率 <= 0.15: 两模型都不误报"""
    p = noise_table[(kind, rms)]
    for k, v in p.items():
        assert v <= NOISE_MAX_MEAN, f"{k} on {kind} rms={rms}: {v:.3f} > {NOISE_MAX_MEAN}"


def test_noise_no_fired_segments(runners):
    """集成: 白噪声(rms 0.15)经四件套判定 => 0 个误触发段 (部署口径)"""
    probs = stream_probs(runners["int8"], noise_clip("white", 5.0, 0.15))
    segs = SpeechSegmenter()
    segs.feed(probs)
    assert segs.final_segments() == []


def test_snr_int8_detects_20db(snr_curve):
    m = snr_curve[20]["int8"]["mean"]
    assert m >= SNR_INT8_20DB_MIN, f"int8 @20dB: {m:.3f} < {SNR_INT8_20DB_MIN}"


def test_snr_int8_detects_10db(snr_curve):
    m = snr_curve[10]["int8"]["mean"]
    assert m >= SNR_INT8_10DB_MIN, f"int8 @10dB: {m:.3f} < {SNR_INT8_10DB_MIN}"


def test_snr_int8_detects_5db(snr_curve):
    """嘈杂环境产品底线: int8 (移植蓝本) 在 5dB SNR 仍稳定检出 (实测覆盖 93%)"""
    m = snr_curve[5]["int8"]["mean"]
    assert m >= SNR_INT8_5DB_MIN, f"int8 @5dB: {m:.3f} < {SNR_INT8_5DB_MIN}"


def test_snr_uint8_clean_floor_20db(snr_curve):
    m = snr_curve[20]["uint8"]["mean"]
    assert m >= SNR_UINT8_20DB_MIN, f"uint8 @20dB: {m:.3f} < {SNR_UINT8_20DB_MIN}"


def test_snr_uint8_fades_in_deep_noise_5db(snr_curve):
    """特征化: uint8 深噪声渐退 (实测 0.336)。模型更换/升级时此断言提醒重标定"""
    m = snr_curve[5]["uint8"]["mean"]
    assert m <= SNR_UINT8_5DB_MAX, \
        f"uint8 @5dB 均值 {m:.3f} > {SNR_UINT8_5DB_MAX}: 抗噪改善了, 请更新阈值文档"


def test_snr_int8_advantage_in_deep_noise(snr_curve):
    """深噪声下 int8 显著优于 uint8 (实测 5dB 差 0.549) —— 移植选型依据"""
    adv = snr_curve[5]["int8"]["mean"] - snr_curve[5]["uint8"]["mean"]
    assert adv >= SNR_INT8_ADVANTAGE_MIN, f"5dB 下 int8 优势仅 {adv:.3f} < {SNR_INT8_ADVANTAGE_MIN}"


@pytest.mark.parametrize("key", ["uint8", "int8"])
def test_snr_monotonic_degradation(key, snr_curve):
    """SNR 下降 => 语音段均值单调下降 (20dB > 10dB > 5dB)"""
    means = [snr_curve[s][key]["mean"] for s in (20, 10, 5)]
    assert means[0] > means[1] > means[2], f"{key} 非单调递减: {means}"
