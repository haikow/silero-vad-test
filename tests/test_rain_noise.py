# -*- coding: utf-8 -*-
"""真实雨声场景: 稳态雨噪误报 + 雨中轻声旁白的漏检对比 (与咖啡馆互补的假阴性故事)

素材 (B 站 BV1tkohB5Eqc, UP 云南山师傅, 2026-09-22):
  rain_noise_16k.wav  90-150s, 60s 雨打树叶+采菌脆响 (稳态宽带+瞬态冲击)
  rain_speech_16k.wav 25-62s, 37s 治愈系轻声旁白埋在雨声下 (~-10dB SNR, 峰值 0.29)
实测依据 (2026-09-22, Mac M3, ORT CPU):
  裸雨声: 两模型均值 0.001/0.023, 四件套误触发均 0 段 —— 雨声+瞬态脆响骗不动 VAD
  雨中旁白: int8 检出 5 段, 活动窗覆盖 55%, 交叠 69%; uint8 均值 0.004, 全漏 (0 段)
  注: 已排除电平因果 (放大到峰值 0.8 仍全漏), 是低 SNR 假阴性 —— 与合成白噪声
  SNR 阶梯、咖啡馆误报共同构成 uint8 的双向短板证据链
"""
import pytest

# ---- 裸雨声误报 ----
RAIN_MAX_TRIGGERS = 1        # 实测 uint8/int8 均 0 段/60s
RAIN_UINT8_MEAN_MAX = 0.05   # 实测 0.001
RAIN_INT8_MEAN_MAX = 0.10    # 实测 0.023

# ---- 雨中轻声旁白 (真语音, 低 SNR) ----
RAIN_INT8_MIN_SEGMENTS = 3   # 实测 5 段
RAIN_INT8_COVERAGE_MIN = 0.40   # 实测 0.55 (活动窗内 >=0.5 占比)
RAIN_INT8_OVERLAP_MIN = 0.50    # 实测 0.69 (段与活动区交叠比)
RAIN_UINT8_MISS_MEAN_MAX = 0.05  # 实测 0.004 (特征化: 假阴性)
RAIN_UINT8_MISS_COVERAGE_MAX = 0.05


def test_rain_fixtures_present():
    import soundfile as sf
    for name in ("rain_noise_16k.wav", "rain_speech_16k.wav"):
        info = sf.info(f"test_fixtures/{name}")
        assert info.samplerate == 16000 and info.channels == 1, name


@pytest.mark.parametrize("key", ["uint8", "int8"])
def test_rain_no_false_triggers(key, rain_noise_stats):
    """雨打树叶+采菌脆响: 四件套误触发 <=1 段/60s (实测 0)"""
    n = rain_noise_stats[key]["triggers"]
    assert n <= RAIN_MAX_TRIGGERS, f"{key} 雨声误触发 {n} 段 > {RAIN_MAX_TRIGGERS}"


def test_rain_quiet_mean(rain_noise_stats):
    m = rain_noise_stats["uint8"]["mean"]
    assert m <= RAIN_UINT8_MEAN_MAX, f"uint8 雨声均值 {m:.3f} > {RAIN_UINT8_MEAN_MAX}"
    m = rain_noise_stats["int8"]["mean"]
    assert m <= RAIN_INT8_MEAN_MAX, f"int8 雨声均值 {m:.3f} > {RAIN_INT8_MEAN_MAX}"


def test_rain_speech_detected_by_int8(rain_speech_stats):
    """产品断言: 雨中轻声旁白 int8 仍能检出大部分 (实测 5 段/覆盖 55%/交叠 69%)"""
    s = rain_speech_stats["int8"]
    assert s["segments"] >= RAIN_INT8_MIN_SEGMENTS, \
        f"int8 仅检出 {s['segments']} 段 < {RAIN_INT8_MIN_SEGMENTS}"
    assert s["coverage"] >= RAIN_INT8_COVERAGE_MIN, \
        f"int8 活动窗覆盖 {s['coverage']:.0%} < {RAIN_INT8_COVERAGE_MIN:.0%}"
    assert s["overlap"] >= RAIN_INT8_OVERLAP_MIN, \
        f"int8 段交叠 {s['overlap']:.0%} < {RAIN_INT8_OVERLAP_MIN:.0%}"


def test_rain_speech_missed_by_uint8(rain_speech_stats):
    """特征化: uint8(v5) 对雨中轻声真人旁白整体漏检 (实测 0 段, 均值 0.004)。
    与电平无关(放大 0.8 峰值仍漏), 系低 SNR 假阴性 —— 唤醒类产品的漏唤醒风险。
    模型更换时此断言提醒重标定。"""
    s = rain_speech_stats["uint8"]
    assert s["segments"] == 0, f"uint8 检出 {s['segments']} 段: 假阴性特性变了, 请重标定"
    assert s["mean"] <= RAIN_UINT8_MISS_MEAN_MAX, \
        f"uint8 均值 {s['mean']:.3f} > {RAIN_UINT8_MISS_MEAN_MAX}"
    assert s["coverage"] <= RAIN_UINT8_MISS_COVERAGE_MAX
