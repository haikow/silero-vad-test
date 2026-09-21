# -*- coding: utf-8 -*-
"""真实咖啡馆 babble 场景 (Tier 2): 部署口径的核心指标

素材: test_fixtures/cafe_noise_16k.wav (60s, B 站 BV1eW41137G7 的 1:30-2:30,
人群嘈杂人声 babble —— VAD 最难干扰, 白噪声/粉噪声模拟不了)。
实测依据 (2026-09-21, Mac M3, ORT CPU):
  裸噪声: uint8 均值 0.235, 四件套误触发 9 段/60s ; int8 0.172, 0 段/60s
  SNR 混合(语音峰值0.3): int8 10/5/0dB = 0.94/0.96/0.92 (覆盖 98%+)
                         uint8 10/5/0dB = 0.32/0.12/0.07 (深噪声下趴掉)
"""
import pytest

from scenario_audio import padded, speech_from_fixture, stream_probs, region_mask
from test_vad import WINDOW
from vad_decision import SpeechSegmenter

CAFE_SNR_STEPS_DB = (10, 5, 0)

# ---- 裸噪声误触发 (部署口径: 四件套判定) ----
CAFE_INT8_MAX_TRIGGERS = 3    # 实测 0/60s, 留跨平台余量
CAFE_INT8_MEAN_MAX = 0.22     # 实测 0.172
CAFE_UINT8_TRIGGERS_MIN = 3   # 实测 9: babble 确实迷惑 v5 (特征化)
CAFE_UINT8_TRIGGERS_MAX = 15
CAFE_UINT8_MEAN_MIN = 0.15    # 实测 0.235 (特征化)

# ---- 咖啡馆噪声下 SNR (语音峰值 0.3) ----
CAFE_INT8_10DB_MEAN_MIN = 0.88   # 实测 0.943
CAFE_INT8_5DB_MEAN_MIN = 0.90    # 实测 0.960
CAFE_INT8_0DB_MEAN_MIN = 0.85    # 实测 0.921 —— 真实嘈杂环境产品底线
CAFE_INT8_COVERAGE_MIN = 0.90    # 实测 98% @0dB
CAFE_UINT8_10DB_MEAN_MIN = 0.25  # 实测 0.317 (尚可底线)
CAFE_UINT8_5DB_MEAN_MAX = 0.20   # 实测 0.119 (特征化: babble 下衰减)
CAFE_UINT8_0DB_MEAN_MAX = 0.15   # 实测 0.069 (特征化: 深噪声趴掉)
CAFE_INT8_ADVANTAGE_0DB_MIN = 0.60  # 实测 0.852


def test_cafe_fixture_present(cafe_audio):
    import soundfile as sf
    info = sf.info("test_fixtures/cafe_noise_16k.wav")
    assert info.samplerate == 16000 and info.channels == 1


def test_cafe_int8_no_false_triggers(cafe_raw):
    """产品断言: int8 + 四件套在真实咖啡馆 babble 下 60s 误触发 <= 3 段 (实测 0)"""
    n = cafe_raw["int8"]["triggers"]
    assert n <= CAFE_INT8_MAX_TRIGGERS, \
        f"int8 误触发 {n} 段/60s > {CAFE_INT8_MAX_TRIGGERS}, 咖啡馆场景退化"


def test_cafe_int8_prob_mean(cafe_raw):
    m = cafe_raw["int8"]["mean"]
    assert m <= CAFE_INT8_MEAN_MAX, f"int8 均值 {m:.3f} > {CAFE_INT8_MEAN_MAX}"


def test_cafe_uint8_babble_confusion(cafe_raw):
    """特征化: uint8(v5) 被 babble 迷惑 (实测 9 段/60s)。模型更换时此断言提醒重标定"""
    n = cafe_raw["uint8"]["triggers"]
    m = cafe_raw["uint8"]["mean"]
    assert CAFE_UINT8_TRIGGERS_MIN <= n <= CAFE_UINT8_TRIGGERS_MAX, \
        f"uint8 误触发 {n} 段/60s, 超出特征化区间 [{CAFE_UINT8_TRIGGERS_MIN},{CAFE_UINT8_TRIGGERS_MAX}]"
    assert m >= CAFE_UINT8_MEAN_MIN, f"uint8 均值 {m:.3f} < {CAFE_UINT8_MEAN_MIN}, 迷惑特性变了"


@pytest.mark.parametrize("snr_db", CAFE_SNR_STEPS_DB)
def test_cafe_snr_int8_detects(snr_db, cafe_snr):
    """真实嘈杂环境底线: int8 在咖啡馆 0dB SNR 仍 >=85% 均值 / 90% 覆盖"""
    m = cafe_snr[snr_db]["int8"]["mean"]
    cov = cafe_snr[snr_db]["int8"]["coverage"]
    floor = {10: CAFE_INT8_10DB_MEAN_MIN, 5: CAFE_INT8_5DB_MEAN_MIN,
             0: CAFE_INT8_0DB_MEAN_MIN}[snr_db]
    assert m >= floor, f"int8 @café {snr_db}dB: 均值 {m:.3f} < {floor}"
    assert cov >= CAFE_INT8_COVERAGE_MIN, f"int8 @café {snr_db}dB: 覆盖 {cov:.0%} < {CAFE_INT8_COVERAGE_MIN:.0%}"


def test_cafe_snr_uint8_floor_10db(cafe_snr):
    m = cafe_snr[10]["uint8"]["mean"]
    assert m >= CAFE_UINT8_10DB_MEAN_MIN, f"uint8 @café 10dB: {m:.3f} < {CAFE_UINT8_10DB_MEAN_MIN}"


@pytest.mark.parametrize("snr_db,hi", [(5, CAFE_UINT8_5DB_MEAN_MAX), (0, CAFE_UINT8_0DB_MEAN_MAX)])
def test_cafe_snr_uint8_fades(snr_db, hi, cafe_snr):
    """特征化: uint8 在咖啡馆深噪声下衰减/趴掉 (实测 0.119/0.069)"""
    m = cafe_snr[snr_db]["uint8"]["mean"]
    assert m <= hi, f"uint8 @café {snr_db}dB 均值 {m:.3f} > {hi}, 抗 babble 改善了? 请重标定"


def test_cafe_snr_int8_advantage_0db(cafe_snr):
    """真实咖啡馆 0dB: int8 显著优于 uint8 (实测差 0.852) —— 移植选型最终依据"""
    adv = cafe_snr[0]["int8"]["mean"] - cafe_snr[0]["uint8"]["mean"]
    assert adv >= CAFE_INT8_ADVANTAGE_0DB_MIN, \
        f"0dB 咖啡馆 int8 优势仅 {adv:.3f} < {CAFE_INT8_ADVANTAGE_0DB_MIN}"
