# -*- coding: utf-8 -*-
"""公开数据集场景: LibriSpeech 多说话人语音 sanity + RIR 混响新维度

素材:
  libri_spk_{a,b}_60s.wav — LibriSpeech test-clean(真实多说话人英语朗读)两位说话人,
    各章 utterance 拼 62s session (0.3s 间隔), 峰值归一 0.8
  reverb_{small,large}_room.wav — smoke 语音(峰值0.8) 卷 openslr 28 的 RIR:
    仿真小房间(500ms) / 真实大房间 RVB2014(1000ms, 8通道阵列取第1麦)
标定实测 (2026-10-01):
  libri 覆盖 int8 76/81% (0.3s 间隙拖低全程口径) ; reverb int8 73/77% vs 干净 72%
  —— 混响几乎不伤 VAD(尾音填谷效应, 与 ASR 相反), 新维度基线
"""
import pytest

from conftest import DS_REVERB_FIXTURES, DS_SPEECH_FIXTURES

INT8_LIBRI_COVERAGE_MIN = 0.65      # 实测 76/81% (0.3s 间隙拖低)
INT8_REVERB_COVERAGE_MIN = 0.60     # 实测 73/77%
INT8_LIBRI_MIN_SEGMENTS = 8         # 实测 16/13 段 (多条 utterance)
WRTC_LIBRI_COVERAGE_MIN = 0.85      # 实测 97/97%
FR_LIBRI_COVERAGE_MIN = 0.70        # 实测 80/86%


def test_dataset_fixtures_present():
    from pathlib import Path
    base = Path("test_fixtures")
    for n in DS_SPEECH_FIXTURES + DS_REVERB_FIXTURES:
        assert (base / n).exists(), f"缺 {n}"


@pytest.mark.parametrize("name", DS_SPEECH_FIXTURES)
def test_libri_int8_speakers_detected(name, ds_metrics):
    """真实说话人 sanity: int8 覆盖 >=65% 且检出多段"""
    cov = ds_metrics["ds/libri"][name]["silero-int8"]
    assert cov >= INT8_LIBRI_COVERAGE_MIN, \
        f"int8 在 {name} 覆盖 {cov:.0%} < {INT8_LIBRI_COVERAGE_MIN:.0%}"


def test_libri_int8_multi_segment(musan_engines):
    import soundfile as sf
    from vad_decision import SpeechSegmenter
    audio, _ = sf.read("test_fixtures/libri_spk_a_60s.wav", dtype="float32")
    for ename, run in musan_engines:
        if ename != "silero-int8":
            continue
        probs, shift_s, _fl, thr, st, en = run(audio)
        s = SpeechSegmenter(start=st, end=en, window_s=shift_s)
        s.feed(probs)
        assert len(s.final_segments()) >= INT8_LIBRI_MIN_SEGMENTS


@pytest.mark.parametrize("name", DS_LIBRI_ALL := DS_SPEECH_FIXTURES)
def test_libri_other_engines_sanity(name, ds_metrics):
    d = ds_metrics["ds/libri"][name]
    assert d["WRTC-0"] >= WRTC_LIBRI_COVERAGE_MIN, f"WebRTC {name} {d['WRTC-0']:.0%}"
    assert d["FireRed"] >= FR_LIBRI_COVERAGE_MIN, f"FireRed {name} {d['FireRed']:.0%}"


@pytest.mark.parametrize("name", DS_REVERB_FIXTURES)
def test_reverb_int8_robust(name, ds_metrics):
    """混响不伤 int8 (实测 500ms/1s RIR 覆盖 >=60%, 与干净持平——尾音填谷效应)"""
    cov = ds_metrics["ds/reverb"][name]["silero-int8"]
    assert cov >= INT8_REVERB_COVERAGE_MIN, \
        f"int8 在 {name} 覆盖 {cov:.0%} < {INT8_REVERB_COVERAGE_MIN:.0%}"


def test_reverb_no_engine_collapse(ds_metrics):
    """全引擎混响覆盖 >=55%: 没有引擎因混响崩溃"""
    for name in DS_REVERB_FIXTURES:
        for ename, cov in ds_metrics["ds/reverb"][name].items():
            assert cov >= 0.55, f"{ename} 在 {name} 混响覆盖 {cov:.0%} < 55%"
