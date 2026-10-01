# -*- coding: utf-8 -*-
"""MUSAN 真实素材场景 (选自 106 文件全库扫描的代表点 + 音乐 + 多说话人朗读)

fixture 选择依据 ci/musan_sweep.py 的逐文件触发矩阵 (2026-10-01 扫描, 全库 843 个
noise 文件抽 106 个跑 7 引擎): 6 个噪声覆盖误报分布的各层(全免疫/仅WebRTC/WebRTC+FireRed/
WebRTC+int8/WebRTC+uint8/全员中招), 2 个流派音乐(古典/电子), 1 个 LibriVox 朗读 sanity。
库规模 142h(music 42h + speech 60h + noise 930 文件), CC-BY; 官方 openslr.org/17。

分布结论 (全库扫描): WebRTC 89~95% 文件误触发(帧误报均值 0.65~0.81) / FireRed 26%
(median 0.005 但 p95 0.43) / int8 21% / uint8 14%(全场最低——高查准画像再证)。
"""
import pytest

from conftest import (MUSAN_MUSIC_FIXTURES, MUSAN_NOISE_FIXTURES,
                      MUSAN_SPEECH_FIXTURE)

# ---- 产品断言 (int8 = 固件蓝本) ----
INT8_NOISE_MAX_SEGMENTS_EACH = 1   # 6 个噪声 fixture 上单个 <=1 段 (实测最多 1)
INT8_NOISE_FP_MEAN_MAX = 0.10      # 6 件噪声平均帧误报率 (实测 ~0.04)
INT8_CLASSICAL_SEGMENTS = 0        # 古典乐实测 0 段
INT8_JAMENDO_MAX_SEGMENTS = 3      # 电子乐实测 2 段/60s (音乐常含人声/节拍)
INT8_SPEECH_COVERAGE_MIN = 0.50    # 朗读检出覆盖 (实测 64%)

# ---- 特征化 (其余引擎, 失败=行为变化需重标定) ----
FR_CLASSICAL_SEGMENTS = 0            # FireRed 古典乐实测 0 段
FR_JAMENDO_MAX_SEGMENTS = 9          # 实测 7 段(全场最差, 音乐含人声时 FireRed 软肋)
FR_HARD_MAX_SEGMENTS = 4             # hard 噪声实测 2 段
U8_JAMENDO_FP_MAX = 0.30             # 实测 20%
WRTC_MUSIC_FP_MIN = 0.90             # 音乐上 WebRTC 全开 (实测 100%)


@pytest.fixture(scope="session")
def per_fixture(musan_engines):
    """逐 fixture 逐引擎 (fp_rate, triggers)"""
    import soundfile as sf
    from conftest import _fourpiece
    out = {}
    for f in sorted(list(MUSAN_NOISE_FIXTURES) + list(MUSAN_MUSIC_FIXTURES)):
        audio, _ = sf.read(f, dtype="float32")
        out[f.name] = {name: _fourpiece(run, audio) for name, run in musan_engines}
    return out


# ---- 素材完整性 ----
def test_musan_fixtures_present():
    n_noise, n_music = len(MUSAN_NOISE_FIXTURES), len(MUSAN_MUSIC_FIXTURES)
    assert n_noise == 6 and n_music == 2, f"fixture 数变了: noise={n_noise} music={n_music}"
    assert MUSAN_SPEECH_FIXTURE.exists()


# ---- 产品断言: int8 ----
@pytest.mark.parametrize("f", MUSAN_NOISE_FIXTURES, ids=lambda p: p.stem)
def test_musan_int8_noise_quiet(f, per_fixture):
    n = per_fixture[f.name]["silero-int8"][1]
    assert n <= INT8_NOISE_MAX_SEGMENTS_EACH, \
        f"int8 在 {f.name} 误触发 {n} 段 > {INT8_NOISE_MAX_SEGMENTS_EACH}"


def test_musan_int8_noise_fp_mean_agg(musan_metrics):
    fp = musan_metrics["musan/noise"]["silero-int8"]["fp_mean"]
    assert fp <= INT8_NOISE_FP_MEAN_MAX, f"int8 噪声平均帧误报 {fp:.3f} > {INT8_NOISE_FP_MEAN_MAX}"


def test_musan_int8_classical(per_fixture):
    n = per_fixture["musan_music_classical_60s.wav"]["silero-int8"][1]
    assert n <= INT8_CLASSICAL_SEGMENTS, f"int8 古典乐误触发 {n} 段"


def test_musan_int8_jamendo(per_fixture):
    n = per_fixture["musan_music_jamendo_60s.wav"]["silero-int8"][1]
    assert n <= INT8_JAMENDO_MAX_SEGMENTS, f"int8 电子乐误触发 {n} 段 > {INT8_JAMENDO_MAX_SEGMENTS}"


def test_musan_int8_speech_coverage(musan_metrics):
    cov = musan_metrics["musan/speech"]["silero-int8"]
    assert cov >= INT8_SPEECH_COVERAGE_MIN, f"int8 朗读覆盖 {cov:.0%} < {INT8_SPEECH_COVERAGE_MIN:.0%}"


# ---- 特征化锁定 ----
def test_musan_firered_classical(per_fixture):
    n = per_fixture["musan_music_classical_60s.wav"]["FireRed"][1]
    assert n == FR_CLASSICAL_SEGMENTS, f"FireRed 古典乐 {n} 段 (基准 0), 行为变了"


def test_musan_firered_jamendo(per_fixture):
    n = per_fixture["musan_music_jamendo_60s.wav"]["FireRed"][1]
    assert n <= FR_JAMENDO_MAX_SEGMENTS, \
        f"FireRed 电子乐 {n} 段 > {FR_JAMENDO_MAX_SEGMENTS}: 音乐人声鲁棒性变了, 请重标定"


def test_musan_firered_hard(per_fixture):
    n = per_fixture["musan_noise_hard.wav"]["FireRed"][1]
    assert n <= FR_HARD_MAX_SEGMENTS, f"FireRed hard 噪声 {n} 段 > {FR_HARD_MAX_SEGMENTS}"


def test_musan_u8_jamendo_fp(per_fixture):
    fp = per_fixture["musan_music_jamendo_60s.wav"]["silero-uint8"][0]
    assert fp <= U8_JAMENDO_FP_MAX, f"uint8 电子乐帧误报 {fp:.0%} > {U8_JAMENDO_FP_MAX}"


@pytest.mark.parametrize("mus", ["musan_music_classical_60s.wav", "musan_music_jamendo_60s.wav"])
def test_musan_wrtc_music_blind(mus, per_fixture):
    """特征化: WebRTC 对音乐全开 (帧误报 >=90%)"""
    fp = per_fixture[mus]["WRTC-0"][0]
    assert fp >= WRTC_MUSIC_FP_MIN, f"WebRTC 在 {mus} 帧误报 {fp:.0%} < {WRTC_MUSIC_FP_MIN}"


def test_musan_sweep_consistency(musan_metrics):
    """easy 噪声上 WebRTC 之外的引擎全 0 段 (fixture 选择的自我校验)"""
    assert musan_metrics["musan/noise"]["silero-int8"]["segments"] <= 6
    assert musan_metrics["musan/noise"]["FireRed"]["segments"] <= 10
