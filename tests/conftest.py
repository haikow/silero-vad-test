# -*- coding: utf-8 -*-
"""共享 fixtures: 素材/基准/推理结果 session 级只算一次, 用例零重复开销"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from test_vad import MODELS, SAMPLE_RATE, WINDOW, VadRunner  # noqa: E402

FIXTURE = BASE / "test_fixtures" / "smoke_test.wav"
BASELINE = BASE / "test_fixtures" / "baseline_probs.json"

# 素材固定结构: 白噪声(1.5s) + 语音 + 白噪声(1.5s), 见 test_vad.prepare_audio()
EDGE_SECONDS = 1.5

# "uint8"/"int8" 短名 -> 模型路径 (MODELS 键名带备注, 不能直接用)
# 注意: "uint8" 字符串包含子串 "int8", 匹配必须用 startswith, 否则会把两个模型配反
MODEL_PATHS = {
    key: str(p)
    for key, p in (
        ("uint8", next(p for n, p in MODELS.items() if n.startswith("uint8"))),
        ("int8", next(p for n, p in MODELS.items() if n.startswith("int8"))),
    )
}
assert len(set(MODEL_PATHS.values())) == 2, "两个模型路径相同, 匹配逻辑坏了"


def infer_probs(audio) -> dict:
    """整段流式推理, 返回 {uint8: [p...], int8: [p...]} (与 ci/run_tests.py 同口径)"""
    out = {}
    for key, path in MODEL_PATHS.items():
        r = VadRunner(path)
        out[key] = [float(r.process(audio[i:i + WINDOW]))
                    for i in range(0, len(audio) - WINDOW, WINDOW)]
    return out


# ---------- 场景套件 (test_scenarios.py 用): session 级复用 runner, 指标落盘 ----------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_audio as sa  # noqa: E402

LEVEL_PEAKS = (0.05, 0.1, 0.3, 0.8)
NOISE_KINDS = ("white", "pink", "brown")
NOISE_RMS = (0.05, 0.15)
SNR_STEPS_DB = (20, 10, 5)
SPEECH_PAD_S = 0.75


@pytest.fixture(scope="session")
def runners():
    """每个模型一个常驻 runner (ONNX 会话只建一次; 用前必须 reset, sa.stream_probs 负责)"""
    return {k: VadRunner(p) for k, p in MODEL_PATHS.items()}


@pytest.fixture(scope="session")
def level_curve(runners):
    """电平矩阵: {peak: {model: 语音段平均概率}}"""
    curve = {}
    for peak in LEVEL_PEAKS:
        x = sa.padded(sa.speech_from_fixture(peak), SPEECH_PAD_S)
        curve[peak] = {k: float(sa.stream_probs(r, x)[
            sa.region_mask(len(range(0, len(x) - WINDOW, WINDOW)),
                           SPEECH_PAD_S, SPEECH_PAD_S + len(x) / 16000 - 2 * SPEECH_PAD_S
                           )].mean()) for k, r in runners.items()}
        sa.METRICS["scenarios"][f"level/peak={peak}"] = curve[peak]
    return curve


@pytest.fixture(scope="session")
def noise_table(runners):
    """噪声误报: {(kind, rms): {model: 全程平均概率}}"""
    table = {}
    for kind in NOISE_KINDS:
        for rms in NOISE_RMS:
            x = sa.noise_clip(kind, 5.0, rms)
            p = {k: float(sa.stream_probs(r, x)[5:].mean()) for k, r in runners.items()}
            table[(kind, rms)] = p
            sa.METRICS["scenarios"][f"noise/{kind}/rms={rms}"] = p
    return table


@pytest.fixture(scope="session")
def snr_curve(runners):
    """SNR 阶梯: {snr_db: {model: {"mean": 语音段均值, "coverage": >=0.5 检出覆盖}}}"""
    curve = {}
    for snr_db in SNR_STEPS_DB:
        x = sa.snr_mix(speech_peak=0.3, snr_db=snr_db)
        mask = sa.region_mask(len(range(0, len(x) - WINDOW, WINDOW)),
                              SPEECH_PAD_S, len(x) / 16000 - SPEECH_PAD_S)
        entry = {}
        for k, r in runners.items():
            p = sa.stream_probs(r, x)[mask]
            entry[k] = {"mean": round(float(p.mean()), 4),
                        "coverage": round(float((p >= 0.5).mean()), 4)}
        curve[snr_db] = entry
        sa.METRICS["scenarios"][f"snr/{snr_db}dB"] = entry
    return curve


# ---------- 真实咖啡馆 babble 场景 (test_cafe_noise.py) ----------
CAFE_FIXTURE = BASE / "test_fixtures" / "cafe_noise_16k.wav"
CAFE_SNR_STEPS_DB = (10, 5, 0)


@pytest.fixture(scope="session")
def cafe_audio():
    audio, sr = sf.read(CAFE_FIXTURE, dtype="float32")
    assert sr == SAMPLE_RATE, "咖啡馆素材不是 16k"
    return audio


@pytest.fixture(scope="session")
def cafe_raw(runners, cafe_audio):
    """裸咖啡馆噪声: 概率均值 + 四件套误触发段数 (部署口径)"""
    from vad_decision import SpeechSegmenter
    raw = {}
    for k, r in runners.items():
        p = sa.stream_probs(r, cafe_audio)
        seg = SpeechSegmenter(); seg.feed(p)
        segs = seg.final_segments()
        raw[k] = {"mean": round(float(p.mean()), 4),
                  "triggers": len(segs),
                  "false_speech_s": round(sum(e - s for s, e in segs), 2)}
    sa.METRICS["scenarios"]["cafe/raw"] = raw
    return raw


@pytest.fixture(scope="session")
def cafe_snr(runners, cafe_audio):
    """真实咖啡馆噪声下的 SNR 混合: {snr_db: {model: {mean, coverage}}}"""
    curve = {}
    speech = sa.speech_from_fixture(0.3)
    for snr_db in CAFE_SNR_STEPS_DB:
        noise = (cafe_audio[:len(speech)]
                 * (speech.std() / (cafe_audio.std() + 1e-9) / 10 ** (snr_db / 20)))
        x = sa.padded(speech + noise.astype(np.float32))
        mask = sa.region_mask(len(range(0, len(x) - WINDOW, WINDOW)),
                              0.75, len(x) / SAMPLE_RATE - 0.75)
        entry = {}
        for k, r in runners.items():
            p = sa.stream_probs(r, x)[mask]
            entry[k] = {"mean": round(float(p.mean()), 4),
                        "coverage": round(float((p >= 0.5).mean()), 4)}
        curve[snr_db] = entry
        sa.METRICS["scenarios"][f"cafe/snr/{snr_db}dB"] = entry
    return curve


# ---------- 真实雨声场景 (test_rain_noise.py) ----------
RAIN_NOISE_FIXTURE = BASE / "test_fixtures" / "rain_noise_16k.wav"
RAIN_SPEECH_FIXTURE = BASE / "test_fixtures" / "rain_speech_16k.wav"
# 旁白活动窗 (素材内相对时间, 源视频 29.9-59.4s, 文件起点 25s)
RAIN_ACT0, RAIN_ACT1 = 4.9, 34.4


@pytest.fixture(scope="session")
def rain_noise_stats(runners):
    """裸雨声+采菌脆响: 均值 + 四件套误触发段数"""
    from vad_decision import SpeechSegmenter
    audio, sr = sf.read(RAIN_NOISE_FIXTURE, dtype="float32")
    assert sr == SAMPLE_RATE
    stats = {}
    for k, r in runners.items():
        p = sa.stream_probs(r, audio)
        seg = SpeechSegmenter(); seg.feed(p)
        stats[k] = {"mean": round(float(p.mean()), 4),
                    "triggers": len(seg.final_segments())}
    sa.METRICS["scenarios"]["rain/noise"] = stats
    return stats


@pytest.fixture(scope="session")
def rain_speech_stats(runners):
    """雨中轻声旁白: 均值/段数/活动窗覆盖率/交叠比 (uint8 假阴性 vs int8 检出)"""
    from vad_decision import SpeechSegmenter
    audio, sr = sf.read(RAIN_SPEECH_FIXTURE, dtype="float32")
    assert sr == SAMPLE_RATE
    stats = {}
    for k, r in runners.items():
        p = sa.stream_probs(r, audio)
        seg = SpeechSegmenter(); seg.feed(p)
        segs = seg.final_segments()
        t = np.arange(len(p)) * WINDOW / SAMPLE_RATE + WINDOW / SAMPLE_RATE / 2
        act = (t >= RAIN_ACT0) & (t < RAIN_ACT1)
        overlap = sum(max(0.0, min(e, RAIN_ACT1) - max(s0, RAIN_ACT0)) for s0, e in segs)
        stats[k] = {"mean": round(float(p.mean()), 4),
                    "segments": len(segs),
                    "coverage": round(float((p[act] >= 0.5).mean()), 4),
                    "overlap": round(overlap / (RAIN_ACT1 - RAIN_ACT0), 4)}
    sa.METRICS["scenarios"]["rain/speech"] = stats
    return stats


def pytest_sessionfinish(session, exitstatus):
    """把场景指标落盘为 CI 工件 (失败不影响测试结论; 为将来固件 HIL 对比预埋)"""
    if not sa.METRICS["scenarios"]:
        return
    import json
    out = BASE / "scenario_metrics.json"
    try:
        out.write_text(json.dumps(sa.METRICS, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n场景指标已写入: {out}")
    except OSError as e:  # 只读工作区等情况
        print(f"\n(场景指标写盘失败, 忽略: {e})")


@pytest.fixture(scope="session")
def audio():
    a, sr = sf.read(FIXTURE, dtype="float32")
    return a, sr


@pytest.fixture(scope="session")
def baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def probs(audio):
    a, _ = audio
    return infer_probs(a)


@pytest.fixture(scope="session")
def speech_mask(audio):
    """按窗中心时刻落在语音区 [1.5s, 总长-1.5s) 内取 True"""
    a, _ = audio
    n = len(range(0, len(a) - WINDOW, WINDOW))
    times = np.arange(n) * WINDOW / SAMPLE_RATE
    return (times + WINDOW / SAMPLE_RATE / 2 >= EDGE_SECONDS) & \
           (times + WINDOW / SAMPLE_RATE / 2 < len(a) / SAMPLE_RATE - EDGE_SECONDS)
