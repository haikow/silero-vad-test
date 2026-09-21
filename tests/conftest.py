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
