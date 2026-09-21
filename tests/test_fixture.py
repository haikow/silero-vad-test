# -*- coding: utf-8 -*-
"""素材完整性: 模型/音频/基准文件齐全且互相一致 (失败时给出人话而不是报错栈)"""
from pathlib import Path

from test_vad import SAMPLE_RATE, WINDOW
from conftest import FIXTURE, MODEL_PATHS


def test_models_present():
    for key, path in MODEL_PATHS.items():
        assert Path(path).exists(), f"模型缺失: {key} -> {path}, 先跑 download_models.py"


def test_fixture_exists_and_sample_rate(audio):
    assert FIXTURE.exists(), f"素材缺失: {FIXTURE}"
    _, sr = audio
    assert sr == SAMPLE_RATE, f"素材采样率 {sr} != {SAMPLE_RATE}"


def test_baseline_window_count_consistent(audio, baseline):
    """素材窗口数与基准逐窗数组长度必须一致, 否则回归比对无意义"""
    a, _ = audio
    n_windows = len(range(0, len(a) - WINDOW, WINDOW))
    assert baseline["window"] == WINDOW and baseline["sr"] == SAMPLE_RATE, \
        "基准的窗口/采样率配置与 test_vad 不一致"
    for key in ("uint8", "int8"):
        assert len(baseline[key]) == n_windows, \
            f"{key}: 基准窗数 {len(baseline[key])} != 素材窗数 {n_windows}, " \
            "素材变了? 用 ci/run_tests.py --update-baseline 重新生成"
