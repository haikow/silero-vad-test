# -*- coding: utf-8 -*-
"""算法基准回归: 与 test_fixtures/baseline_probs.json 逐窗比对

判定阈值与 ci/run_tests.py 同源 (import 常量, 不另写一份)
"""
import numpy as np
import pytest

from ci.run_tests import MIN_CORR, TOLERANCE

KEYS = ["uint8", "int8"]


@pytest.mark.parametrize("key", KEYS)
def test_window_count_matches_baseline(key, probs, baseline):
    assert len(probs[key]) == len(baseline[key]), \
        f"{key}: 窗数 {len(baseline[key])} -> {len(probs[key])}, 素材或窗口配置变了"


@pytest.mark.parametrize("key", KEYS)
def test_max_deviation_within_tolerance(key, probs, baseline):
    p, b = np.asarray(probs[key]), np.asarray(baseline[key])
    dev = np.abs(p - b).max()
    n_over = int((np.abs(p - b) > TOLERANCE).sum())
    assert dev <= TOLERANCE, \
        f"{key}: 最大偏差 {dev:.4f} 超容差 {TOLERANCE} (超限窗 {n_over}/{len(p)}); " \
        "如属预期变更, ci/run_tests.py --update-baseline 后提交"


@pytest.mark.parametrize("key", KEYS)
def test_prob_curve_correlation(key, probs, baseline):
    p, b = np.asarray(probs[key]), np.asarray(baseline[key])
    corr = np.corrcoef(p, b)[0, 1]
    assert corr >= MIN_CORR, f"{key}: 相关系数 {corr:.5f} < {MIN_CORR}, 曲线形状变了"
