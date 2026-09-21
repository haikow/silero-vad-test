# -*- coding: utf-8 -*-
"""流式链路: 状态传递/确定性/HIL 比对逻辑 —— 移植阶段最容易踩的坑都在这"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_vad import WINDOW, VadRunner
from conftest import BASE, MODEL_PATHS, infer_probs

KEYS = ["uint8", "int8"]
RESET_DETECT_MIN_DIFF = 0.1   # 误重置状态时概率曲线应显著不同(实测差异远大于此值)


@pytest.mark.parametrize("key", KEYS)
def test_inference_determinism(key, audio):
    """同一输入两次完整流式推理结果应一致 (排除 ORT 随机性/线程不确定)"""
    a, _ = audio
    r1 = infer_probs(a)[key]
    r2 = infer_probs(a)[key]
    diff = np.abs(np.asarray(r1) - np.asarray(r2)).max()
    assert diff < 1e-9, f"{key}: 两次推理最大差异 {diff:.2e}, 推理不确定"


@pytest.mark.parametrize("key", KEYS)
def test_stream_state_is_carried(key, audio):
    """状态必须在窗间传递: 误把每窗状态清零会得到明显不同的概率曲线。

    这是嵌入式移植头号事故点(README 要点 2), 此用例专门守住它。
    """
    a, _ = audio
    normal = infer_probs(a)[key]
    r = VadRunner(MODEL_PATHS[key])
    reset_every = []
    for i in range(0, len(a) - WINDOW, WINDOW):
        r.reset()
        reset_every.append(r.process(a[i:i + WINDOW]))
    diff = np.abs(np.asarray(normal) - np.asarray(reset_every)).max()
    assert diff > RESET_DETECT_MIN_DIFF, \
        f"{key}: 每窗重置状态与正常流式最大差异仅 {diff:.4f}, " \
        "该用例失去检测力, 请复核素材里语音长度后调整阈值"


def test_hil_mock_passes_acceptance(audio):
    """HIL 比对链路自测: --mock 模式 (PC 模拟固件) 应满足固件验收标准"""
    result = subprocess.run(
        [sys.executable, str(BASE / "ci" / "firmware_test.py"), "--mock"],
        cwd=BASE, capture_output=True, text=True)
    out = BASE / "hil_result.json"
    try:
        assert result.returncode == 0, \
            f"firmware_test.py --mock 退出码 {result.returncode}:\n{result.stdout}\n{result.stderr}"
        rep = json.loads(out.read_text(encoding="utf-8"))
        assert rep["pass"] is True, f"HIL mock 报告未通过: {rep}"
    finally:
        out.unlink(missing_ok=True)
