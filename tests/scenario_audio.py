# -*- coding: utf-8 -*-
"""场景素材合成 + 指标登记: 全部确定性(固定种子), 零新增音频文件

素材来源: 从已提交的 test_fixtures/smoke_test.wav 抽出语音段(峰值归一),
按场景缩放电平/叠加种子化噪声。METRICS 收集各场景指标, 由 conftest 落盘
scenario_metrics.json (CI 工件) —— 将来固件 HIL 跑同一组场景, 比对该表。
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TESTS_DIR = Path(__file__).resolve().parent
BASE = TESTS_DIR.parent
sys.path.insert(0, str(BASE))

from test_vad import SAMPLE_RATE as SR, WINDOW  # noqa: E402

METRICS: dict = {"scenarios": {}}   # conftest 在 session 结束时落盘


def speech_from_fixture(peak: float = 1.0) -> np.ndarray:
    """从 smoke_test.wav 抽语音段 [1.5s, 总长-1.5s), 峰值归一后缩放到 peak"""
    audio, _ = sf.read(BASE / "test_fixtures" / "smoke_test.wav", dtype="float32")
    s0, s1 = int(1.5 * SR), len(audio) - int(1.5 * SR)
    speech = audio[s0:s1]
    return (speech / (np.abs(speech).max() + 1e-9) * peak).astype(np.float32)


def padded(speech: np.ndarray, pad_s: float = 0.75) -> np.ndarray:
    pad = np.zeros(int(pad_s * SR), np.float32)
    return np.concatenate([pad, speech, pad])


def _unit_noise(kind: str, n: int, seed: int) -> np.ndarray:
    """单位标准差的有色噪声 (去均值)"""
    rng = np.random.default_rng(seed)
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":      # FFT 1/sqrt(f) 着色, 通风/道路噪声近似
        f = np.fft.rfft(rng.standard_normal(n))
        f[1:] /= np.sqrt(np.arange(1, len(f)))
        x = np.fft.irfft(f, n)
    elif kind == "brown":     # 布朗噪声(累积), 低频隆隆
        x = np.cumsum(rng.standard_normal(n))
        x -= x.mean()
    else:
        raise ValueError(kind)
    return (x / (x.std() + 1e-12)).astype(np.float32)


def noise_clip(kind: str, seconds: float, rms: float, seed: int = 7) -> np.ndarray:
    return (_unit_noise(kind, int(seconds * SR), seed) * rms).astype(np.float32)


def snr_mix(speech_peak: float, snr_db: float, seed: int = 11) -> np.ndarray:
    """语音(峰值 speech_peak) + 白噪声, 按语音段 RMS 定标到指定 SNR"""
    speech = speech_from_fixture(speech_peak)
    noise_std = float(speech.std()) / 10 ** (snr_db / 20)
    noise = (np.random.default_rng(seed).standard_normal(len(speech)) * noise_std)
    return padded((speech + noise).astype(np.float32))


def stream_probs(runner, x: np.ndarray) -> np.ndarray:
    """整段流式推理 (先 reset, 状态窗间传递)"""
    runner.reset()
    return np.array([runner.process(x[i:i + WINDOW]) for i in range(0, len(x) - WINDOW, WINDOW)])


def region_mask(n_windows: int, t0: float, t1: float) -> np.ndarray:
    """窗中心时刻落在 [t0, t1) 的掩码 (t 为流内相对时间)"""
    t = np.arange(n_windows) * WINDOW / SR + WINDOW / SR / 2
    return (t >= t0) & (t < t1)
