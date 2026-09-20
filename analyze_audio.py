# -*- coding: utf-8 -*-
"""
用 Silero VAD 分析任意音频文件的人声分布
用法: python analyze_audio.py <音频路径>
- 任意格式 (m4a/mp3/ogg/wav...), 内部用 ffmpeg 转 16k mono wav
- 依次跑 uint8/v5 与 int8/v4 两个量化模型, 输出概率时间线与语音区间
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

from test_vad import (MODELS, SAMPLE_RATE, THRESHOLD, WINDOW,
                      VadRunner, segments_from_probs)

STEP_SEC = 0.5  # 时间线每格时长


def load_audio(path: str) -> np.ndarray:
    """任意音频 -> float32 mono 16kHz; wav 直读, mp3 用 miniaudio, 其余走 ffmpeg"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return audio if sr == SAMPLE_RATE else _resample(audio, sr)
    if ext == ".mp3":
        import miniaudio
        dec = miniaudio.decode_file(path, nchannels=1, sample_rate=SAMPLE_RATE,
                                    output_format=miniaudio.SampleFormat.FLOAT32)
        return np.array(dec.samples, dtype=np.float32)
    ffmpeg = _find_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = os.path.join(td, "tmp_16k.wav")
        cmd = [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-f", "wav", tmp_wav]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg 转码失败:\n{r.stderr[-1500:]}")
        audio, _ = sf.read(tmp_wav, dtype="float32")
        return audio


def _find_ffmpeg() -> str:
    from shutil import which
    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise SystemExit("需要 ffmpeg: pip install imageio-ffmpeg 或安装系统 ffmpeg")


def _resample(x, sr_from):
    n_out = int(len(x) * SAMPLE_RATE / sr_from)
    return np.interp(np.linspace(0, len(x) - 1, n_out),
                     np.arange(len(x)), x).astype(np.float32)


def analyze(path: str):
    audio = load_audio(path)
    dur = len(audio) / SAMPLE_RATE
    peak = np.abs(audio).max()
    print(f"文件: {path}")
    print(f"时长: {dur:.2f}s, 采样率: {SAMPLE_RATE} Hz, 峰值幅度: {peak:.3f}")
    if peak < 0.005:
        print("警告: 音频几乎静音, 请检查文件是否下载完整")

    for name, model_path in MODELS.items():
        runner = VadRunner(model_path)
        ver = "v5" if runner.is_v5 else "v4"
        probs, times, lat = [], [], []
        for i in range(0, len(audio) - WINDOW, WINDOW):
            t0 = time.perf_counter()
            probs.append(runner.process(audio[i:i + WINDOW]))
            lat.append((time.perf_counter() - t0) * 1000)
            times.append(i / SAMPLE_RATE)
        probs, times = np.array(probs), np.array(times)

        speech_ratio = (probs >= THRESHOLD).mean()
        segs = segments_from_probs(probs, times, THRESHOLD)
        print(f"\n===== {name} [{ver}] =====")
        print(f"语音占比 (阈值{THRESHOLD}): {speech_ratio*100:.1f}%  "
              f"平均概率: {probs.mean():.3f}  每窗耗时: {np.mean(lat[5:]):.3f} ms")
        segs_str = ", ".join(f"[{a:.2f},{b:.2f}]" for a, b in segs[:30])
        print(f"语音区间: {segs_str}{' ...' if len(segs) > 30 else ''}")

        print("概率时间线 (每格 0.5s):")
        step = int(STEP_SEC * SAMPLE_RATE / WINDOW)
        for j in range(0, len(probs), step):
            p = probs[j:j + step].mean()
            bar = "#" * int(p * 40)
            print(f"  {times[j]:6.2f}s {p:.3f} |{bar:<40}|")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法: python analyze_audio.py <音频文件>")
    analyze(sys.argv[1])
