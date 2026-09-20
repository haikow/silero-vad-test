# -*- coding: utf-8 -*-
"""
Silero VAD 量化模型对比测试（uint8 vs int8）
- 自动适配 v4 (input/h/c, 无 sr) 和 v5 (input/state/sr) 两种 ONNX 接口
- 测试音频: 白噪声(1.5s) + 真人语音(TTS) + 白噪声(1.5s), 16kHz
- 输出: 噪声段/语音段概率、语音起止区间、每窗耗时、实时率(RTF)
"""
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import onnxruntime as ort

BASE = Path(__file__).resolve().parent
MODELS = {
    "uint8 (onnx-community, 639KB)": str(BASE / "models" / "silero_vad_uint8.onnx"),
    "int8  (sherpa-onnx, 208KB)":    str(BASE / "models" / "silero_vad_int8.onnx"),
}

SAMPLE_RATE = 16000
WINDOW = 512  # 32ms @ 16kHz
THRESHOLD = 0.5

RAW_WAV = str(BASE / "audio" / "speech_raw.wav")
MIX_WAV = str(BASE / "audio" / "test_mixed_16k.wav")


def resample_linear(x, sr_from, sr_to):
    """简单线性插值重采样，测试用途足够"""
    if sr_from == sr_to:
        return x
    n_out = int(len(x) * sr_to / sr_from)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x)


def prepare_audio():
    """读 TTS 语音 -> 重采样 16k -> 前后拼白噪声 -> 存盘"""
    audio, sr = sf.read(RAW_WAV, dtype="float32", always_2d=True)
    speech = audio[:, 0] if audio.shape[1] > 1 else audio.mean(axis=1)
    speech = speech / (np.abs(speech).max() + 1e-9) * 0.8
    speech = resample_linear(speech, sr, SAMPLE_RATE).astype(np.float32)

    rng = np.random.default_rng(42)
    noise_pre = (rng.standard_normal(int(1.5 * SAMPLE_RATE)) * 0.05).astype(np.float32)
    noise_post = (rng.standard_normal(int(1.5 * SAMPLE_RATE)) * 0.05).astype(np.float32)
    mixed = np.concatenate([noise_pre, speech, noise_post])
    sf.write(MIX_WAV, mixed, SAMPLE_RATE)
    speech_start = len(noise_pre)
    speech_end = len(noise_pre) + len(speech)
    print(f"测试音频: 总长 {len(mixed)/SAMPLE_RATE:.2f}s, "
          f"语音区间 [{speech_start/SAMPLE_RATE:.2f}s, {speech_end/SAMPLE_RATE:.2f}s]")
    return mixed, speech_start, speech_end


class VadRunner:
    """按 ONNX 输入签名自动适配 v4/v5 接口"""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        inputs = {i.name: i.shape for i in self.session.get_inputs()}
        self.is_v5 = "state" in inputs and "sr" in inputs
        # 主输入名: v5 一般叫 input, sherpa v4 叫 x —— 取不属于状态量的那个
        state_names = {"state", "sr", "h", "c"}
        self.main_input = next(n for n in inputs if n not in state_names)
        if self.is_v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
        else:  # v4: h / c
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def reset(self):
        if self.is_v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def process(self, chunk: np.ndarray) -> float:
        x = chunk.reshape(1, -1).astype(np.float32)
        if self.is_v5:
            out, self.state = self.session.run(
                None, {self.main_input: x, "state": self.state, "sr": self.sr})
        else:
            out, self.h, self.c = self.session.run(
                None, {self.main_input: x, "h": self.h, "c": self.c})
        return float(out[0][0])


def segments_from_probs(probs, times, threshold):
    """由逐窗概率提取语音区间"""
    segs, start = [], None
    for p, t in zip(probs, times):
        if p >= threshold and start is None:
            start = t
        elif p < threshold and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, times[-1] + WINDOW / SAMPLE_RATE))
    return segs


def run_model(name, path, audio, speech_start, speech_end):
    print("\n" + "=" * 72)
    print(f"模型: {name}")
    print(f"文件: {path}")
    runner = VadRunner(path)
    ver = "v5 (input/state/sr)" if runner.is_v5 else "v4 (input/h/c)"
    print(f"接口: {ver}")
    print(f"输入: {[(i.name, i.shape) for i in runner.session.get_inputs()]}")
    print(f"输出: {[(o.name, o.shape) for o in runner.session.get_outputs()]}")

    probs, times, lat_ms = [], [], []
    for i in range(0, len(audio) - WINDOW, WINDOW):
        chunk = audio[i:i + WINDOW]
        t0 = time.perf_counter()
        p = runner.process(chunk)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        probs.append(p)
        times.append(i / SAMPLE_RATE)

    probs = np.array(probs)
    times = np.array(times)
    half_win_s = WINDOW / SAMPLE_RATE / 2
    in_speech = (times + half_win_s >= speech_start / SAMPLE_RATE) & \
                (times + half_win_s < speech_end / SAMPLE_RATE)
    noise_p, speech_p = probs[~in_speech], probs[in_speech]
    lat = np.array(lat_ms)[5:]  # 丢前几窗(含首次推理的初始化开销)

    print(f"\n噪声段  平均概率: {noise_p.mean():.4f}  最大: {noise_p.max():.4f}")
    print(f"语音段  平均概率: {speech_p.mean():.4f}  最小: {speech_p.min():.4f}")
    print(f"区分度 (语音均值-噪声均值): {speech_p.mean() - noise_p.mean():.4f}")
    print(f"每窗耗时: 平均 {lat.mean():.3f} ms / 最大 {lat.max():.3f} ms"
          f"  (窗口长度 {WINDOW/SAMPLE_RATE*1000:.0f} ms)")
    print(f"实时率 RTF: {lat.mean() / (WINDOW/SAMPLE_RATE*1000):.5f}"
          f"  (越小于 1 越好, 嵌入式参考)")

    segs = segments_from_probs(probs, times, THRESHOLD)
    segs_str = ", ".join(f"[{a:.2f}s, {b:.2f}s]" for a, b in segs) or "无"
    print(f"检测到的语音区间 (阈值{THRESHOLD}): {segs_str}")
    print(f"真实语音区间: [{speech_start/SAMPLE_RATE:.2f}s, "
          f"{speech_end/SAMPLE_RATE:.2f}s]")

    print("\n概率时间线 (每格 0.25s, # 越多概率越高):")
    step = int(0.25 * SAMPLE_RATE / WINDOW)
    for j in range(0, len(probs), step):
        seg_p = probs[j:j + step].mean()
        bar = "#" * int(seg_p * 40)
        t = times[j]
        tag = "语音" if speech_start / SAMPLE_RATE <= t < speech_end / SAMPLE_RATE else "噪声"
        print(f"  {t:5.2f}s [{tag}] {seg_p:.3f} |{bar:<40}|")
    return probs


def main():
    audio, s0, s1 = prepare_audio()
    results = {}
    for name, path in MODELS.items():
        results[name] = run_model(name, path, audio, s0, s1)

    print("\n" + "=" * 72)
    print("结论: 两个模型概率曲线一致性 (相关系数):")
    keys = list(results)
    a, b = results[keys[0]], results[keys[1]]
    n = min(len(a), len(b))
    print(f"  {np.corrcoef(a[:n], b[:n])[0,1]:.4f}")


if __name__ == "__main__":
    main()
