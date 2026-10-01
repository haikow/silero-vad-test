# -*- coding: utf-8 -*-
"""
FireRedVAD (Engineering 版, MIT) 流式 runner —— 供 CI 场景对比

复刻 leospark/FireRedVAD-Engineering 的推理管线但去掉 torch 依赖
(原版 audio_feat.py 仅用 torch.from_numpy().float(), 对 float32 fbank 是恒等操作):
  int16 幅度波形 -> kaldi_native_fbank 80 维 (25ms 窗 / 10ms 移, snip_edges, dither=0)
  -> CMVN (kaldiio 读 ark, x-mean)*istd -> ONNX (输入 1 帧 + 8x cache_[0..7])
  -> sigmoid(logits) = 语音概率

帧网格: 10ms (与 Silero 32ms / WebRTC 30ms 不同, 场景评估按各自网格做, 见 ci/compare_vad.py)。
模型与 CMVN 在 models/firered/ (版权归 FireRedTeam, MIT), 来源:
https://github.com/leospark/FireRedVAD-Engineering
"""
from __future__ import annotations

import math
from pathlib import Path

import kaldi_native_fbank as knf
import kaldiio
import numpy as np
import onnxruntime as ort

BASE = Path(__file__).resolve().parent
ONNX_PATH = str(BASE / "models" / "firered" / "model_with_caches.onnx")
CMVN_PATH = str(BASE / "models" / "firered" / "cmvn.ark")

FRAME_LEN = 400   # 25ms @16k
FRAME_SHIFT = 160  # 10ms @16k
N_CACHE = 8
CACHE_SHAPE = (1, 128, 19)


def _load_cmvn(path: str):
    stats = kaldiio.load_mat(path)
    assert stats.shape[0] == 2
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    means, istds = [], []
    for d in range(dim):
        mean = stats[0, d] / count
        var = max(stats[1, d] / count - mean * mean, 1e-20)
        means.append(mean)
        istds.append(1.0 / math.sqrt(var))
    return np.asarray(means, np.float32), np.asarray(istds, np.float32)


class FireRedRunner:
    """流式: probs(audio) 逐 10ms 帧概率; reset() 清 8 个模型 cache"""

    def __init__(self, onnx_path: str = ONNX_PATH, cmvn_path: str = CMVN_PATH):
        self.means, self.istds = _load_cmvn(cmvn_path)
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        self._fbank_opts = opts
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.caches = [np.zeros(CACHE_SHAPE, np.float32) for _ in range(N_CACHE)]

    def _feat(self, frame_i16: np.ndarray) -> np.ndarray:
        # 与原版一致: 每帧独立建 OnlineFbank, 喂 int16 幅度 (tolist 语义 = 整数幅度)
        fbank = knf.OnlineFbank(self._fbank_opts)
        fbank.accept_waveform(16000, frame_i16.astype(np.float64).tolist())
        feat = np.vstack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        feat = (feat.astype(np.float32) - self.means) * self.istds
        return feat.reshape(1, 1, -1)

    def probs(self, audio: np.ndarray) -> np.ndarray:
        """float32 [-1,1] 波形 -> 逐 10ms 帧概率 (帧数 = (N-400)//160+1)

        每次调用视为一条新流: 先清 cache (与原版 process_audio 的 reset 语义一致)
        """
        self.reset()
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        n = (len(pcm) - FRAME_LEN) // FRAME_SHIFT + 1
        out = np.empty(n, np.float32)
        feed = {}
        for i in range(n):
            frame = pcm[i * FRAME_SHIFT:i * FRAME_SHIFT + FRAME_LEN]
            feed.clear()
            feed["input"] = self._feat(frame)
            for c in range(N_CACHE):
                feed[f"cache_{c}"] = self.caches[c]
            outputs = self.session.run(None, feed)
            self.caches = [outputs[c + 1] for c in range(N_CACHE)]
            out[i] = 1.0 / (1.0 + math.exp(-float(outputs[0][0, 0, 0])))
        return out
