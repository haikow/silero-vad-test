# -*- coding: utf-8 -*-
"""
工程级 VAD 判定四件套 —— Python 参考实现

README 规格: 起始阈值高 (0.6) + 结束阈值低 (0.35, 迟滞) + 最短语音时长 (0.5s)
+ 最短静音时长 (0.25s)。实测 B 站咖啡馆嘈杂素材: 裸阈值 25.5% 误触发 -> 四件套后 13%。

本模块是规格本体: 语义由 tests/test_decision_logic.py 的单测逐条锁定,
将来固件 C 移植按"同输入同输出"对照实现。
"""
from __future__ import annotations

WINDOW_S = 512 / 16000  # 32ms 一窗 (与 test_vad.WINDOW/SAMPLE_RATE 一致)


class SpeechSegmenter:
    """逐窗概率 -> 语音段。

    - 迟滞: p >= start 进入语音; 进入后 p >= end 即算语音持续(含 0.35~0.6 迟滞带),
      连续低于 end 达 min_silence 才退出
    - 最短语音: 候选段语音累计不足 min_speech 则丢弃(防毛刺误触发)
    - 段结束时刻回退到静音起点(不含静音尾)
    """

    def __init__(self, start: float = 0.6, end: float = 0.35,
                 min_speech: float = 0.5, min_silence: float = 0.25,
                 window_s: float = WINDOW_S):
        assert 0 < end <= start < 1, "要求 end <= start (迟滞)"
        self.start, self.end = start, end
        self.min_speech, self.min_silence = min_speech, min_silence
        self.w = window_s

        self._idx = 0            # 已处理窗数
        self._in_speech = False  # 迟滞后的实时门控状态
        self._seg_start = 0.0    # 候选段起点 (秒)
        self._speech_n = 0       # 候选段内语音窗数
        self._silence_n = 0      # 连续静音窗数
        self.segments: list[tuple[float, float]] = []  # 已判定成立的段

    @property
    def live(self) -> bool:
        """当前是否处于语音(实时门控用, 含未满 min_speech 的候选)"""
        return self._in_speech

    def process(self, p: float) -> bool:
        """送入一窗概率, 返回该窗的实时门控状态"""
        w = self.w
        if not self._in_speech:
            if p >= self.start:
                self._in_speech = True
                self._seg_start = self._idx * w
                self._speech_n, self._silence_n = 1, 0
        else:
            if p >= self.end:          # 迟滞带内都算语音持续
                self._speech_n += 1
                self._silence_n = 0
            else:
                self._silence_n += 1
                if self._silence_n * w >= self.min_silence:
                    # 静音从第 (idx - silence_n + 1) 窗开始, 段尾回退到静音起点
                    self._close((self._idx - self._silence_n + 1) * w)
        self._idx += 1
        return self._in_speech

    def feed(self, probs) -> list[bool]:
        return [self.process(p) for p in probs]

    def _close(self, end_s: float):
        if self._speech_n * self.w >= self.min_speech:
            self.segments.append((self._seg_start, end_s))
        self._in_speech = False
        self._speech_n = self._silence_n = 0

    def flush(self):
        """流结束: 未闭合的候选段按当前时刻收尾判定(不补静音尾)"""
        if self._in_speech:
            self._close(self._idx * self.w)

    def final_segments(self) -> list[tuple[float, float]]:
        """返回全部成立的段(先 flush)。噪声流上的段数即误触发次数"""
        self.flush()
        return list(self.segments)
