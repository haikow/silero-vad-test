# -*- coding: utf-8 -*-
"""原厂 WebRTC VAD (同源开源实现) vs Silero 量化模型 —— 同一组场景对比

背景: 原厂 VAD 是闭源 libwq_sw_vad.a (Xtensa HiFi5 静态库, Mac 上不可链接);
算法本体是开源 WebRTC VAD (BSD)。本脚本用 webrtcvad(官方 C 源码的 Python 绑定)
作为原厂代表, 在与 CI 场景套件完全相同的素材上跑 4 档激进度 (0 最宽松, 3 最保守)。
Silero 侧数值直接读 scenario_metrics.json (与 run 页摘要同源)。
真原厂数字: 拿到 libwq_sw_vad.a 后用 xt-run 指令集仿真器跑 (见 README)。

用法: python ci/compare_webrtc.py           # 需要 pip install webrtcvad 'setuptools<81'
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

from scenario_audio import padded, speech_from_fixture  # noqa: E402
from vad_decision import SpeechSegmenter                # noqa: E402

FRAME = 480          # 30ms @16k (WebRTC 支持的帧长)
FRAME_S = FRAME / 16000
AGGRESSIVENESS = (0, 1, 2, 3)


def decisions(audio: np.ndarray, aggr: int) -> np.ndarray:
    """WebRTC VAD 逐 30ms 帧判定 (每条流新建实例, 干净状态)"""
    vad = webrtcvad.Vad(aggr)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    n = len(audio) // FRAME
    return np.array([vad.is_speech(pcm[i * 2 * FRAME:(i + 1) * 2 * FRAME], 16000)
                     for i in range(n)], dtype=np.int8)


def rate_in(dec: np.ndarray, t0: float, t1: float) -> float:
    c = (np.arange(len(dec)) * FRAME + FRAME / 2) / 16000
    return float(dec[(c >= t0) & (c < t1)].mean())


def fourpiece_triggers(dec: np.ndarray) -> int:
    s = SpeechSegmenter(window_s=FRAME_S)
    s.feed(dec.astype(float))
    return len(s.final_segments())


def load(name):
    a, sr = sf.read(BASE / "test_fixtures" / name, dtype="float32")
    assert sr == 16000
    return a


def scenarios():
    """与 tests/ 场景套件同口径的素材与评估区, 返回 [(名称, 音频, 区间或 None, 指标型)]"""
    items = []
    smoke = load("smoke_test.wav")
    items.append(("smoke/语音区检出", smoke, (1.5, len(smoke) / 16000 - 1.5), "rate"))
    items.append(("smoke/噪声区误报", np.concatenate([smoke[:int(1.5 * 16000)],
                                                       smoke[-int(1.5 * 16000):]]), None, "trig"))
    for peak in (0.05, 0.1, 0.3, 0.8):
        x = padded(speech_from_fixture(peak))
        items.append((f"电平{peak}/语音区检出", x, (0.75, len(x) / 16000 - 0.75), "rate"))
    from scenario_audio import snr_mix
    for db in (20, 10, 5, 0):
        x = snr_mix(speech_peak=0.3, snr_db=db)
        items.append((f"白噪SNR{db}dB/检出", x, (0.75, len(x) / 16000 - 0.75), "rate"))
    cafe = load("cafe_noise_16k.wav")
    items.append(("咖啡馆/裸噪声误触发", cafe, None, "trig"))
    speech = speech_from_fixture(0.3)
    for db in (10, 5, 0):
        noise = cafe[:len(speech)] * (speech.std() / (cafe.std() + 1e-9) / 10 ** (db / 20))
        x = padded(speech + noise.astype(np.float32))
        items.append((f"咖啡馆SNR{db}dB/检出", x, (0.75, len(x) / 16000 - 0.75), "rate"))
    rain = load("rain_noise_16k.wav")
    items.append(("雨声/裸噪声误触发", rain, None, "trig"))
    rs = load("rain_speech_16k.wav")
    items.append(("雨中旁白/检出", rs, (4.9, 34.4), "rate"))
    return items


def main():
    ours = json.loads((BASE / "scenario_metrics.json").read_text(encoding="utf-8"))["scenarios"]

    print("=" * 88)
    print("原厂 WebRTC VAD (开源同源, 激进度 0=宽松 ~ 3=保守) vs Silero (CI 实测)")
    print("=" * 88)
    hdr = f"{'场景':<22}" + "".join(f"{'WRTC-'+str(a):>9}" for a in AGGRESSIVENESS) + \
          f"{'int8':>9}{'uint8':>9}"
    print(hdr)
    print("-" * 88)

    for name, audio, region, kind in scenarios():
        cells = []
        for a in AGGRESSIVENESS:
            dec = decisions(audio, a)
            if kind == "trig":
                # 帧级误报率(主) + 四件套段数(括号): 连续误开时段数会误导性偏低
                cells.append(f"{dec.mean():>4.0%}/{fourpiece_triggers(dec)}段")
            else:
                cells.append(f"{rate_in(dec, *region):>8.0%} ")
        # Silero 侧对照值 (从 CI 指标表取)
        sil = ("-", "-")
        if name.startswith("smoke/语音"):
            sil = (f"{ours['snr/20dB']['int8']['coverage']:.0%}", "")
        if "白噪SNR" in name:
            db = name.split("SNR")[1].split("dB")[0]
            if f"snr/{db}dB" in ours:
                sil = (f"{ours[f'snr/{db}dB']['int8']['coverage']:.0%}",
                       f"{ours[f'snr/{db}dB']['uint8']['coverage']:.0%}")
        if "咖啡馆SNR" in name:
            db = name.split("SNR")[1].split("dB")[0]
            sil = (f"{ours[f'cafe/snr/{db}dB']['int8']['coverage']:.0%}",
                   f"{ours[f'cafe/snr/{db}dB']['uint8']['coverage']:.0%}")
        if name.startswith("咖啡馆/裸"):
            sil = (f"{ours['cafe/raw']['int8']['triggers']}段",
                   f"{ours['cafe/raw']['uint8']['triggers']}段")
        if name.startswith("雨中旁白"):
            sil = (f"{ours['rain/speech']['int8']['coverage']:.0%}",
                   f"{ours['rain/speech']['uint8']['coverage']:.0%}")
        if name.startswith("雨声/裸"):
            sil = (f"{ours['rain/noise']['int8']['triggers']}段", "0段")
        if "电平" in name:
            # 电平矩阵的覆盖率不在指标表里(表存的是均值), 现场推理补算
            peak = float(name.split("电平")[1].split("/")[0])
            from scenario_audio import stream_probs
            from test_vad import WINDOW, VadRunner
            x = padded(speech_from_fixture(peak))
            cov = {}
            for k in ("int8", "uint8"):
                pr = stream_probs(VadRunner(str(BASE / "models" / f"silero_vad_{k}.onnx")), x)
                t = np.arange(len(pr)) * WINDOW / 16000 + WINDOW / 32000
                m = (t >= 0.75) & (t < len(x) / 16000 - 0.75)
                cov[k] = f"{(pr[m] >= 0.5).mean():.0%}"
            sil = (cov["int8"], cov["uint8"])
        print(f"{name:<22}" + "".join(cells) + f"{sil[0]:>9}{sil[1]:>9}")
    print("\n注: '段' = 四件套判定误触发段数(素材时长见 tests/); 其余为区域内判定覆盖率")


if __name__ == "__main__":
    main()
