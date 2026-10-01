# -*- coding: utf-8 -*-
"""VAD 多引擎对比: 原厂 WebRTC (同源开源) / FireRedVAD / Silero 量化 —— 同一组场景

引擎与帧网格(场景评估按各自网格, 判定逻辑统一为四件套 0.6/0.35/0.5s/0.25s):
  WebRTC VAD  激进度 0~3, 30ms 帧, 0/1 判定 (原厂 libwq_sw_vad.a 同算法)
  FireRedVAD   Engineering 版(MIT), 10ms 帧, sigmoid 概率 (vad_firered.py, 无 torch)
  Silero       int8/uint8 量化模型 (数值 = 先跑用例套件生成的 scenario_metrics.json)

用法: python ci/compare_vad.py
依赖: webrtcvad, kaldiio, kaldi-native-fbank (均在 requirements.txt)
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tests"))

from scenario_audio import padded, snr_mix, speech_from_fixture  # noqa: E402
from vad_decision import SpeechSegmenter                          # noqa: E402

AGGRESSIVENESS = (0, 1, 2, 3)


# ---------------- 引擎适配层 ----------------
# 统一返回 (概率数组, 帧移秒, 帧长秒, 覆盖率阈值, 四件套 start, 四件套 end)
# FireRed 概率软校准(噪声~0.50-0.51 / 语音~0.67-0.73, 永不过 0.75): 其默认 0.5 阈值
# 下连纯雨声(0.506)都过线近乎全开, 故按其输出分布取 0.6 分界 + 0.6/0.55 迟滞(实测验证)


def webrtc_engine(aggr: int):
    import webrtcvad
    frame, shift = 480, 480  # 30ms 帧, 无重叠

    def run(audio):
        vad = webrtcvad.Vad(aggr)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        n = len(audio) // frame
        dec = np.array([vad.is_speech(pcm[i * 2 * frame:(i + 1) * 2 * frame], 16000)
                        for i in range(n)], dtype=np.float32)
        return dec, shift / 16000, frame / 16000, 0.5, 0.6, 0.35
    return run


def firered_engine():
    from vad_firered import FRAME_LEN, FRAME_SHIFT, FireRedRunner
    runner = FireRedRunner()

    def run(audio):
        p = runner.probs(audio)
        return p, FRAME_SHIFT / 16000, FRAME_LEN / 16000, 0.6, 0.6, 0.55
    return run


ENGINES = [(f"WRTC-{a}", webrtc_engine(a)) for a in AGGRESSIVENESS] + [("FireRed", firered_engine())]


# ---------------- 统一评估: 覆盖率 / 四件套误触发段 ----------------

def rate_in(probs, shift_s, frame_len_s, thr, t0, t1):
    c = (np.arange(len(probs)) * shift_s + frame_len_s / 2)
    m = (c >= t0) & (c < t1)
    return float((probs[m] >= thr).mean())


def fourpiece_triggers(probs, shift_s, start, end):
    s = SpeechSegmenter(start=start, end=end, window_s=shift_s)
    s.feed(probs)
    return len(s.final_segments())


def load(name):
    a, sr = sf.read(BASE / "test_fixtures" / name, dtype="float32")
    assert sr == 16000
    return a


def scenarios():
    items = []
    smoke = load("smoke_test.wav")
    items.append(("smoke/语音区检出", smoke, (1.5, len(smoke) / 16000 - 1.5), "rate"))
    items.append(("smoke/噪声区误报", np.concatenate([smoke[:int(1.5 * 16000)],
                                                       smoke[-int(1.5 * 16000):]]), None, "trig"))
    for peak in (0.05, 0.1, 0.3, 0.8):
        x = padded(speech_from_fixture(peak))
        items.append((f"电平{peak}/语音区检出", x, (0.75, len(x) / 16000 - 0.75), "rate"))
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
    for name in ("libri_spk_a_60s.wav", "libri_spk_b_60s.wav"):
        a = load(name)
        items.append((f"Libri/{name[10]}检出", a, (0.5, len(a) / 16000 - 0.5), "rate"))
    for name in ("reverb_small_room.wav", "reverb_large_room.wav"):
        a = load(name)
        items.append((f"混响{name[7:12]}/检出", a, (0.5, len(a) / 16000 - 0.5), "rate"))
    return items


def silero_cells(name, ours):
    """Silero 侧对照值: 优先读指标表; 电平行现场推理补覆盖率"""
    if "白噪SNR" in name:
        db = name.split("SNR")[1].split("dB")[0]
        if f"snr/{db}dB" in ours:
            return (f"{ours[f'snr/{db}dB']['int8']['coverage']:.0%}",
                    f"{ours[f'snr/{db}dB']['uint8']['coverage']:.0%}")
    if "咖啡馆SNR" in name:
        db = name.split("SNR")[1].split("dB")[0]
        return (f"{ours[f'cafe/snr/{db}dB']['int8']['coverage']:.0%}",
                f"{ours[f'cafe/snr/{db}dB']['uint8']['coverage']:.0%}")
    if name.startswith("咖啡馆/裸"):
        return (f"{ours['cafe/raw']['int8']['triggers']}段",
                f"{ours['cafe/raw']['uint8']['triggers']}段")
    if name.startswith("雨中旁白"):
        return (f"{ours['rain/speech']['int8']['coverage']:.0%}",
                f"{ours['rain/speech']['uint8']['coverage']:.0%}")
    if name.startswith("雨声/裸"):
        return (f"{ours['rain/noise']['int8']['triggers']}段", "0段")
    if name.startswith("Libri/"):
        key = f"ds/libri_spk_{name[6]}_60s.wav"
        if key in ours:
            return (f"{ours[key]['silero-int8']:.0%}", f"{ours[key]['silero-uint8']:.0%}")
    if name.startswith("混响"):
        room = name[2:7]  # small / large
        key = f"ds/reverb_{room}_room.wav"
        if key in ours:
            return (f"{ours[key]['silero-int8']:.0%}", f"{ours[key]['silero-uint8']:.0%}")
    if name.startswith("smoke/语音"):
        return (f"{ours['snr/20dB']['int8']['coverage']:.0%}", "")
    if "电平" in name:
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
        return cov["int8"], cov["uint8"]
    return "-", "-"


def main():
    ours = json.loads((BASE / "scenario_metrics.json").read_text(encoding="utf-8"))["scenarios"]

    print("=" * 100)
    print("VAD 多引擎对比: 原厂 WebRTC(同源开源, 激进度 0=宽松~3=保守) / FireRedVAD / Silero 量化")
    print("=" * 100)
    print(f"{'场景':<20}" + "".join(f"{n:>9}" for n, _ in ENGINES) + f"{'int8':>9}{'uint8':>9}")
    print("-" * 100)

    for name, audio, region, kind in scenarios():
        cells = []
        for _, engine in ENGINES:
            probs, shift_s, frame_len_s, thr, start, end = engine(audio)
            if kind == "trig":
                cells.append(f"{(probs >= thr).mean():>4.0%}/"
                             f"{fourpiece_triggers(probs, shift_s, start, end)}段")
            else:
                cells.append(f"{rate_in(probs, shift_s, frame_len_s, thr, *region):>8.0%} ")
        sil = silero_cells(name, ours)
        print(f"{name:<20}" + "".join(cells) + f"{sil[0]:>9}{sil[1]:>9}")
    print("\n注: 裸噪声行 = 帧级误报率/四件套段数(连续误开时段数偏低, 看帧级率); 其余为区域内覆盖率")
    print("阈值: WebRTC=自身判定; FireRed=0.6(其概率软校准: 噪声~0.50/语音~0.73, 默认 0.5 下噪声全过线); Silero=0.5")

    # ---- MUSAN 真实素材区块 (fixture 由 tests/test_musan.py 的扫描标定选出) ----
    if all(k in ours for k in ("musan/noise", "musan/music", "musan/speech")):
        order = [n for n, _ in ENGINES] + ["silero-int8", "silero-uint8"]
        print("\n" + "=" * 100)
        print("MUSAN 真实素材 (ci/musan_sweep.py 全库扫描 106/843 选点;"
              " 分布: WebRTC 89~95% 文件误触发 / FireRed 26% / int8 21% / uint8 14%)")
        print("=" * 100)
        for label, title in (("musan/noise", "噪声 6 件 误报均值/总段数:"),
                             ("musan/music", "音乐 2 件 误报均值/总段数:")):
            d = ours[label]
            cells = "  ".join(f"{n}={d[n]['fp_mean']:.0%}/{d[n]['segments']}段"
                              for n in order if n in d)
            print(f"{title:<22}{cells}")
        sp = ours["musan/speech"]
        cells = "  ".join(f"{n}={sp[n]:.0%}" for n in order
                          if isinstance(sp.get(n), (int, float)))
        print(f"{'LibriVox 朗读检出覆盖:':<22}{cells}")


if __name__ == "__main__":
    main()
