# -*- coding: utf-8 -*-
"""官方协议回归 (FLEURS-VAD 式): 模型健康度基线 (区别于部署口径的场景矩阵)

素材 fleurs_like_spk_{a,b}: LibriSpeech 朗读 + 1s 停顿(填 -45dBFS 底噪), 拼接边界即真值。
2026-10-02 实测 (Mac M3, 流式 checkpoint, 帧级):
  Silero uint8/int8: F1 89~93% @0.5, FAR 0~2%, AUC 0.90~0.97 (uint8 最高)
  FireRed 流式: F1 93~95% @0.5 但 FAR 100% —— 概率下限焊死 0.50 (数字零/底噪均 >=0.5),
    帧级阈值无区分力; AUC 0.92~0.94
  WebRTC: 连 -45dB 底噪都全开 (FAR 77~92%, aggr3 27%) —— FLEURS 停顿必须真安静
官方 97.6 的差异来源: 官方头条数字大概率出自非流式 VAD 头 (官方发布 VAD + Stream-VAD
两个模型, Engineering ONNX 仅流式), 且标注集未开源。此处守住的是"协议级"基线。
"""
import pytest

INT8_F1_MIN = 0.85          # 实测 0.885/0.901 @0.5
INT8_FAR_MAX = 0.05         # 实测 0.020/0.012
U8_AUC_MIN = 0.93           # 实测 0.9446/0.9667
FR_AUC_MIN = 0.90           # 实测 0.9150/0.9429
FR_DEFAULT_F1_MIN = 0.90    # 实测 0.927/0.950 (FAR=100% 但语音占比高时的上限值)
WRTC_FLOOR_FAR_MIN = 0.60   # 实测 aggr0-2 FAR 0.78~0.92: 底噪全开 (特征化)


def test_fleurs_fixtures_present():
    from pathlib import Path
    for stem in ("fleurs_like_spk_a", "fleurs_like_spk_b"):
        assert Path(f"test_fixtures/{stem}.wav").exists()
        assert Path(f"test_fixtures/{stem}.gt.json").exists()


@pytest.mark.parametrize("stem", ["fleurs_like_spk_a", "fleurs_like_spk_b"])
def test_official_silero_int8_f1(stem, fleurs_metrics):
    """模型健康度: int8 在官方式干净素材上 F1 >= 0.85 且 FAR <= 5% (阈值 0.5)"""
    r = fleurs_metrics[stem]["Silero-int8"]
    assert r["default"][0] >= INT8_F1_MIN, f"{stem}: int8 F1 {r['default'][0]:.1%} < {INT8_F1_MIN:.0%}"
    assert r["default"][1] <= INT8_FAR_MAX, f"{stem}: int8 FAR {r['default'][1]:.1%} > {INT8_FAR_MAX:.0%}"


@pytest.mark.parametrize("stem", ["fleurs_like_spk_a", "fleurs_like_spk_b"])
def test_official_uint8_best_auc(stem, fleurs_metrics):
    """uint8 概率分离度最高 (实测 AUC 0.94/0.97) —— 高查准画像的概率层证据"""
    a = fleurs_metrics[stem]["Silero-uint8"]["auc"]
    assert a >= U8_AUC_MIN, f"{stem}: uint8 AUC {a:.4f} < {U8_AUC_MIN}"


@pytest.mark.parametrize("stem", ["fleurs_like_spk_a", "fleurs_like_spk_b"])
def test_official_firered_probability_floor(stem, fleurs_metrics):
    """特征化: FireRed 流式头 @0.5 阈值 FAR=100% (概率下限焊死 0.50), 但 AUC/排序能力正常
    —— 这是"必须单调更高阈值"的原理性证据, 而非我们设置错误"""
    r = fleurs_metrics[stem]["FireRed"]
    assert r["default"][1] >= 0.99, \
        f"{stem}: FireRed FAR {r['default'][1]:.1%}, 概率下限行为变了, 请重标定其工作点"
    assert r["auc"] >= FR_AUC_MIN, f"{stem}: FireRed AUC {r['auc']:.4f} < {FR_AUC_MIN}"


@pytest.mark.parametrize("stem", ["fleurs_like_spk_a", "fleurs_like_spk_b"])
def test_official_firered_ranking_sane(stem, fleurs_metrics):
    assert fleurs_metrics[stem]["FireRed"]["default"][0] >= FR_DEFAULT_F1_MIN


@pytest.mark.parametrize("stem", ["fleurs_like_spk_a", "fleurs_like_spk_b"])
def test_official_wrtc_floor_sensitivity(stem, fleurs_metrics):
    """特征化: WebRTC 对 -45dB 麦克风底噪全开 (FLEURS 官方 FAR 2.8% 要求真数字安静)"""
    far = fleurs_metrics[stem]["WRTC-0"]["default"][1]
    assert far >= WRTC_FLOOR_FAR_MIN, \
        f"{stem}: WebRTC-0 FAR {far:.1%} < {WRTC_FLOOR_FAR_MIN:.0%}, 行为变了"
