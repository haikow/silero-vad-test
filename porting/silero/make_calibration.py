# 从测试音频生成 XNNC 校准/验证数据集(流式模式)
# 用法: python3 make_calibration.py <音频目录或文件...>  <输出目录>
# 生成(每个 wav 一个 seq 目录, 窗间状态传递, 段间状态复位):
#   <输出目录>/x/seq_XXXX/0.npy 1.npy ...   每窗 float32 (1,512)
#   <输出目录>/calibration_list.txt / validation_list.txt
#     —— 首行为序列数, 之后每行一个 x 序列目录(状态 h/c 由 state.yaml 管理,
#        XNNC 自动以零初始化, 不需要单独的数据目录)
# 注意: 文件名不能补零, 必须是 0.npy,1.npy,...(Analyzer 流式加载约定)
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOW = 512  # 16k 下 32ms 窗


def windows(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"{path}: 采样率 {sr} != 16000"
    for i in range(0, len(audio) - WINDOW, WINDOW):
        yield audio[i:i + WINDOW]


def main():
    srcs, out = sys.argv[1:-1], Path(sys.argv[-1])
    x_dir = out / "x"
    seqs = []
    for seq_i, src in enumerate(srcs):
        p = Path(src)
        wavs = [p] if p.is_file() else sorted(p.glob("*.wav"))
        for wav in wavs:
            buf = list(windows(wav))
            if not buf:
                continue
            name = f"seq_{len(seqs):04d}"
            d = x_dir / name
            d.mkdir(parents=True, exist_ok=True)
            for j, w in enumerate(buf):
                np.save(d / f"{j}.npy", w.reshape(1, WINDOW))
            seqs.append(name)
    if not seqs:
        raise SystemExit("没有可用音频, 请检查输入")

    lines = [str(x_dir / n) for n in seqs]
    n_cal = max(1, int(len(lines) * 0.6))
    cal, val = lines[:n_cal], lines[n_cal:] or lines[:1]
    (out / "calibration_list.txt").write_text(f"{len(cal)}\n" + "\n".join(cal) + "\n")
    (out / "validation_list.txt").write_text(f"{len(val)}\n" + "\n".join(val) + "\n")
    print(f"序列 {len(seqs)} 个; "
          f"校准 {len(cal)} seq / 验证 {len(val)} seq -> {out}")


if __name__ == "__main__":
    main()
