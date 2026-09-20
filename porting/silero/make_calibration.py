# 从测试音频生成 XNNC 校准/验证数据集(流式模式要求: 每输入一个目录, 逐窗 .npy)
# 用法: python3 make_calibration.py <音频目录或文件...>  <输出目录>
# 生成:
#   <输出目录>/x/seq_XXXXXX/YYYY.npy   每窗 float32 (1,512) —— 音频窗
#   <输出目录>/h/seq_XXXXXX/YYYY.npy   float32 (2,1,64) 零初始状态
#   <输出目录>/c/seq_XXXXXX/YYYY.npy   float32 (2,1,64) 零初始状态
#   <输出目录>/calibration_list.txt / validation_list.txt  (流式: 每行 x目录,h目录,c目录)
#   连续 3 窗以上语音段为一个 seq 目录(窗间状态传递), 段与段之间状态复位
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOW = 512          # 16k 下 32ms 窗
H_SHAPE = (2, 1, 64)  # silero v4 LSTM 状态
SPEECH_ENERGY_TH = 1e-4  # 粗略的语音/静音分段: 窗 RMS 平方阈值


def windows(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"{path}: 采样率 {sr} != 16000"
    for i in range(0, len(audio) - WINDOW, WINDOW):
        yield audio[i:i + WINDOW]


def main():
    srcs, out = sys.argv[1:-1], Path(sys.argv[-1])
    x_dir, h_dir, c_dir = out / "x", out / "h", out / "c"
    seqs = []  # [(seq_name, n_win)]
    seq_i = 0
    for src in srcs:
        p = Path(src)
        wavs = [p] if p.is_file() else sorted(p.glob("*.wav"))
        for wav in wavs:
            buf = []
            for w in windows(wav):
                buf.append(w)
            # 整个文件为一个 seq(状态连续), 文件之间状态复位
            if not buf:
                continue
            name = f"seq_{seq_i:04d}"
            (x_dir / name).mkdir(parents=True, exist_ok=True)
            (h_dir / name).mkdir(parents=True, exist_ok=True)
            (c_dir / name).mkdir(parents=True, exist_ok=True)
            for j, w in enumerate(buf):
                np.save(x_dir / name / f"{j:05d}.npy", w.reshape(1, WINDOW))
                np.save(h_dir / name / f"{j:05d}.npy", np.zeros(H_SHAPE, np.float32))
                np.save(c_dir / name / f"{j:05d}.npy", np.zeros(H_SHAPE, np.float32))
            seqs.append((name, len(buf)))
            seq_i += 1
    if not seqs:
        raise SystemExit("没有可用音频, 请检查输入")

    lines = [f"{x_dir / n},{h_dir / n},{c_dir / n}" for n, _ in seqs]
    n_cal = max(1, int(len(lines) * 0.6))
    (out / "calibration_list.txt").write_text("\n".join(lines[:n_cal]) + "\n")
    (out / "validation_list.txt").write_text("\n".join(lines[n_cal:]) + "\n")
    print(f"序列 {len(seqs)} 个, 共 {sum(c for _, c in seqs)} 窗; "
          f"校准 {n_cal} seq / 验证 {len(lines) - n_cal} seq -> {out}")


if __name__ == "__main__":
    main()
