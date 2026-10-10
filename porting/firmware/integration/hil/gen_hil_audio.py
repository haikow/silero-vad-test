#!/usr/bin/env python3
"""gen_hil_audio.py —— 从 golden_x.f32 生成 HIL 自环音频头(int16)
注意: 量化必须是 ×32768(与固件 /32768.0f 互逆, golden 音频本身在 int16 网格上则无损)
     此前版本误用 ×32767, 每个非零样本朝零偏 1 LSB → 板端 stft 全局偏 ~0.05% (HH 实测定位)
"""
import sys
import numpy as np

BASE = "/Users/a1234/.zcode/workspace/default/wq7036-silero-vad/firmware-silero"
OUT = sys.argv[1] if len(sys.argv) > 1 else "silero_hil_audio.h"
X = np.fromfile(f"{BASE}/golden_x.f32", np.float32)
assert len(X) % 512 == 0
S = np.round(X * 32768.0).astype(np.int64)
assert np.abs(S).max() < 32768
back = S / 32768.0
assert np.array_equal(back.astype(np.float32), X), "golden 不在 int16 网格上"
S = S.astype(np.int16)
with open(OUT, "w") as f:
    f.write("/* 自动生成: HIL 自环黄金音频(smoke_test 前 40 窗, int16)勿手改\n"
            "   生成: gen_hil_audio.py — ×32768 量化(勿用 32767, 见脚本头注释) */\n")
    f.write("#ifndef WQ_SILERO_HIL_AUDIO_H\n#define WQ_SILERO_HIL_AUDIO_H\n")
    f.write(f"#define SILERO_HIL_WINDOWS {len(X)//512}\n")
    f.write("static const short silero_hil_audio[%d] = {\n" % len(S))
    for i in range(0, len(S), 16):
        f.write(",".join(str(v) for v in S[i:i+16]) + ",\n")
    f.write("};\n#endif\n")
print(f"已生成 {OUT}: {len(S)} 样本, 与 golden 逐位一致(无损)")
