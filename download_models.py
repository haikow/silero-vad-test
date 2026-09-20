# -*- coding: utf-8 -*-
"""下载两个量化版 Silero VAD 模型到 models/ 目录 (带国内镜像 fallback)"""
import os
import urllib.request

# 每个模型按顺序尝试: 国内镜像优先, 失败回退官方源
MODELS = {
    "silero_vad_uint8.onnx": [
        "https://hf-mirror.com/onnx-community/silero-vad/resolve/main/onnx/model_uint8.onnx",
        "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model_uint8.onnx",
    ],
    "silero_vad_int8.onnx": [
        "https://mirror.ghproxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx",
    ],
}

os.makedirs("models", exist_ok=True)
for name, urls in MODELS.items():
    dst = os.path.join("models", name)
    if os.path.exists(dst):
        print(f"已存在, 跳过: {dst}")
        continue
    for url in urls:
        for attempt in range(3):
            try:
                print(f"下载: {url}")
                urllib.request.urlretrieve(url, dst)
                print(f"完成: {dst} ({os.path.getsize(dst)} bytes)")
                break
            except Exception as e:
                print(f"  失败: {e}")
        else:
            continue
        break
    else:
        raise SystemExit(f"下载失败: {name}, 请手动下载上述任一 URL 到 models/")
print("全部完成")
