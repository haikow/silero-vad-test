# -*- coding: utf-8 -*-
"""下载两个量化版 Silero VAD 模型到 models/ 目录"""
import os
import urllib.request

MODELS = {
    "silero_vad_uint8.onnx": "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model_uint8.onnx",
    "silero_vad_int8.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx",
}

os.makedirs("models", exist_ok=True)
for name, url in MODELS.items():
    dst = os.path.join("models", name)
    if os.path.exists(dst):
        print(f"已存在, 跳过: {dst}")
        continue
    print(f"下载: {url}")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"完成: {dst} ({os.path.getsize(dst)} bytes)")
            break
        except Exception as e:
            print(f"  第{attempt+1}次失败: {e}, 重试...")
    else:
        raise SystemExit(f"下载失败: {name}")
print("全部完成")
