# -*- coding: utf-8 -*-
"""
Silero VAD 可视化 GUI: 生成单文件 HTML (波形 + 概率曲线 + 点击播放)
用法: python vad_gui.py <音频路径> [--max-seconds 300] [--no-open]
- 浏览器打开生成的 HTML: 曲线悬停看数值, 点击任意位置跳转播放, 播放游标跟随
- 完全离线, 音频以 base64 wav 内嵌, 无外部依赖
"""
import argparse
import base64
import io
import json
import os
import time
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf

from analyze_audio import load_audio
from test_vad import (MODELS, SAMPLE_RATE, THRESHOLD, WINDOW, VadRunner,
                      segments_from_probs)

LABELS = {"uint8 (onnx-community, 639KB)": "v5/uint8",
          "int8  (sherpa-onnx, 208KB)": "v4/int8"}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Silero VAD - __TITLE__</title>
<style>
  body { margin:0; background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; }
  .wrap { max-width:1200px; margin:0 auto; padding:16px; }
  h1 { font-size:18px; margin:8px 0; }
  .meta { color:#94a3b8; font-size:13px; margin-bottom:12px; }
  .cards { display:flex; gap:12px; margin-bottom:12px; flex-wrap:wrap; }
  .card { background:#1e293b; border-radius:8px; padding:10px 14px; min-width:150px; }
  .card .k { font-size:12px; color:#94a3b8; }
  .card .v { font-size:20px; font-weight:600; margin-top:2px; }
  .c1 { border-left:3px solid #38bdf8; } .c2 { border-left:3px solid #fb923c; }
  .c3 { border-left:3px solid #a78bfa; }
  #chart { width:100%; height:460px; background:#1e293b; border-radius:8px;
           display:block; cursor:crosshair; }
  .tip { position:fixed; pointer-events:none; background:#334155; border:1px solid #475569;
         border-radius:6px; padding:6px 10px; font-size:12px; display:none; z-index:9;
         white-space:nowrap; }
  .legend { display:flex; gap:18px; font-size:13px; margin:10px 2px; align-items:center; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;
         vertical-align:-1px; }
  .btn { background:#334155; color:#e2e8f0; border:1px solid #475569; border-radius:6px;
         padding:6px 16px; cursor:pointer; font-size:13px; }
  .btn:hover { background:#475569; }
  #status { font-size:12px; color:#94a3b8; margin-left:10px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Silero VAD 人声检测 <span style="color:#64748b">|</span> __TITLE__</h1>
  <div class="meta">__META__</div>
  <div class="cards">
    <div class="card c1"><div class="k">v5/uint8 语音占比 (阈值 __TH__)</div><div class="v" id="r1">-</div></div>
    <div class="card c2"><div class="k">v4/int8 语音占比 (阈值 __TH__)</div><div class="v" id="r2">-</div></div>
    <div class="card c3"><div class="k">音频时长</div><div class="v">__DUR__</div></div>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:#475569"></span>音频波形</span>
    <span><span class="dot" style="background:#38bdf8"></span>v5/uint8 概率</span>
    <span><span class="dot" style="background:#fb923c"></span>v4/int8 概率</span>
    <span><span class="dot" style="background:rgba(52,211,153,.35)"></span>检出人声区间</span>
    <span style="color:#64748b">|</span>
    <button class="btn" onclick="togglePlay()">播放 / 暂停</button>
    <span id="status">提示: 点击图上任意位置跳转播放</span>
  </div>
  <canvas id="chart"></canvas>
</div>
<div class="tip" id="tip"></div>
<audio id="au" src="__AUDIO__" preload="auto"></audio>
<script>
const D = __DATA__;
const au = document.getElementById('au');
const cv = document.getElementById('chart');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
const WIN_S = D.window / D.sr;

// 布局: 上 42% 波形, 中间时间轴, 下 58% 概率
function layout() {
  const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, waveH: h * 0.38, axisY: h * 0.38 + 18, probTop: h * 0.38 + 34,
           probH: h - (h * 0.38 + 34) - 14 };
}
const t2x = (t, L) => t / D.duration * L.w;
const x2t = (x, L) => x / L.w * D.duration;

function draw(playT) {
  const L = layout();
  ctx.clearRect(0, 0, L.w, L.h);
  // 检出区间背景 (取两模型并集)
  ctx.fillStyle = 'rgba(52,211,153,.20)';
  for (const [a, b] of D.segs_union) {
    const x1 = t2x(a, L), x2 = t2x(b, L);
    ctx.fillRect(x1, L.probTop - 6, x2 - x1, L.probH + 12);
  }
  // 波形
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 0.6;
  ctx.beginPath();
  const mid = L.waveH / 2, amp = L.waveH / 2 * 0.92;
  for (let i = 0; i < D.peaks.length; i++) {
    const x = i / (D.peaks.length - 1) * L.w;
    const [lo, hi] = D.peaks[i];
    ctx.moveTo(x, mid - hi * amp); ctx.lineTo(x, mid - lo * amp + 0.5);
  }
  ctx.stroke();
  ctx.strokeStyle = '#334155'; ctx.beginPath();
  ctx.moveTo(0, mid); ctx.lineTo(L.w, mid); ctx.stroke();
  // 时间轴刻度 (首尾标签避免被裁切)
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px system-ui';
  const step = D.duration > 300 ? 60 : D.duration > 60 ? 15 : 5;
  for (let t = 0; t <= D.duration; t += step) {
    const x = t2x(t, L);
    ctx.textAlign = x < 16 ? 'left' : x > L.w - 16 ? 'right' : 'center';
    ctx.fillText(D.duration > 3600 ? (t/60).toFixed(0)+'m' : t.toFixed(0)+'s',
                 Math.min(Math.max(x, 2), L.w - 2), L.axisY + 13);
    ctx.strokeStyle = '#334155'; ctx.beginPath();
    ctx.moveTo(x, L.axisY - 4); ctx.lineTo(x, L.axisY); ctx.stroke();
  }
  // 概率区网格+阈值
  ctx.strokeStyle = 'rgba(148,163,184,.12)';
  for (const p of [0.25, 0.5, 0.75]) {
    const y = L.probTop + L.probH - p * L.probH;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(L.w, y); ctx.stroke();
  }
  const thy = L.probTop + L.probH - D.threshold * L.probH;
  ctx.strokeStyle = 'rgba(248,113,113,.95)'; ctx.lineWidth = 1.4; ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(0, thy); ctx.lineTo(L.w, thy); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#f87171'; ctx.textAlign = 'right'; ctx.font = 'bold 11px system-ui';
  ctx.fillText('阈值 ' + D.threshold, L.w - 8, thy - 5);
  // 两条概率曲线
  const curves = [['uint8', '#38bdf8'], ['int8', '#fb923c']];
  for (const [key, color] of curves) {
    const ps = D.probs[key]; if (!ps) continue;
    const n = ps.length, xw = D.duration / n;
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * xw / D.duration * L.w;
      const y = L.probTop + L.probH - ps[i] * L.probH * 0.96 - 2;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    // 曲线下半透明填充
    ctx.globalAlpha = 0.10; ctx.lineTo(L.w, L.probTop + L.probH); ctx.lineTo(0, L.probTop + L.probH);
    ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
  }
  // 播放游标
  if (playT != null) {
    const x = t2x(playT, L);
    ctx.strokeStyle = '#facc15'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, L.h - 8); ctx.stroke();
  }
  ctx.textAlign = 'left'; ctx.fillStyle = '#64748b';
  ctx.fillText('波形', 8, 16);
  ctx.fillText('人声概率', 8, L.probTop + 14);
}

function loop() { draw(au.paused ? null : au.currentTime); requestAnimationFrame(loop); }
requestAnimationFrame(loop);

cv.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect(), L = layout();
  const t = Math.max(0, Math.min(x2t(e.clientX - r.left, L), D.duration));
  const i = Math.min(D.probs.uint8.length - 1, Math.floor(t / WIN_S));
  tip.style.display = 'block';
  tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 10) + 'px';
  tip.innerHTML = `t = ${t.toFixed(2)}s` +
    (D.probs.uint8 ? `<br><span style="color:#38bdf8">v5: ${D.probs.uint8[i].toFixed(3)}</span>` : '') +
    (D.probs.int8 ? `<br><span style="color:#fb923c">v4: ${D.probs.int8[i].toFixed(3)}</span>` : '');
});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');
cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect(), L = layout();
  au.currentTime = Math.max(0, x2t(e.clientX - r.left, L));
  au.play();
});
function togglePlay() { au.paused ? au.play() : au.pause(); }
window.addEventListener('resize', () => draw(au.paused ? null : au.currentTime));

document.getElementById('r1').textContent = (D.stats.uint8 * 100).toFixed(1) + '%';
document.getElementById('r2').textContent = (D.stats.int8 * 100).toFixed(1) + '%';
</script>
</body>
</html>
"""


def build_gui(audio_path: str, max_seconds: float, open_browser: bool = True):
    audio = load_audio(audio_path, max_seconds=max_seconds)
    duration = len(audio) / SAMPLE_RATE

    probs_out, segs_all = {}, {}
    for name, path in MODELS.items():
        runner = VadRunner(path)
        t0 = time.perf_counter()
        probs = [runner.process(audio[i:i + WINDOW])
                 for i in range(0, len(audio) - WINDOW, WINDOW)]
        probs = np.array(probs)
        print(f"{LABELS[name]}: 推理完成, 每窗 {(time.perf_counter()-t0)/len(probs)*1000:.2f} ms")
        probs_out["uint8" if "uint8" in name else "int8"] = [
            round(float(v), 4) for v in probs]
        segs_all[name] = segments_from_probs(probs, np.arange(len(probs))
                                             * WINDOW / SAMPLE_RATE, THRESHOLD)

    union = sorted([s for segs in segs_all.values() for s in segs])
    merged = []
    for a, b in union:  # 合并重叠区间
        if merged and a <= merged[-1][1] + 0.1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([float(a), float(b)])

    # 波形峰值 (min/max), ~4000 桶
    n_px = 4000
    chunk = max(1, len(audio) // n_px)
    peaks = [[round(float(audio[i:i+chunk].min()), 3),
              round(float(audio[i:i+chunk].max()), 3)]
             for i in range(0, len(audio), chunk)]

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    data = {
        "duration": round(duration, 2), "sr": SAMPLE_RATE, "window": WINDOW,
        "threshold": THRESHOLD, "peaks": peaks, "probs": probs_out,
        "stats": {("uint8" if "uint8" in k else "int8"):
                  round(float((np.array(v) >= THRESHOLD).mean()), 4)
                  for k, v in probs_out.items()},
        "segs_union": merged,
    }

    html = (HTML_TEMPLATE
            .replace("__TITLE__", Path(audio_path).name)
            .replace("__META__", f"{audio_path} | {duration:.1f}s | "
                     f"{SAMPLE_RATE} Hz mono | 窗口 {WINDOW} 样本 "
                     f"({WINDOW/SAMPLE_RATE*1000:.0f} ms)")
            .replace("__TH__", str(THRESHOLD))
            .replace("__DUR__", f"{duration:.0f}s")
            .replace("__AUDIO__", f"data:audio/wav;base64,{audio_b64}")
            .replace("__DATA__", json.dumps(data)))
    out = Path(audio_path).with_name(Path(audio_path).stem + "_vad_gui.html")
    out.write_text(html, encoding="utf-8")
    print(f"已生成: {out} ({out.stat().st_size/1e6:.1f} MB)")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silero VAD 可视化 GUI")
    ap.add_argument("audio", help="音频文件 (wav/mp3/m4a...)")
    ap.add_argument("--max-seconds", type=float, default=300,
                    help="长音频截断分析时长 (默认 300s)")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()
    build_gui(a.audio, a.max_seconds, not a.no_open)
