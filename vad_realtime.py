# -*- coding: utf-8 -*-
"""
Silero VAD 实时演示 (Siri 风格): 麦克风 -> 实时 VAD -> 浏览器跳动波形
用法: python vad_realtime.py [--gate v5|v4] [--no-open]
- 只有检测到人声时波形才跳动, 嘈杂声音波形几乎不动
- v5/uint8 与 v4/int8 两个模型同时推理, 概率条实时对比
- 页面: http://127.0.0.1:8766/  WebSocket: ws://127.0.0.1:8767/
依赖: pip install sounddevice websockets
"""
import argparse
import asyncio
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

from test_vad import MODELS, SAMPLE_RATE, WINDOW, VadRunner

HTTP_PORT, WS_PORT = 8766, 8767
N_BARS = 56          # 前端波形条数量
ENV_POINTS = 56      # 每窗送的包络点数
DEBUG_WINDOWS = 400  # /debug 保留的最近窗数 (~13 秒)

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Silero VAD 实时演示</title>
<style>
  body { margin:0; background:#0b1020; color:#e2e8f0; font-family:system-ui,sans-serif;
         min-height:100vh; display:flex; flex-direction:column; align-items:center; }
  h1 { font-size:20px; margin:26px 0 4px; letter-spacing:1px; }
  .sub { color:#64748b; font-size:13px; margin-bottom:14px; }
  .badge { display:inline-flex; align-items:center; gap:8px; padding:5px 16px;
           border-radius:999px; background:#1e293b; font-size:14px; margin-bottom:10px;
           border:1px solid #334155; }
  .dot { width:10px; height:10px; border-radius:50%; background:#475569; }
  .dot.on { background:#34d399; box-shadow:0 0 10px #34d399; animation:pulse 1s infinite; }
  @keyframes pulse { 50% { box-shadow:0 0 22px #34d399; } }
  #bars { background:radial-gradient(ellipse at center,#141c36 0%,#0b1020 75%);
          border-radius:16px; }
  .gauges { width:640px; max-width:92vw; margin-top:6px; }
  .g { margin:10px 0; }
  .g .lab { display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px; }
  .track { position:relative; height:14px; background:#1e293b; border-radius:7px; }
  .fill { position:absolute; left:0; top:0; bottom:0; border-radius:7px; width:0;
          transition:width .06s linear; }
  .f5 { background:linear-gradient(90deg,#0ea5e9,#38bdf8); }
  .f4 { background:linear-gradient(90deg,#ea580c,#fb923c); }
  .th { position:absolute; top:-3px; bottom:-3px; width:2px; background:#f87171; left:50%; }
  .th::after { content:'0.5'; position:absolute; top:-16px; left:-10px; color:#f87171;
               font-size:11px; }
  #hist { margin-top:14px; background:#141c36; border-radius:12px; }
  .hint { color:#64748b; font-size:12px; margin:12px 0 24px; text-align:center; }
  .conn { color:#f87171; font-size:13px; margin-top:8px; display:none; }
</style>
</head>
<body>
  <h1>SILERO VAD 实时演示</h1>
  <div class="sub">对着麦克风说话 → 波形跳动；嘈杂声音 → 波形静止</div>
  <div class="badge"><span class="dot" id="dot"></span><span id="st">等待音频…</span></div>
  <canvas id="bars" width="640" height="240"></canvas>
  <div class="gauges">
    <div class="g"><div class="lab"><span style="color:#38bdf8">v5/uint8 人声概率</span>
      <span id="p5">0.00</span></div>
      <div class="track"><div class="fill f5" id="f5"></div><div class="th"></div></div></div>
    <div class="g"><div class="lab"><span style="color:#fb923c">v4/int8 人声概率</span>
      <span id="p4">0.00</span></div>
      <div class="track"><div class="fill f4" id="f4"></div><div class="th"></div></div></div>
  </div>
  <canvas id="hist" width="640" height="120"></canvas>
  <div class="hint">上图: 实时波形 (人声门控) &nbsp;|&nbsp; 中: 两模型实时概率 &nbsp;|&nbsp;
    下: 最近 12 秒概率历史 &nbsp;|&nbsp; 门控模型: <b id="gm">v5</b></div>
  <div class="conn" id="conn">与服务的连接已断开, 请确认 vad_realtime.py 正在运行</div>
<script>
const W = 64, NBAR = 56;
const barsC = document.getElementById('bars').getContext('2d');
const histC = document.getElementById('hist').getContext('2d');
const hist5 = [], hist4 = [];
let cur = {p5:0, p4:0, wave:new Array(NBAR).fill(0.02), has:false};
let gain = 0, bh = new Array(NBAR).fill(2);
const GATE = {v5:'p5', v4:'p4'}['__GATE__'];
document.getElementById('gm').textContent = '__GATE__';

function connect() {
  const ws = new WebSocket('ws://127.0.0.1:__WSPORT__/');
  ws.onopen = () => document.getElementById('conn').style.display = 'none';
  ws.onclose = () => { document.getElementById('conn').style.display = 'block';
                       setTimeout(connect, 1500); };
  ws.onmessage = e => { cur = JSON.parse(e.data); };
}
connect();

function loop() {
  // 门控: 概率 0.3 以下全关, 0.6 以上全开; 快开慢关
  const p = cur[GATE] ?? 0;
  const target = Math.max(0, Math.min(1, (p - 0.3) / 0.3));
  gain += (target - gain) * (target > gain ? 0.4 : 0.06);

  // ---- 波形条 (Siri 风格, 上下对称) ----
  const cw = 640, ch = 240;
  barsC.clearRect(0, 0, cw, ch);
  const bw = cw / NBAR * 0.55, gap = cw / NBAR;
  const mid = ch / 2;
  for (let i = 0; i < NBAR; i++) {
    const env = cur.wave[i] ?? 0;
    const th = Math.max(3, Math.pow(env, 0.75) * (mid - 14) * (0.15 + 0.85 * gain));
    bh[i] += (th - bh[i]) * 0.45;
    const h = bh[i];
    const x = i * gap + (gap - bw) / 2;
    const hue = 199 - 28 * gain;
    const grad = barsC.createLinearGradient(0, mid - h, 0, mid + h);
    const a = 0.25 + 0.75 * gain;
    grad.addColorStop(0, `hsla(${hue},95%,68%,${a})`);
    grad.addColorStop(0.5, `hsla(${hue+10},95%,60%,${a})`);
    grad.addColorStop(1, `hsla(${hue},95%,68%,${a})`);
    barsC.fillStyle = gain < 0.03 ? 'rgba(71,85,105,.35)' : grad;
    const r = Math.min(bw / 2, 4);
    roundRect(barsC, x, mid - h, bw, h * 2, r); barsC.fill();
  }
  // ---- 概率仪表 ----
  document.getElementById('p5').textContent = (cur.p5||0).toFixed(2);
  document.getElementById('p4').textContent = (cur.p4||0).toFixed(2);
  document.getElementById('f5').style.width = ((cur.p5||0)*100) + '%';
  document.getElementById('f4').style.width = ((cur.p4||0)*100) + '%';
  const on = (cur[GATE] ?? 0) >= 0.5;
  const st = document.getElementById('st');
  document.getElementById('dot').className = 'dot' + (on ? ' on' : '');
  if (cur.err) { st.textContent = '错误: ' + cur.err; st.style.color = '#f87171'; }
  else if (!cur.has) { st.textContent = '等待音频… (检查麦克风是否被占用/禁用)';
                       st.style.color = '#94a3b8'; }
  else { st.textContent = on ? '检测到人声' : '环境音 (未触发)';
         st.style.color = ''; }
  // ---- 历史曲线 (12s) ----
  if (cur.has) { hist5.push(cur.p5||0); hist4.push(cur.p4||0);
                 if (hist5.length > 380) { hist5.shift(); hist4.shift(); } }
  const hw = 640, hh = 120;
  histC.clearRect(0, 0, hw, hh);
  histC.strokeStyle = 'rgba(248,113,113,.8)'; histC.setLineDash([5,4]);
  histC.beginPath(); histC.moveTo(0, hh-0.5*hh); histC.lineTo(hw, hh-0.5*hh);
  histC.stroke(); histC.setLineDash([]);
  for (const [arr, color] of [[hist5,'#38bdf8'],[hist4,'#fb923c']]) {
    if (arr.length < 2) continue;
    histC.strokeStyle = color; histC.lineWidth = 1.6; histC.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = i / 379 * hw, y = hh - arr[i] * (hh - 8) - 4;
      i ? histC.lineTo(x, y) : histC.moveTo(x, y);
    }
    histC.stroke();
  }
  requestAnimationFrame(loop);
}
function roundRect(c, x, y, w, h, r) {
  c.beginPath(); c.moveTo(x+r, y);
  c.arcTo(x+w, y, x+w, y+h, r); c.arcTo(x+w, y+h, x, y+h, r);
  c.arcTo(x, y+h, x, y, r); c.arcTo(x, y, x+w, y, r); c.closePath();
}
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def make_stream(gate: str):
    runners = {k: VadRunner(p) for k, p in MODELS.items()}
    state = {"p5": 0.0, "p4": 0.0, "wave": [0.02] * ENV_POINTS,
             "has": False, "err": "", "count": 0, "resets": 0,
             "frames": {}}
    lock = threading.Lock()
    quiet = 0           # v5 连续静音窗计数
    ring = []           # 最近 DEBUG_WINDOWS 窗原始数据, 供 /debug 导出
    frames_seen = {}    # 回调实际收到的样本数分布 (排查 blocksize 问题)
    agc = {"gain": 1.0}  # 自动增益状态

    def callback(indata, frames, time_info, status):
        nonlocal quiet
        try:
            n_raw = len(indata)
            frames_seen[n_raw] = frames_seen.get(n_raw, 0) + 1
            x = indata[:, 0].astype(np.float32)
            if len(x) != WINDOW:  # 48k 回调则 3:1 降采样到 16k
                x = x.reshape(WINDOW, len(x) // WINDOW).mean(axis=1)
            # AGC: v5/uint8 对低电平语音响应差。用对电平鲁棒的 v4 判定语音,
            # 只在语音期间校准增益 (目标 RMS ~0.06, 上限 8x), 噪声期增益冻结
            p4_raw = None
            for name, runner in runners.items():
                if "int8" in name:
                    p4_raw = runner.process(x)
            rms_in = float(np.sqrt(np.mean(x ** 2)))
            if p4_raw is not None and p4_raw > 0.3:
                target_gain = min(8.0, max(1.0, 0.06 / max(rms_in, 1e-4)))
                agc["gain"] += 0.05 * (target_gain - agc["gain"])
            xg = np.clip(x * agc["gain"], -1.0, 1.0).astype(np.float32)
            seg = max(1, len(x) // ENV_POINTS)
            env = np.abs(xg[:seg * ENV_POINTS].reshape(ENV_POINTS, seg)).max(axis=1)
            p = {"p4": round(float(p4_raw), 4)}
            for name, runner in runners.items():
                if "uint8" in name:
                    p["p5"] = round(float(runner.process(xg)), 4)
            # v5 已知特性: 长时间环境音/静音后 LSTM 状态漂移变迟钝,
            # 连续 3s 低概率即复位状态 (官方流式用法也建议定期 reset)
            quiet = quiet + 1 if max(p["p5"], p["p4"]) < 0.15 else 0
            if quiet >= int(3 * SAMPLE_RATE / WINDOW):
                for r in runners.values():
                    r.reset()
                quiet = 0
                with lock:
                    state["resets"] += 1
            with lock:
                state.update(p, wave=[round(float(v), 4) for v in env],
                             has=True, count=state["count"] + 1,
                             frames=dict(frames_seen), gain=round(agc["gain"], 2))
                ring.append((float(np.sqrt(np.mean(xg ** 2))),
                             p["p5"], p["p4"], xg.copy(),
                             hash(xg.tobytes())))
                if len(ring) > DEBUG_WINDOWS:
                    ring.pop(0)
        except Exception as e:  # 异常必须可见, 不能静默吞掉
            with lock:
                state["err"] = repr(e)
            print(f"音频回调异常: {e!r}")

    # 优先 16k 直采; 设备不支持则 48k + 回调内 3:1 降采样
    for sr, block in ((SAMPLE_RATE, WINDOW), (48000, WINDOW * 3)):
        try:
            stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                    blocksize=block, callback=callback)
            stream.start()
            return stream, state, lock, ring, sr
        except sd.PortAudioError as e:
            print(f"采样率 {sr} 打开失败: {e}")
    raise SystemExit("没有可用的麦克风输入设备")


async def ws_serve(state, lock, gate):
    async def handler(ws):
        print(f"浏览器已连接: {ws.remote_address}")
        try:
            while True:
                with lock:
                    payload = json.dumps(state)
                await ws.send(payload)
                await asyncio.sleep(0.032)
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(handler, "127.0.0.1", WS_PORT):
        print(f"WebSocket 服务: ws://127.0.0.1:{WS_PORT}/")
        await asyncio.Future()  # run forever


class PageHandler(BaseHTTPRequestHandler):
    ring = None   # 由 main 注入
    lock = None
    state = None

    def do_GET(self):
        if self.path == "/":
            body = (HTML.replace("__GATE__", PageHandler.gate)
                        .replace("__WSPORT__", str(WS_PORT))
                        .encode("utf-8"))
            ctype = "text/html; charset=utf-8"
        elif self.path.startswith("/debug") and self.ring is not None:
            with self.lock:  # 快照, 避免与音频回调竞争
                snap = list(self.ring)
                frames = dict(self.state.get("frames", {})) if self.state else {}
            if self.path == "/debug":
                body = json.dumps({
                    "windows": len(snap),
                    "callback_frames_histogram": frames,
                    "resets": self.state.get("resets", 0) if self.state else 0,
                    "per_window": [[round(r, 5), p5, p4, h]
                                   for r, p5, p4, _, h in snap],
                }).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:  # /debug.wav
                import io
                audio = np.concatenate([s[3] for s in snap]) \
                    if snap else np.zeros(0, dtype=np.float32)
                buf = io.BytesIO()
                sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                body = buf.getvalue()
                ctype = "audio/wav"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 静默访问日志
        pass


def main():
    ap = argparse.ArgumentParser(description="Silero VAD 实时演示 (Siri 风格)")
    ap.add_argument("--gate", choices=["v5", "v4"], default="v5",
                    help="波形门控用哪个模型的概率 (默认 v5)")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()
    PageHandler.gate = a.gate

    stream, state, lock, ring, sr = make_stream(a.gate)
    print(f"麦克风已启动 (采样率 {sr}), 门控模型: {a.gate}, Ctrl+C 退出")
    PageHandler.ring = ring
    PageHandler.lock = lock
    PageHandler.state = state

    httpd = HTTPServer(("127.0.0.1", HTTP_PORT), PageHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{HTTP_PORT}/"
    print(f"页面服务: {url}")
    if not a.no_open:
        time.sleep(0.5)
        webbrowser.open(url)
    try:
        asyncio.run(ws_serve(state, lock, a.gate))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stream.stop(); stream.close(); httpd.shutdown()


if __name__ == "__main__":
    main()
