#!/usr/bin/env python3
"""wq_studio.py —— WQ7036 Mac 烧录+日志调试台(双页面)

用法: python3 wq_studio.py   (自动打开浏览器 http://127.0.0.1:8765)
依赖: pip3 install pyserial

页面一[烧录]: 选 wpk + 串口 → 握手→提速→分区表→逐镜像烧写→芯片校验→复位, 带进度条
页面二[日志]: 2M 8N1 实时日志, [SHIL] 绿色高亮 / [crash]·WDT 红色高亮,
              一键保存 + 一键验收(对比 golden_prob.f32, 线 0.1)

协议来源: 与 wq_burn.py 同源(SDK updater 源码 + 官方工具 DLL 对拍, 已实烧验证)
"""
import json
import os
import re
import struct
import sys
import threading
import time
import webbrowser
import zipfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial
import serial.tools.list_ports

# ============ 协议层(与 wq_burn.py 一致, 已实烧验证) ============
CMD_CONNECT, CMD_GET_VERSION, CMD_CHIP_RESET = 0, 1, 2
CMD_CHANGE_BAUDRATE, CMD_GET_SOC_ID, CMD_FACTORY_MODE = 4, 5, 8
CMD_BOOTMAP_SET, CMD_IMAGE_WRITE, CMD_IMAGE_VERIFY = 0x20, 0x21, 0x23
START, END = b"\x5c\x53", b"\x5c\x45"
WQ_IMAGE_MAGIC = 0xA4E49A17
CHUNK = 8188


def image_header(code, start=0, stack=0):
    return struct.pack("<5IB3xI4x", WQ_IMAGE_MAGIC, len(code), 0, start,
                       zlib.crc32(code) & 0xFFFFFFFF, 0, stack)


class WqSerial:
    def __init__(self, port, baud=115200, timeout=3.0):
        self.ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout)
        self.ser.reset_input_buffer()
        self.seq = 0

    def send(self, cmd, payload=b""):
        self.seq = (self.seq + 1) & 0xFFFF
        crc = zlib.crc32(payload) & 0xFFFF if payload else 0
        self.ser.write(START + struct.pack("<HHH", self.seq, cmd, len(payload)) +
                       payload + struct.pack("<H", crc) + END)
        self.ser.flush()

    def recv_frame(self, timeout=None):
        deadline = time.time() + (timeout or self.ser.timeout)
        buf = b""
        while time.time() < deadline:
            n = self.ser.in_waiting
            if n:
                buf += self.ser.read(n)
            else:
                time.sleep(0.01)
                continue
            i = buf.find(START)
            if i < 0:
                buf = buf[-1:]
                continue
            buf = buf[i:]
            if len(buf) < 10:
                continue
            cmd, result, plen = struct.unpack("<HHH", buf[2:8])
            total = 10 + plen + 4
            if len(buf) < total:
                continue
            return cmd, result, buf[10:10 + plen]
        return None

    def cmd(self, cmd_id, payload=b"", timeout=None, retries=1):
        for _ in range(retries):
            self.send(cmd_id, payload)
            r = self.recv_frame(timeout)
            if r and r[0] == cmd_id:
                return r[1], r[2]
        return None, None

    def close(self):
        self.ser.close()


# ============ 全局状态 ============
S = {
    "burn": {"running": False, "done": False, "ok": False, "step": "", "pct": 0,
             "logs": [], "wpk": ""},
    "log": {"running": False, "lines": [], "raw": bytearray(), "port": "", "saved": ""},
    "verify": {"state": "idle", "msg": "", "maxdev": None, "pass": None},
}
LOCK = threading.Lock()
LOG_THREAD = [None]


def blog(msg, pct=None):
    with LOCK:
        S["burn"]["logs"].append(msg)
        if pct is not None:
            S["burn"]["pct"] = pct


# ============ 烧录流程 ============
def burn_worker(port, wpk_path, hi_baud):
    try:
        S["burn"].update(running=True, done=False, ok=False, step="握手", pct=0, logs=[])
        z = zipfile.ZipFile(wpk_path)
        cfg = json.loads(z.read("memory_config.json"))
        blog(f"wpk: {os.path.basename(wpk_path)} | {cfg['chip']} | {len(cfg['images'])} 分区")

        s = WqSerial(port)
        blog("115200 握手中… 请确认板子已进下载模式(上电时 GPIO_101 为高)")
        ok = False
        for _ in range(10):
            r, _ = s.cmd(CMD_CONNECT, timeout=0.5)
            if r == 0:
                ok = True
                break
        if not ok:
            blog("✗ 握手失败:芯片无应答(未进下载模式/接线/串口不对)")
            return
        r, v = s.cmd(CMD_GET_VERSION)
        if r == 0 and v and len(v) >= 4:
            mj, mi = struct.unpack("<HH", v[:4])
            blog(f"✓ 芯片应答, updater v{mj}.{mi}")

        if hi_baud and hi_baud != 115200:
            S["burn"]["step"] = "提速"
            blog(f"提速到 {hi_baud} …")
            s.cmd(CMD_CHANGE_BAUDRATE, struct.pack("<II", hi_baud, 50), timeout=2)
            time.sleep(0.1)
            s.ser.baudrate = hi_baud
            s.ser.timeout = 10
            r, _ = s.cmd(CMD_CONNECT, timeout=1.5, retries=3)
            if r != 0:
                blog("提速失联,回退 115200")
                s.ser.baudrate = 115200
            else:
                blog("✓ 提速成功")
        s.cmd(CMD_FACTORY_MODE)
        blog("已取消 10 秒超时窗口(FACTORY_MODE)")

        blocks, nmap = b"", 0
        for im in cfg["images"]:
            if im["id"] == 8:
                continue
            blocks += struct.pack("<B3xIII", im["id"], im["lma"], im.get("vma", 0), im["length"])
            nmap += 1
        r, _ = s.cmd(CMD_BOOTMAP_SET, struct.pack("<I", 0) + blocks, timeout=30)
        blog(f"分区表({nmap} 项): {'✓' if r == 0 else '✗ result=' + str(r)}")
        if r != 0:
            return

        todo = [im for im in cfg["images"] if im["id"] != 8 and im.get("name", "") in z.namelist()]
        done_imgs, base = 0, 0
        for im in todo:
            name = im["name"]
            code = z.read(name)
            data = image_header(code, im.get("start", 0) or im.get("vma", 0), im.get("stack", 0)) + code
            total = (len(data) + CHUNK - 1) // CHUNK
            blog(f"烧写 {name} ({len(code):,}B, {total} 块)")
            for seq in range(total):
                pl = struct.pack("<BBH", im["id"], 0, seq) + data[seq * CHUNK:(seq + 1) * CHUNK]
                r, _ = s.cmd(CMD_IMAGE_WRITE, pl, timeout=90 if seq == 0 else 10, retries=3)
                if r != 0:
                    blog(f"✗ {name} 块 {seq} 失败 result={r}")
                    return
                pct = int((base + (seq + 1) / total * len(code)) / sum(len(z.read(i["name"])) for i in todo) * 100)
                S["burn"]["pct"] = min(99, pct)
            r, _ = s.cmd(CMD_IMAGE_VERIFY, struct.pack("<B3x", im["id"]), timeout=60)
            blog(f"  校验: {'✓ 通过' if r == 0 else '✗ FAIL(' + str(r) + ')'}")
            if r != 0:
                return
            base += len(code)
            done_imgs += 1

        S["burn"]["pct"] = 100
        blog("复位芯片…")
        s.cmd(CMD_CHIP_RESET, timeout=2)
        s.close()
        blog(f"✓✓ 烧录完成: {done_imgs} 个镜像全部写入并通过芯片侧校验")
        blog("下一步: 切到[日志]页 → 开始捕获 → 板子重新上电 → 看 [SHIL]")
        S["burn"].update(ok=True, done=True, running=False, step="完成")
    except Exception as e:
        blog(f"✗ 异常: {e}")
        S["burn"].update(done=True, running=False, step="异常")
    finally:
        S["burn"]["running"] = False


# ============ 日志捕获 ============
def log_worker(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.3)
        S["log"].update(running=True, port=port)
        while S["log"]["running"]:
            d = ser.read(4096)
            if d:
                with LOCK:
                    S["log"]["raw"].extend(d)
                    try:
                        for ln in d.decode("utf-8", "replace").splitlines():
                            S["log"]["lines"].append(ln)
                            if len(S["log"]["lines"]) > 20000:
                                del S["log"]["lines"][:5000]
                    except Exception:
                        pass
        ser.close()
    except Exception as e:
        with LOCK:
            S["log"]["lines"].append(f"[工具] 日志线程异常: {e}")
        S["log"]["running"] = False


# ============ 验收(hil_parse 逻辑) ============
def run_verify(golden_path):
    with LOCK:
        text = "\n".join(S["log"]["lines"])
    probs = [float(m.group(1)) for m in re.finditer(r"\[SHIL\] win=\d+ p=(\d+\.\d+)", text)]
    if not probs:
        S["verify"] = {"state": "done", "pass": False, "maxdev": None,
                       "msg": "日志里没有 [SHIL] win= 行——确认烧的是带自检的固件、板子重新上电过"}
        return
    g = open(golden_path, "rb").read()
    golden = struct.unpack(f"<{len(probs)}f", g[:len(probs) * 4])
    maxd = max(abs(a - b) for a, b in zip(probs, golden))
    crashed = "[crash]" in text or "WDT" in text.upper()
    ok = maxd <= 0.1 and not crashed
    msg = f"{len(probs)} 窗 vs PC 基准: 最大偏差 {maxd:.4f}"
    if crashed:
        msg += " | 检测到 [crash]/WDT 崩溃!"
    msg += "  → PASS ✅" if ok else "  → FAIL ❌(验收线 0.1)"
    S["verify"] = {"state": "done", "pass": ok, "maxdev": round(maxd, 4), "msg": msg}


# ============ HTTP 服务 ============
PAGE = """<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>
<title>WQ7036 烧录·日志调试台</title>
<style>
*{box-sizing:border-box}body{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#0f1115;color:#d7dce3}
header{padding:14px 22px;background:#161a22;border-bottom:1px solid #2a3040;display:flex;align-items:center;gap:16px}
header h1{font-size:17px;margin:0}header .dot{width:9px;height:9px;border-radius:50%;background:#3fb950}
main{max-width:980px;margin:22px auto;padding:0 16px}
.tabs{display:flex;gap:8px;margin-bottom:16px}
.tab{padding:9px 22px;border-radius:8px 8px 0 0;background:#1b2029;cursor:pointer;border:1px solid #2a3040;border-bottom:none}
.tab.on{background:#232a36;color:#fff;font-weight:600}
.panel{display:none;background:#161a22;border:1px solid #2a3040;border-radius:0 10px 10px 10px;padding:20px}
.panel.on{display:block}
.row{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
label{font-size:13px;color:#9aa4b2}
select,input[type=text],input[type=number]{background:#0f1115;color:#dfe6ee;border:1px solid #343c4c;border-radius:6px;padding:7px 10px;font-size:13px}
button{background:#2f81f7;color:#fff;border:none;border-radius:6px;padding:9px 20px;font-size:14px;cursor:pointer}
button:hover{background:#4493f8}button:disabled{background:#30363d;color:#777;cursor:not-allowed}
button.gray{background:#30363d}button.green{background:#2ea043}
.meter{height:22px;background:#0d1117;border-radius:6px;overflow:hidden;border:1px solid #2a3040}
.meter>div{height:100%;background:linear-gradient(90deg,#2f81f7,#3fb950);width:0;transition:width .3s;text-align:center;font-size:12px;line-height:22px;color:#fff}
pre{background:#0d1117;border:1px solid #2a3040;border-radius:8px;padding:12px;height:300px;overflow:auto;font-size:12px;line-height:1.55;font-family:Menlo,monospace;white-space:pre-wrap;word-break:break-all}
.logline{color:#c9d1d9}.shil{color:#3fb950;font-weight:600}.crash{color:#f85149;font-weight:600}.sys{color:#8b949e}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px}
.badge.ok{background:#12261e;color:#3fb950}.badge.bad{background:#2d1416;color:#f85149}
#verdict{margin-top:10px;padding:10px 14px;border-radius:8px;display:none;font-size:14px}
.hint{font-size:12px;color:#8b949e;margin:6px 0 0}
</style></head><body>
<header><div class=dot></div><h1>WQ7036 烧录 · 日志调试台</h1><span style="font-size:12px;color:#8b949e">协议源: SDK updater + 官方工具对拍</span></header>
<main>
<div class=tabs>
 <div class="tab on" onclick="sw(0)">📦 烧录固件</div>
 <div class="tab" onclick="sw(1)">📜 日志 / 验收</div>
</div>
<div class="panel on" id=p0>
 <div class=row>
  <label>wpk 文件</label><input type=file id=wpk accept=.wpk>
  <label>串口</label><select id=bport></select>
  <label>波特率</label><select id=bbaud><option>921600</option><option>460800</option><option selected>115200</option></select>
  <button class=green id=burnbtn onclick=burn()>▶ 开始烧录</button>
 </div>
 <div class=meter><div id=bbar>0%</div></div>
 <p class=hint>烧录前:板子断电 → 按住下载条件(GPIO_101 为高/板上下载键) → 上电进入下载模式,再点开始。</p>
 <pre id=blog>等待开始…</pre>
</div>
<div class="panel" id=p1>
 <div class=row>
  <label>串口</label><select id=lport></select>
  <label>波特率</label><input type=number id=lbaud value=2000000 step=1 style=width:110px>
  <button class=green onclick=logstart()>▶ 开始捕获</button>
  <button class=gray onclick=logstop()>■ 停止</button>
  <button class=gray onclick=logsave()>💾 保存日志</button>
  <button onclick=doverify()>✅ 一键验收</button>
  <input type=file id=golden accept=.f32 style=font-size:12px>
 </div>
 <p class=hint>日志口接法:RX→GPIO41(AX)/GPIO98(AC), GND→GND;先点开始捕获,再给板子上电。[SHIL] 绿色 / 崩溃红色。golden 留空则用内置基准。</p>
 <div id=verdict></div>
 <pre id=llog>等待捕获…</pre>
</div>
</main>
<script>
let goldenData=null, polling=false;
document.getElementById('golden').onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{goldenData=r.result;};r.readAsArrayBuffer(f);};
function sw(i){document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',i===j));document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('on',i===j));}
async function ports(){let r=await(await fetch('/api/ports')).json();let opts=r.ports.map(p=>`<option>${p}</option>`).join('');document.getElementById('bport').innerHTML=opts;document.getElementById('lport').innerHTML=opts;}
async function burn(){
 const f=document.getElementById('wpk').files[0]; if(!f){alert('先选 wpk 文件');return;}
 const port=document.getElementById('bport').value; if(!port){alert('未检测到串口');return;}
 const fd=new FormData(); fd.append('wpk',f); fd.append('port',port); fd.append('baud',document.getElementById('bbaud').value);
 document.getElementById('burnbtn').disabled=true;
 await fetch('/api/burn',{method:'POST',body:fd});
 pollBurn();
}
function pollBurn(){if(polling)return;polling=true;const t=setInterval(async()=>{
 let s=await(await fetch('/api/burn/status')).json();
 document.getElementById('bbar').style.width=s.pct+'%';document.getElementById('bbar').textContent=s.pct+'% '+s.step;
 document.getElementById('blog').textContent=s.logs.join('\\n');document.getElementById('blog').scrollTop=1e9;
 if(!s.running){clearInterval(t);polling=false;document.getElementById('burnbtn').disabled=false;}
},400);}
async function logstart(){
 const port=document.getElementById('lport').value;
 await fetch('/api/log/start',{method:'POST',body:JSON.stringify({port,baud:+document.getElementById('lbaud').value})});
 document.getElementById('llog').textContent='';
 pollLog();
}
async function logstop(){await fetch('/api/log/stop',{method:'POST'});}
async function logsave(){let r=await(await fetch('/api/log/save')).json();alert('已保存: '+r.path);}
let seen=0,lp=null;
function pollLog(){if(lp)return;lp=setInterval(async()=>{
 let r=await(await fetch('/api/log/data?after='+seen)).json();
 if(r.lines.length){seen+=r.lines.length;
  const pre=document.getElementById('llog');
  for(const ln of r.lines){const d=document.createElement('div');d.className='logline '+(/\\[SHIL\\]/.test(ln)?'shil':(/crash|WDT|assert|fault/i.test(ln)?'crash':(/^\\[/.test(ln)?'sys':'')));d.textContent=ln;pre.appendChild(d);}
  pre.scrollTop=1e9;}
},300);}
async function doverify(){
 let body={};
 if(goldenData){body.golden_b64=btoa(String.fromCharCode(...new Uint8Array(goldenData)));}
 const r=await(await fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 const v=document.getElementById('verdict');v.style.display='block';
 v.className=r.pass===null?'':'badge '+(r.pass?'ok':'bad');
 v.style.background=r.pass?'#12261e':'#2d1416';v.style.color=r.pass?'#3fb950':'#f85149';
 v.textContent=r.msg;
}
ports();setInterval(ports,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/ports":
            self.j({"ports": [p.device for p in serial.tools.list_ports.comports() if "usb" in p.device.lower() or "modem" in p.device.lower()]})
        elif self.path == "/api/burn/status":
            with LOCK:
                self.j(S["burn"])
        elif self.path.startswith("/api/log/data"):
            after = int(self.path.split("after=")[1])
            with LOCK:
                self.j({"lines": S["log"]["lines"][after:]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if self.path == "/api/burn":
            if S["burn"]["running"]:
                return self.j({"ok": False, "err": "已在烧录中"})
            ct = self.headers.get("Content-Type", "")
            port, baud, wpk_path = "", 921600, None
            if "multipart" in ct:
                body = self.rfile.read(n)
                boundary = ct.split("boundary=")[1].encode()
                for part in body.split(b"--" + boundary):
                    if b'name="wpk"' in part[:300] and b"filename=" in part[:300]:
                        head, _, data = part.partition(b"\r\n\r\n")
                        fm = re.search(rb'filename="([^"]*)"', head[:300])
                        if data.endswith(b"\r\n"):
                            data = data[:-2]
                        tmp = os.path.join("/tmp", os.path.basename(fm.group(1).decode()))
                        open(tmp, "wb").write(data)
                        wpk_path = tmp
                    elif b'name="port"' in part[:300]:
                        port = part.partition(b"\r\n\r\n")[2].strip().decode()
                    elif b'name="baud"' in part[:300]:
                        try:
                            baud = int(part.partition(b"\r\n\r\n")[2].strip())
                        except ValueError:
                            pass
            else:
                d = json.loads(self.rfile.read(n) or b"{}")
                port, wpk_path = d.get("port"), d.get("wpk")
            if not port or not wpk_path:
                return self.j({"ok": False, "err": "缺参数"})
            threading.Thread(target=burn_worker, args=(port, wpk_path, baud), daemon=True).start()
            self.j({"ok": True})
        elif self.path == "/api/log/start":
            d = json.loads(self.rfile.read(n) or b"{}")
            if S["log"]["running"]:
                return self.j({"ok": True, "note": "已在运行"})
            with LOCK:
                S["log"]["lines"].clear()
            LOG_THREAD[0] = threading.Thread(target=log_worker, args=(d["port"], d.get("baud", 2000000)), daemon=True)
            LOG_THREAD[0].start()
            self.j({"ok": True})
        elif self.path == "/api/log/stop":
            S["log"]["running"] = False
            self.j({"ok": True})
        elif self.path == "/api/log/save":
            p = os.path.expanduser(f"~/Desktop/shil_{time.strftime('%Y%m%d_%H%M%S')}.log")
            with LOCK:
                open(p, "w", errors="replace").write("\n".join(S["log"]["lines"]))
            self.j({"ok": True, "path": p})
        elif self.path == "/api/verify":
            import base64
            d = json.loads(self.rfile.read(n) or b"{}")
            gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_prob.f32")
            if d.get("golden_b64"):
                gp = "/tmp/golden_upload.f32"
                open(gp, "wb").write(base64.b64decode(d["golden_b64"]))
            if not os.path.exists(gp):
                return self.j({"state": "done", "pass": None,
                               "msg": "未找到 golden_prob.f32(可上传,或放到本脚本同目录)"})
            S["verify"].update(state="running")
            run_verify(gp)
            self.j(S["verify"])


def main():
    port_http = 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port_http), H)
    url = f"http://127.0.0.1:{port_http}"
    print(f"WQ 烧录·日志调试台 → {url}  (Ctrl-C 退出)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    srv.serve_forever()


if __name__ == "__main__":
    main()
