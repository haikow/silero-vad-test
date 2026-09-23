#!/usr/bin/env python3
"""wq_burn.py —— 物奇 WQ7036AX/AC Mac/Linux 原生烧录器(Silero VAD 联调用)

协议来源(全部实证):
  - 芯片侧: wq-audio SDK wqcore/pre-project/recover/components/updater/ 源码
  - PC 侧:  官方研发烧录工具 Wuqi.Uart.dll 命令枚举(1:1 对应)
帧格式:  5C 53 + {u16 seq, u16 cmd, u16 len} + payload + u16LE(crc32(payload)&0xFFFF) + 5C 45
镜像格式: 36B wq_image_header_t(guard=0xA4E49A17, ..., crc=zlib.crc32(bin)) + bin 原文
用法:
  python3 wq_burn.py probe  [-p PORT]              # 只读握手, 不写任何东西
  python3 wq_burn.py sniff  [-p PORT] [-b BAUD]    # 原始收 3 秒, 判断接的是哪个口
  python3 wq_burn.py burn   <wpk> [-p PORT] [--baud 高速] [--keep-baud]
  python3 wq_burn.py shil   [-p PORT] [-o 输出文件]  # 2M 抓 [SHIL] 日志
依赖: pip3 install pyserial   (清华源: -i https://pypi.tuna.tsinghua.edu.cn/simple)
"""
import argparse
import json
import struct
import sys
import time
import zipfile
import zlib

import serial

# ---- 协议常量(updater_command_id.h) ----
CMD_CONNECT = 0
CMD_GET_VERSION = 1
CMD_CHIP_RESET = 2
CMD_CHANGE_BAUDRATE = 4
CMD_GET_SOC_ID = 5
CMD_FACTORY_MODE = 8
CMD_BOOTMAP_SET = 0x20
CMD_IMAGE_WRITE = 0x21
CMD_IMAGE_VERIFY = 0x23

START = b"\x5c\x53"
END = b"\x5c\x45"
WQ_IMAGE_MAGIC = 0xA4E49A17
HDR_LEN = 32  # IMAGE_HEADER_LEN 0x20, 见 recover/bbb/memory_config.h (wq_image_header_t packed=32B)
CHUNK = 8188  # 每块数据字节数(4B 命令头 + 8188 = 8192 payload < 0x4020 上限)
HANDSHAKE_BAUD = 115200

DEFAULT_PORT = "/dev/cu.usbserial-A50285BI"


class WqSerial:
    def __init__(self, port, baud=HANDSHAKE_BAUD, timeout=3.0):
        self.ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout)
        self.ser.reset_input_buffer()
        self.seq = 0

    def baudrate(self, b, timeout=None):
        if timeout:
            self.ser.timeout = timeout
        self.ser.baudrate = b

    # ---- 帧层 ----
    def _frame(self, cmd, payload=b""):
        self.seq = (self.seq + 1) & 0xFFFF
        head = struct.pack("<HHH", self.seq, cmd, len(payload))
        crc = zlib.crc32(payload) & 0xFFFF if payload else 0
        return START + head + payload + struct.pack("<H", crc) + END

    def send(self, cmd, payload=b""):
        self.ser.write(self._frame(cmd, payload))
        self.ser.flush()

    def recv_frame(self, want_cmd=None, timeout=None):
        """读一个完整响应帧; 返回 (cmd, result, payload) 或 None"""
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
                buf = buf[-1:]  # 留最后 1 字节防半截 5C
                continue
            buf = buf[i:]
            if len(buf) < 2 + 8:
                continue
            cmd, result, plen = struct.unpack("<HHH", buf[2:8])
            total = 2 + 8 + plen + 2 + 2
            if len(buf) < total:
                continue
            return cmd, result, buf[10:10 + plen]
        return None

    def cmd(self, cmd_id, payload=b"", timeout=None, retries=1):
        for _ in range(retries):
            self.send(cmd_id, payload)
            r = self.recv_frame(cmd_id, timeout)
            if r is None:
                continue
            c, result, pl = r
            if c != cmd_id:
                continue
            return result, pl
        return None, None


def image_header(code: bytes, start=0, stack=0, version=0) -> bytes:
    """wq_image_header_t(32B packed): guard,length,version,start,crc,tlv(1)+pad(3),stack,reserved(4)"""
    return struct.pack("<5IB3xI4x", WQ_IMAGE_MAGIC, len(code), version, start,
                       zlib.crc32(code) & 0xFFFFFFFF, 0, stack)


def open_and_connect(port, verbose=True):
    s = WqSerial(port)
    if verbose:
        print(f"[1] {port} @115200 已打开, 发 CONNECT 探测(重试 5 秒)...")
    for _ in range(10):
        r, _ = s.cmd(CMD_CONNECT, timeout=0.5)
        if r == 0:
            r2, ver = s.cmd(CMD_GET_VERSION)
            if r2 == 0 and len(ver) >= 4:
                mj, mi = struct.unpack("<HH", ver[:4])
                if verbose:
                    print(f"    芯片应答! updater 版本 {mj}.{mi}")
            r3, soc = s.cmd(CMD_GET_SOC_ID)
            if r3 == 0 and soc:
                if verbose:
                    print(f"    SOC_ID: {soc.hex()}")
            return s
        time.sleep(0.5)
    return None


# ---------------- 子命令 ----------------

def do_probe(args):
    s = open_and_connect(args.port)
    if s is None:
        print("无应答。两种可能:\n"
              "  a) 芯片没进下载模式 —— 断电, 按住烧录条件(GPIO_101 上电为高/板上下载键), 再上电, 10 秒窗口内重跑\n"
              "  b) 接的是日志口(GPIO_41/44) —— 跑 `sniff -b 2000000`, 有滚动的日志即证明接的是日志口")
        return 1
    print("握手成功, 芯片在下载模式, 可以烧录。")
    s.ser.close()
    return 0


def do_sniff(args):
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    print(f"嗅探 {args.port} @ {args.baud}, 3 秒...")
    data = b""
    t0 = time.time()
    while time.time() - t0 < 3:
        data += ser.read(4096)
    ser.close()
    if not data:
        print("3 秒无数据。")
        return 1
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in data)
    if printable / len(data) > 0.7:
        print(f"收到 {len(data)}B, 大部分可打印 —— 这是日志口。内容片段:")
        print(data[:400].decode("utf-8", "replace"))
        print("(日志口跑 2M; 若要烧录请把线换到 GPIO_01/02 烧录口)")
    else:
        print(f"收到 {len(data)}B, 二进制协议数据 —— 可能是烧录口。HEX 前 64B:")
        print(data[:64].hex())
    return 0


def do_burn(args):
    z = zipfile.ZipFile(args.wpk)
    cfg = json.loads(z.read("memory_config.json"))
    print(f"[0] {args.wpk}: {cfg['chip']}, {len(cfg['images'])} 个分区")

    s = open_and_connect(args.port)
    if s is None:
        print("握手失败: 芯片未进下载模式(参考 probe 的提示)")
        return 1

    # 提速
    if args.baud != HANDSHAKE_BAUD and not args.keep_baud:
        print(f"[2] 提速到 {args.baud} ...")
        s.cmd(CMD_CHANGE_BAUDRATE, struct.pack("<II", args.baud, 50), timeout=2)
        time.sleep(0.1)
        s.baudrate(args.baud, timeout=10)
        r, _ = s.cmd(CMD_CONNECT, timeout=1.5, retries=3)
        if r != 0:
            print(f"    提速后失联(result={r}), 退回 115200 重试")
            s.baudrate(HANDSHAKE_BAUD, timeout=10)
            r, _ = s.cmd(CMD_CONNECT, timeout=1)
            assert r == 0, "连接丢失"
            args.baud = HANDSHAKE_BAUD
        else:
            print("    提速成功")

    r, _ = s.cmd(CMD_FACTORY_MODE)
    print(f"[3] FACTORY_MODE(取消 10 秒超时): result={r}")

    # BOOTMAP_SET: 全部分区(bootmap 自身除外)
    blocks = b""
    nmap = 0
    for im in cfg["images"]:
        if im["id"] == 8:  # bootmap 自身
            continue
        blocks += struct.pack("<B3xIII", im["id"], im["lma"], im.get("vma", 0), im["length"])
        nmap += 1
    r, _ = s.cmd(CMD_BOOTMAP_SET, struct.pack("<I", 0) + blocks, timeout=30)
    print(f"[4] BOOTMAP_SET({nmap} 项): result={r}")
    if r != 0:
        print("    分区表写入失败, 中止")
        return 1

    # 逐镜像烧录
    ok = 0
    for im in cfg["images"]:
        name = im.get("name", "")
        if im["id"] == 8 or name not in z.namelist():
            continue
        code = z.read(name)
        start = im.get("start", 0)
        stack = im.get("stack", 0)
        if not start:
            # DSP 镜像入口在 FACECAFE 容器里, 头里 start 填 vma 即可
            start = im.get("vma", 0)
        data = image_header(code, start=start, stack=stack) + code
        total = (len(data) + CHUNK - 1) // CHUNK
        print(f"[5] {name}: {len(code):,}B -> id={im['id']} lma={im['lma']:#x} ({total} 块)")
        for seq in range(total):
            chunk = data[seq * CHUNK:(seq + 1) * CHUNK]
            pl = struct.pack("<BBH", im["id"], 0, seq) + chunk
            tmo = 90 if seq == 0 else 10  # 首块触发整区擦除, 给足时间
            r, ack = s.cmd(CMD_IMAGE_WRITE, pl, timeout=tmo, retries=3)
            if r != 0:
                print(f"    块 {seq} 失败 result={r}, 中止")
                return 1
            if seq % 20 == 0 or seq == total - 1:
                pct = (seq + 1) * 100 // total
                print(f"\r    {pct:3d}%  (块 {seq + 1}/{total})", end="", flush=True)
        print()
        r, _ = s.cmd(CMD_IMAGE_VERIFY, struct.pack("<B3x", im["id"]), timeout=60)
        mark = "OK" if r == 0 else f"FAIL({r})"
        print(f"    VERIFY: {mark}")
        if r != 0:
            return 1
        ok += 1

    print(f"[6] 全部 {ok} 个镜像烧录+校验通过, 复位芯片...")
    s.cmd(CMD_CHIP_RESET, timeout=2)
    s.ser.close()
    print("完成。接日志口(GPIO_41/44 @2M)可看 [SHIL]; 或跑: python3 wq_burn.py shil")
    return 0


def do_shil(args):
    ser = serial.Serial(args.port, 2000000, timeout=1)
    print(f"抓 {args.port} @2000000 8N1, 等待 [SHIL](Ctrl-C 结束)...")
    buf = b""
    out = open(args.out, "wb")
    try:
        while True:
            d = ser.read(4096)
            if d:
                out.write(d)
                out.flush()
                sys.stdout.write(d.decode("utf-8", "replace"))
                sys.stdout.flush()
                buf = (buf + d)[-64:]
                if b"[SHIL] end" in buf:
                    print("\n[SHIL] end 已出现, 自动停止。")
                    break
    except KeyboardInterrupt:
        print(f"\n手动停止, 已存 {args.out}")
    finally:
        ser.close()
        out.close()
    print(f"日志已存 {args.out}; 验收: python3 hil_parse.py {args.out} golden_prob.f32")
    return 0


def do_loopback(args):
    """适配器自测: 把适配器上的 TX 针脚和 RX 针脚用杜邦线直接短接后运行"""
    ser = serial.Serial(args.port, 115200, timeout=0.1)
    ok = 0
    rounds = 20
    for i in range(rounds):
        sent = b"\x55" * 32 + bytes([i])
        ser.reset_input_buffer()
        ser.write(sent)
        ser.flush()
        time.sleep(0.12)
        got = ser.read(128)
        if got.startswith(sent[:16]):
            ok += 1
    ser.close()
    print(f"回环 {ok}/{rounds} 组收到自发数据")
    if ok >= 18:
        print("适配器 TX/RX 双向全部正常 —— 故障在板子侧(供电或接线)")
    elif ok > 0:
        print("部分回环成功 —— 短接接触不良, 重新插好再试")
    else:
        print("零回环 —— 适配器 RX 方向故障或未短接, 更换适配器/检查短接线")
    return 0


def main():
    ap = argparse.ArgumentParser(description="WQ7036 Mac 原生烧录器(协议源: SDK updater + 官方工具 DLL 对拍)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="只读握手")
    p.add_argument("-p", "--port", default=DEFAULT_PORT)
    p.set_defaults(fn=do_probe)
    p = sub.add_parser("sniff", help="嗅探串口判断接线")
    p.add_argument("-p", "--port", default=DEFAULT_PORT)
    p.add_argument("-b", "--baud", type=int, default=2000000)
    p.set_defaults(fn=do_sniff)
    p = sub.add_parser("loopback", help="适配器自测(需把适配器 TX/RX 两针短接)")
    p.add_argument("-p", "--port", default=DEFAULT_PORT)
    p.set_defaults(fn=do_loopback)
    p = sub.add_parser("burn", help="烧录 wpk")
    p.add_argument("wpk")
    p.add_argument("-p", "--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=921600, help="提速波特率(默认 921600, 0=保持 115200)")
    p.add_argument("--keep-baud", action="store_true")
    p.set_defaults(fn=do_burn)
    p = sub.add_parser("shil", help="2M 抓 [SHIL] 日志")
    p.add_argument("-p", "--port", default=DEFAULT_PORT)
    p.add_argument("-o", "--out", default="shil.log")
    p.set_defaults(fn=do_shil)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
