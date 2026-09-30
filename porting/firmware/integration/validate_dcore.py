#!/usr/bin/env python3
"""WQ7036AC/AX dcore 镜像(FACECAFE 容器)段表链式校验器

用法: python3 validate_dcore.py <glass_dcore.bin> [更多.bin...]

规则(逆向自官方 v0.7.2.8/v0.9.2.8 镜像 + SDK xtensa_tools.py):
- 镜像 = 16B 头(magic/sect_num/totallen/entry) + 段表项与数据交错
- 每段 16B 表项: type/offset/vma/len, 数据紧随其后
- 加载器按 off+len 链式定位下一表项; cache 段(vma 0x26200000)的
  len **不含**对齐补零(公开版 SDK 的 zfill bug 会把补零计入 len,
  导致末段表项被错读、.dram.data 永不加载——2026-09-30 实证)

判据: 全部表项合法 且 链尾 == 文件长 → PASS(与官方镜像同构)
"""
import struct
import sys

IMAGECACHEADDR = 0x26200000


def validate(data: bytes, tag: str) -> bool:
    magic, n, totallen, entry = struct.unpack('<4I', data[:16])
    if magic != 0xFACECAFE:
        print(f'{tag}: 非 FACECAFE 容器(magic=0x{magic:08x}), 跳过')
        return True
    pos, offset, bad = 0x10, 0x20, []
    for i in range(n):
        if pos + 16 > len(data):
            bad.append(f'段{i} 表项越界@0x{pos:x}')
            break
        t, off, vma, ln = struct.unpack('<4I', data[pos:pos + 16])
        exp = offset
        if vma == IMAGECACHEADDR:
            exp = ((offset + 0x20 + 0xFF) // 0x100) * 0x100 - 0x20
        if off != exp or off < pos + 16 or off + ln > len(data):
            bad.append(f'段{i} 表项@0x{pos:x} 坏: off=0x{off:x}(应0x{exp:x}) vma=0x{vma:08x} len=0x{ln:x}')
        elif ln == 0:
            bad.append(f'段{i} len=0')
        pos = off + ln
        offset = exp + ln + 0x10
    chain_ok = pos == len(data)
    if not bad and chain_ok:
        print(f'{tag}: PASS ({n}段全部合法, 链尾==文件长)')
        return True
    for b in bad:
        print(f'{tag}: {b}')
    if not chain_ok:
        print(f'{tag}: 链尾 0x{pos:x} != 文件长 0x{len(data):x}')
    print(f'{tag}: **FAIL** —— 该镜像的末段(通常是 .clib.data+.dram.data)不会被加载, '
          f'dcore 将带随机 RAM 运行(症状: 秒死零日志 / 数秒后 heap_tlsf / WDT)')
    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [validate(open(p, 'rb').read(), p.split('/')[-1]) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
