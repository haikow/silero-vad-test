#!/usr/bin/env bash
# 从方案商原始 zip 一键还原出可编译的 demo 板工程树
# 用法: ./setup.sh <AI_Record_BT_base-*.zip 路径> [目标目录, 默认 ~/demo_work]
set -euo pipefail

ZIP="${1:?用法: $0 <厂商zip路径> [目标目录]}"
DEST="${2:-$HOME/demo_work}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -f "$ZIP" ] || { echo "找不到 zip: $ZIP"; exit 1; }

echo "==> 1/3 解压厂商 zip (GBK 文件名, 用 python zipfile 转码; macOS unzip 会中断)"
python3 - "$ZIP" "$DEST" <<'PYEOF'
import zipfile, os, sys
src, dst = sys.argv[1], sys.argv[2]
zf = zipfile.ZipFile(src)
n = 0
for info in zf.infolist():
    if info.is_dir():
        continue
    name = info.filename
    try:
        name = info.filename.encode('cp437').decode('gbk')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    target = os.path.join(dst, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'wb') as f:
        f.write(zf.read(info))
    n += 1
print(f"    解压 {n} 个文件 -> {dst}")
PYEOF

echo "==> 2/3 覆盖 Silero 移植 overlay (保持厂商 zip 内相对路径)"
cp -R "$HERE/overlay/." "$DEST/"

echo "==> 3/3 禁用原厂预编译 VAD 库 (改名保留, 可随时回退)"
LIB="$DEST/wq-audio/wq-adk/components/audio_algorithm/lib"
if [ -d "$LIB/wq_sw_vad" ]; then
  mv "$LIB/wq_sw_vad" "$LIB/lib_disabled_wq_sw_vad"
fi

cat <<'NEXT'

✅ 工程树就绪。编译入口:
   <目标目录>/wq-audio/wq-adk/examples/glass

下一步: 按 docs/编译指南.md 在 Docker 里编译 (约 5 条命令)。
固件配置用 config/7036AC/defconfig.record (已含 CONFIG_VAD_ENABLE=y)。
NEXT
