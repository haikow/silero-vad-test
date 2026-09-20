#!/usr/bin/env bash
# 在 XNNC 容器里执行一条命令, 例如:
#   porting/scripts/xnnc.sh ls /opt/XNNC
#   porting/scripts/xnnc.sh python3 /opt/XNNC/Scripts/xnnc.py --help
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE="xnnc:3.2.2"
exec docker run --rm --platform linux/amd64 \
  -v "$REPO_DIR":/work \
  -w /work/porting \
  "$IMAGE" "$@"
