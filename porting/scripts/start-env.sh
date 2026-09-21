#!/usr/bin/env bash
# 一键搭建/启动 XNNC 编译环境 (Apple Silicon Mac)
# 1. 启动 Colima VM (VZ 虚拟化 + Rosetta 跑 x86-64 容器)
# 2. 构建 xnnc:3.2.2 镜像 (首次约 20-40 分钟, 下载依赖为主)
# 3. 进入交互容器, 本仓库根挂载在 /work
#
# 非 Mac 的 x86 Linux 机器: 无需 VM, 直接
#   docker build --platform linux/amd64 -f porting/docker/Dockerfile -t xnnc:3.2.2 porting/
#
# !! 重要: Mac 上启动 VM 必须走本脚本 (limactl), 不要用 `colima start`/`colima restart`:
#    colima 0.10.3 对已存在实例重新渲染 lima.yaml 时会丢掉 rosetta 配置块,
#    导致 amd64 退回 QEMU 模拟(慢一个数量级)。colima stop 是安全的。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORTING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PORTING_DIR/.." && pwd)"
IMAGE="xnnc:3.2.2"
LIMA_HOME_DIR="$HOME/.colima/_lima"
LIMAYAML="$LIMA_HOME_DIR/colima/lima.yaml"

# ---- 0. 前置检查: XNNC SDK 必须已解压到 porting/XNNC (Cadence 授权件, 不入库) ----
if [ ! -f "$PORTING_DIR/XNNC/version.txt" ]; then
  echo "错误: 缺少 $PORTING_DIR/XNNC/" >&2
  echo "  从团队渠道拿到 XNNC_3.2.2_Linux.tar.gz 后:" >&2
  echo "  cd $PORTING_DIR && tar -xzf /path/to/XNNC_3.2.2_Linux.tar.gz" >&2
  exit 1
fi

# ---- 1. VM (仅 Mac 需要) ----
if [[ "$(uname -s)" == "Darwin" ]]; then
  command -v colima >/dev/null || { echo "先安装: brew install colima docker docker-buildx" >&2; exit 1; }
  if ! colima status 2>/dev/null | grep -q running; then
    # 确保 rosetta 配置块在(colima start 会把它冲掉)
    if ! grep -q "^rosetta:" "$LIMAYAML" 2>/dev/null; then
      echo "==> 补写 rosetta 配置到 lima.yaml"
      mkdir -p "$LIMA_HOME_DIR/colima"
      touch "$LIMAYAML"
      cat >> "$LIMAYAML" <<'EOF'

# 手动补充: Rosetta x86-64 转译(colima 0.10.3 对已存在实例不渲染此块, 由 limactl 启动时生效)
rosetta:
  enabled: true
  binfmt: true
EOF
    fi
    echo "==> 启动 VM (limactl, VZ + Rosetta)"
    LIMA_HOME="$LIMA_HOME_DIR" limactl start colima
  fi

  # 确认 Rosetta 真的挂上了(没挂上给出提示而不是静默降级)
  if ! printf 'ls /mnt/lima-rosetta/rosetta\n' | colima ssh 2>/dev/null | grep -q rosetta; then
    echo "警告: VM 里没有 /mnt/lima-rosetta —— amd64 容器将走 QEMU(慢, 但功能一致)。" >&2
    echo "       可执行: colima stop && $0 重试" >&2
  fi

  # Docker Hub 镜像加速(国内网络; 已配置则跳过)
  if ! printf 'cat /etc/docker/daemon.json\n' | colima ssh 2>/dev/null | grep -q registry-mirrors; then
    echo "==> 配置 Docker Hub 镜像加速"
    printf 'sudo sh -c "mkdir -p /etc/docker && printf %s > /etc/docker/daemon.json && systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || rc-service docker restart"\n' \
      '{"registry-mirrors":["https://docker.1ms.run","https://docker.m.daocloud.io","https://dockerproxy.net"]}' \
      | colima ssh >/dev/null
    sleep 5
  fi

  # docker context 在 colima stop 时可能被删, 重建; limactl 启动后 docker 服务偶发未拉起, 兜底
  docker context use colima >/dev/null 2>&1 || \
    docker context create colima --docker "host=unix://$HOME/.colima/docker.sock" >/dev/null
  for i in 1 2 3 4 5; do
    docker info >/dev/null 2>&1 && break
    [ "$i" = 1 ] && printf 'sudo systemctl enable --now docker\n' | colima ssh >/dev/null 2>&1
    sleep 3
  done
fi
docker info --format 'Docker 就绪: {{.ServerVersion}}' >/dev/null

# ---- 2. 镜像 ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> 构建 $IMAGE (x86-64)"
  docker build --platform linux/amd64 -f "$PORTING_DIR/docker/Dockerfile" -t "$IMAGE" "$PORTING_DIR"
fi

# ---- 3. 进入容器 ----
# WQCORE 物奇工具链(可选): 放在仓库根 wqcore/ (含 toolchain/RI-2020.4-linux/...) 自动挂载到 /opt/wqcore
EXTRA_MOUNTS=()
[ -d "$REPO_DIR/wqcore/toolchain" ] && EXTRA_MOUNTS+=(-v "$REPO_DIR/wqcore":/opt/wqcore)

echo "==> 进入 XNNC 容器 (仓库挂载于 /work, 移植材料在 /work/porting/silero${EXTRA_MOUNTS:+, WQCORE 工具链在 /opt/wqcore})"
exec docker run --rm -it --platform linux/amd64 \
  -v "$REPO_DIR":/work \
  "${EXTRA_MOUNTS[@]}" \
  -w /work/porting \
  "$IMAGE"
