# XNNC 3.2.2 (Cadence NeuroWeave SDK) 构建环境
# 目标: 在 Apple Silicon Mac 上通过 Colima(VZ+Rosetta) 运行 x86-64 Linux 容器
# 基础选 Ubuntu 22.04: 系统 Python 恰为 3.10(SDK 固定 3.10.7 系), 仓库含 gcc-9
# 构建命令见 scripts/build-image.sh (务必带 --platform linux/amd64)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ---------- APT 换清华源(先走 http: 基础镜像无 ca-certificates, 装完证书后其余层可用 https) ----------
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g; \
            s|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# ---------- 系统依赖(SDK 文档 §2.2 软件前置) ----------
# gcc-9(文档要求 GCC 9.3, jammy 为 9.5, 多 libc 为 CSTUB 32 位仿真所需)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git \
        bc lua5.1 \
        make build-essential gcc-9 g++-9 gcc-9-multilib g++-9-multilib \
        ninja-build unzip xz-utils file \
        python3 python3-venv python3-pip python3-dev \
        libgomp1 zlib1g libxml2 \
    && rm -rf /var/lib/apt/lists/*

# 默认 gcc/g++ 切到 9 (CSTUB 编译系统编译器)
RUN update-alternatives --install /usr/bin/gcc  gcc  /usr/bin/gcc-9  100 \
    && update-alternatives --install /usr/bin/g++  g++  /usr/bin/g++-9  100 \
    && update-alternatives --install /usr/bin/cc   cc   /usr/bin/gcc-9  100 \
    && update-alternatives --install /usr/bin/c++  c++  /usr/bin/g++-9  100

# ---------- LLVM 16 (文档要求 clang/clang++ + libc++/libc++abi/libunwind) ----------
RUN echo "deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/jammy/ llvm-toolchain-jammy-16 main" \
        > /etc/apt/sources.list.d/llvm16.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-16 lld-16 libc++-16-dev libc++abi-16-dev libunwind-16-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/clang-16   /usr/local/bin/clang \
    && ln -sf /usr/bin/clang++-16 /usr/local/bin/clang++ \
    && ln -sf /usr/bin/lld-16     /usr/local/bin/lld \
    && ldconfig

# ---------- CMake 3.27.7 (文档固定版本, 用 pip 精确安装) ----------
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "cmake==3.27.7"

# ---------- XNNC (Cadence NeuroWeave SDK 3.2.2) ----------
COPY XNNC /opt/XNNC
ENV XNNC_HOME=/opt/XNNC

# ---------- Python 3.10 venv + SDK 固定依赖 ----------
# SDK 文档 §2.3: XNNC 在 Python 3.10.7 上构建测试, 依赖版本固定, 勿随意升级
# requirements 同列 onnxruntime 与 onnxruntime-gpu(同导入名, 覆盖顺序不定), 无 GPU 环境
# 强制 CPU 版最后生效; 依赖会带入新版 cmake 顶掉固定版本, 一并钉回 3.27.7
RUN python3 -m venv /opt/xnnc_venv \
    && /opt/xnnc_venv/bin/pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip \
    && /opt/xnnc_venv/bin/pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /opt/XNNC/requirements.txt

# requirements 同列 onnxruntime 与 onnxruntime-gpu(同导入名, 覆盖顺序不定), 且依赖会带入
# 新版 cmake 顶掉固定版本 —— 单独一层钉回: CPU 版 onnxruntime + 文档要求的 cmake 3.27.7
RUN /opt/xnnc_venv/bin/pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "cmake==3.27.7" "onnxruntime==1.14.1" --force-reinstall --no-deps

ENV PATH=/opt/xnnc_venv/bin:$PATH

WORKDIR /work
CMD ["/bin/bash"]
