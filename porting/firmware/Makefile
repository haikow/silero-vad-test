# libwq_silero_vad —— Silero VAD 的物奇 libwq_sw_vad 替换库
# 主机验证: make test (需要 golden_x.f32 / golden_prob.f32, 由 gen_golden.py 生成)
# 固件集成: make lib => libwq_silero_vad.a, 替换 wq-adk 里的 libwq_sw_vad.a 链接
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -std=c99 -Wall

all: lib

lib: libwq_silero_vad.a

libwq_silero_vad.a: wq_silero_vad.o silero_vad.o silero_vad_weights.o
	$(AR) rcs $@ $^

test: test_ref test_wq
	./test_ref
	./test_wq

test_ref: test_ref.c silero_vad.c silero_vad_weights.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_wq: test_wq.c wq_silero_vad.c silero_vad.c silero_vad_weights.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f *.o *.a test_ref test_wq

.PHONY: all test clean

# ---- 交叉编译(容器内执行, wqcore 挂载在 /opt/wqcore) ----
# GCC 12.2 软浮点版: 可编译/功能验证用; 生产固件(glass/7036AC)实际用 xt-clang(需 license)
WQCORE ?= /opt/wqcore
XGCC = $(WQCORE)/toolchain/xtensa-wuqi-elf-gcc-x64-64-12.2.0/bin/xtensa-wuqi-elf-gcc
XAR  = $(WQCORE)/toolchain/xtensa-wuqi-elf-gcc-x64-64-12.2.0/bin/xtensa-wuqi-elf-ar

cross-gcc:
	mkdir -p build_gcc
	cd build_gcc && $(XGCC) -O2 -std=c99 -Wall -c ../silero_vad.c ../wq_silero_vad.c ../silero_vad_weights.c
	cd build_gcc && $(XAR) rcs libwq_silero_vad.a silero_vad.o wq_silero_vad.o silero_vad_weights.o
	@echo "=> build_gcc/libwq_silero_vad.a (elf32-xtensa, 软浮点, 功能验证用)"

.PHONY: cross-gcc
