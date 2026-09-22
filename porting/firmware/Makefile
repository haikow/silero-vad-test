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
