#ifndef SILERO_VAD_H
#define SILERO_VAD_H
/*
 * Silero VAD v4 (float) 的可移植 C99 实现
 * - 输入: 每窗 512 个 float32 样本 (16kHz, [-1,1])
 * - 状态: 2 层 LSTM 的 h/c 在窗间传递 (流式), reset 清零
 * - 精度目标: 与 onnxruntime float 模型逐窗一致 (<=1e-3)
 * 权重由 silero_vad_weights.c 提供 (flash/const 区)
 */
#include <stdint.h>

#define SV_FRAME 512      /* 32ms @16k */
#define SV_NUM_FREQ 129   /* STFT 频点 */
#define SV_NUM_FRAMES 8  /* 每窗时间帧数 */

typedef struct {
    float lstm_h[2][64];
    float lstm_c[2][64];
} sv_state_t;

/* 处理一窗, 返回人声概率 [0,1]; 状态在 st 内更新 */
float sv_process(sv_state_t *st, const float *x);

/* 状态清零(重新开始一段音频时调用) */
void sv_reset(sv_state_t *st);

/* 最近一窗的中间量(调试/标定用) */
float sv_last_prob(void);

#endif
