/*
 * wq_silero_vad —— 物奇 libwq_sw_vad 接口的 Silero VAD 替换实现
 *
 * 对齐 wq-adk: components/audio_algorithm/lib/wq_sw_vad/inc/wq_sw_vad.h 的 5 个函数,
 * processor 层(sw_vad.c)零改动, 直接替换链接库即可。
 *
 * 帧长适配: 上层按 20ms(320样本@16k)喂 int16 PCM; Silero 需要 512 样本(32ms)窗,
 * 内部攒帧: 320*8 = 2560 = 512*5, 无重叠自然对齐。
 *
 * 阈值: 默认 0.5(Silero 官方), 编译时 -DWQ_SILERO_THRESHOLD=0.6 可调;
 * 建议配合 processor 层 continue_hit/stop_cnt 做迟滞(四件套)。
 */
#ifndef WQ_SILERO_VAD_H
#define WQ_SILERO_VAD_H
#include "silero_vad.h"

#ifndef WQ_SILERO_THRESHOLD
#define WQ_SILERO_THRESHOLD 0.5f
#endif

int   wq_get_sw_vad_hd_size(void);
int   wq_get_sw_vad_scratch_size(void);
void *wq_sw_vad_init(void *vad_hd, void *pscratch);
int   wq_sw_vad_process(void *vad_hd, char *in_data, unsigned int in_len);
void  wq_get_sw_vad_lib_version(char *version_string);

/* 句柄结构(供测试/调试读取; 固件侧按 void* 使用) */
typedef struct {
    sv_state_t sv;          /* LSTM 状态(窗间传递) */
    float acc[512];
    int acc_n;
    float last_prob;        /* 最近完成窗口的人声概率 */
    unsigned int windows;   /* 已完成窗口计数 */
} wq_silero_hd_t;

/* 调试/标定: 最近一窗的人声概率(0~1), 用于 HIL 逐窗比对 */
float wq_silero_last_prob(void);

#endif
