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
void  wq_sw_vad_exit(void);   /* v7 会话级功耗: 流销毁时释放 192M DFS(见 sw_vad_remove) */
void  wq_get_sw_vad_lib_version(char *version_string);

/* 句柄结构(供测试/调试读取; 固件侧按 void* 使用) */
typedef struct {
    sv_state_t sv;          /* LSTM 状态(窗间传递) */
    float acc[512];
    int acc_n;
    float last_prob;        /* 最近完成窗口的人声概率 */
    unsigned int windows;   /* 已完成窗口计数 */
    unsigned char rec_phase;    /* 录音态: 命中后置 1, 降载持续到本流实例销毁(不退出, 见 wq_silero_vad.c 注释) */
    unsigned char skip_cnt;     /* 录音态内的跳窗计数 */
    float noise_floor;          /* v7 自适应跳窗: RMS 噪声底 EMA(静音 1/6, 活跃 1/3) */
    int   pk_rms;               /* v8 瞬态否决: rms 峰值(rms×1e5) */
    unsigned char pk_age;       /* v8: 峰后窗龄(≤2 窗内骤降 20×=吹气/碰撞) */
    signed char veto_cnt;       /* v8: 瞬态否戒倒计数窗 */
} wq_silero_hd_t;

/* 调试/标定: 最近一窗的人声概率(0~1), 用于 HIL 逐窗比对 */
float wq_silero_last_prob(void);

#endif
