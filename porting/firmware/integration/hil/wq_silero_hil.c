/*
 * wq_silero_hil.c —— HIL 自环测试: 上电后自动跑 golden 音频, 逐窗概率经调试串口输出
 * 用途: 开发板到位后 10 分钟完成 HIL 验收(与 PC 基准 golden_prob.f32 比对, 线 0.1)
 * 输出格式: "[SHIL] begin" / "[SHIL] win=NN p=0.xxxx" / "[SHIL] end"
 * 测试构建钩子: 生产版可移除 entry.c 里的调用(或整体删除本文件)
 */
#include "types.h"
#include "stdio.h"
#include "silero_hil_audio.h"
#include "wq_heap.h"
#include "wq_silero_vad.h"

#if is_defined(CONFIG_AUDIO_VAD_ENABLE)

void wq_silero_hil_selftest(void)
{
    void *hd = wq_heap_caps_malloc((uint32_t)wq_get_sw_vad_hd_size(), 0);
    if (!hd) {
        printf("[SHIL] fail: no mem\n");
        return;
    }
    wq_sw_vad_init(hd, NULL);
    printf("[SHIL] begin windows=%d\n", SILERO_HIL_WINDOWS);

    /* 按 processor 真实节奏喂: 20ms(320样本,640B)一帧 —— 同时锻炼攒帧路径 */
    const short *p = silero_hil_audio;
    int total = SILERO_HIL_WINDOWS * 512;
    int win = 0;
    for (int off = 0; off + 320 <= total; off += 320) {
        wq_sw_vad_process(hd, (char *)(p + off), 640);
        float prob = wq_silero_last_prob();
        /* 每 1.6 帧出一个窗概率; 用窗口计数对齐(与 PC 端 test_wq 相同机制) */
        int wins_now = (int)(((long)off + 320) * 1000L / 512L);  /* 采样进度 */
        int wins_full = ((off + 320) / 512);
        (void)wins_now;
        if (((off + 320) % 512) == 0) {
            printf("[SHIL] win=%d p=%d.%04d\n", win,
                   (int)(prob * 10000) / 10000, (int)(prob * 10000) % 10000);
            win++;
        }
        (void)wins_full;
    }
    printf("[SHIL] end wins=%d\n", win);
    wq_heap_caps_free(hd);
}

#endif /* CONFIG_AUDIO_VAD_ENABLE */
