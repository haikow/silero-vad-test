/*
 * wq_silero_vad.c —— 物奇 libwq_sw_vad 五函数接口的 Silero VAD 实现
 * 详见 wq_silero_vad.h 头注释。
 */
#include <string.h>
#include "wq_silero_vad.h"
#include "silero_vad.h"

#define WQ_SILERO_ACC 512

int wq_get_sw_vad_hd_size(void)
{
    return (int)sizeof(wq_silero_hd_t);
}

int wq_get_sw_vad_scratch_size(void)
{
    return 0;   /* 大缓冲用静态区(单实例), 不占 pipeline scratch */
}

void *wq_sw_vad_init(void *vad_hd, void *pscratch)
{
    (void)pscratch;
    wq_silero_hd_t *hd = (wq_silero_hd_t *)vad_hd;
    memset(hd, 0, sizeof(*hd));
    return hd;
}

int wq_sw_vad_process(void *vad_hd, char *in_data, unsigned int in_len)
{
    wq_silero_hd_t *hd = (wq_silero_hd_t *)vad_hd;
    const short *pcm = (const short *)in_data;
    unsigned int n = in_len / 2;    /* 单通道 int16 样本数(上层 len 为通道字节数) */
    int hit = 0;

    while (n > 0) {
        unsigned int take = WQ_SILERO_ACC - hd->acc_n;
        if (take > n) take = n;
        for (unsigned int i = 0; i < take; i++)
            hd->acc[hd->acc_n + i] = (float)pcm[i] / 32768.0f;
        hd->acc_n += (int)take;
        pcm += take;
        n -= take;

        if (hd->acc_n == WQ_SILERO_ACC) {
            hd->last_prob = sv_process(&hd->sv, hd->acc);
            hd->windows++;
            if (hd->last_prob >= WQ_SILERO_THRESHOLD) hit = 1;
            hd->acc_n = 0;      /* 无重叠: 下一窗从零攒 */
        }
    }
    return hit;
}

void wq_get_sw_vad_lib_version(char *version_string)
{
    strcpy(version_string, "silero-vad v4 float C 1.0 (wq interface)");
}

float wq_silero_last_prob(void)
{
    return sv_last_prob();
}
