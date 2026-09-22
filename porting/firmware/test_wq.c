/* WQ 接口测试: golden 音频按 320 样本/帧(20ms)喂入
 * 1) 管道纯度: 同样的 int16 量化数据直接喂 sv_process, 比对攒帧管道 —— 应精确一致
 * 2) int16 量化效应: vs ORT float 基准 —— 容差 5e-2(HIL 验收标准是 1e-1) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wq_silero_vad.h"
#include "silero_vad.h"

int main(void)
{
    FILE *fx = fopen("golden_x.f32", "rb");
    FILE *fp = fopen("golden_prob.f32", "rb");
    if (!fx || !fp) { perror("golden"); return 1; }
    fseek(fx, 0, SEEK_END); long nx = ftell(fx) / 4; rewind(fx);
    int nw = nx / SV_FRAME;
    float *X = malloc(nx * 4), *P = malloc(nw * 4);
    if (fread(X, 4, nx, fx) != (size_t)nx || fread(P, 4, nw, fp) != (size_t)nw) return 2;
    short *pcm = malloc(nx * 2);
    for (long i = 0; i < nx; i++) {
        float v = X[i];
        pcm[i] = (short)(v > 1 ? 32767 : v < -1 ? -32768 : (short)(v * 32767.0f));
    }

    void *hd = malloc(wq_get_sw_vad_hd_size());
    wq_sw_vad_init(hd, NULL);
    char ver[64]; wq_get_sw_vad_lib_version(ver);
    printf("lib: %s, hd_size=%d scratch=%d\n", ver,
           wq_get_sw_vad_hd_size(), wq_get_sw_vad_scratch_size());

    wq_silero_hd_t *H = (wq_silero_hd_t *)hd;
    sv_state_t ref; sv_reset(&ref);
    float racc[1024]; int rn = 0;
    float maxd_pipe = 0, maxd_ort = 0;
    for (long off = 0; off + 320 <= nx; off += 320) {
        wq_sw_vad_process(hd, (char *)(pcm + off), 640);
        for (int i = 0; i < 320; i++) racc[rn++] = pcm[off + i] / 32768.0f;
        while (rn >= 512) {
            float p_ref = sv_process(&ref, racc);
            rn -= 512;
            if (rn > 0) memmove(racc, racc + 512, rn * sizeof(float));
            float d_pipe = fabsf(H->last_prob - p_ref);
            if (d_pipe > maxd_pipe) maxd_pipe = d_pipe;
            if (H->windows <= (unsigned)nw) {
                float d_ort = fabsf(H->last_prob - P[H->windows - 1]);
                if (d_ort > maxd_ort) maxd_ort = d_ort;
            }
        }
    }
    printf("累计喂 %ld 样本 -> 完整窗 %d\n", (nx / 320) * 320, (int)H->windows);
    printf("管道纯度(同量化数据 vs 直接 sv_process): 最大偏差 %.3e\n", maxd_pipe);
    printf("int16 量化效应(vs ORT float 基准):       最大偏差 %.3e\n", maxd_ort);
    int ok = maxd_pipe < 1e-5 && maxd_ort < 5e-2;
    printf(ok ? "PASS: 攒帧管道精确一致, int16 量化损失在预期内\n" : "FAIL\n");
    return ok ? 0 : 3;
}
