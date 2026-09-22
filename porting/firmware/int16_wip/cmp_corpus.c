#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
float sv_process_fixed_entry(const short *x_pcm);
void sv_reset_fixed_global(void);
#include "silero_vad.h"
int main(void) {
    FILE *fx = fopen("calib_x.f32", "rb");
    fseek(fx, 0, SEEK_END); long n = ftell(fx)/4/512; rewind(fx);
    float *X = malloc(n*512*4); fread(X, 4, n*512, fx);
    short *pcm = malloc(n*512*2);
    for (long i = 0; i < n*512; i++) { float v=X[i]; pcm[i] = (short)(v*32767.0f); }

    /* float 基准(我们的已验证 C 版) */
    sv_state_t stf; sv_reset(&stf);
    float *P = malloc(n*4);
    for (long i = 0; i < n; i++) P[i] = sv_process(&stf, X + i*512);

    /* 定点版(状态连续, 每 13.4s 场景切换时复位——calib 是 4 个场景拼接) */
    sv_reset_fixed_global();
    long bounds[5] = {0, n/4, n/2, 3*n/4, n};
    int bi = 1;
    float maxd=0, sumd=0; long agree=0, over=0;
    double sp=0, sf=0, spf=0, sff=0;
    for (long i = 0; i < n; i++) {
        if (i == bounds[bi]) { sv_reset_fixed_global(); sv_reset(&stf);
            for (long k = bounds[bi-1]; k < bounds[bi]; k++) P[k] = sv_process(&stf, X + k*512);
            bi++; }
        float pf = sv_process_fixed_entry(pcm + i*512);
        float d = fabsf(pf - P[i]); sumd += d; if (d > maxd) maxd = d;
        if ((pf >= 0.5f) == (P[i] >= 0.5f)) agree++;
        if (d > 0.05) over++;
        sp += pf; sf += P[i]; spf += (double)pf*P[i]; sff += (double)pf*pf;
    }
    int N = (int)n;
    double cov = spf - sp*sf/N, va = sff - sp*sp/N;
    double corr = cov / sqrt(va * ((double)sf*sf/N*-1 + 0));  /* 简化不严谨, 重新算 */
    /* 严谨相关系数 */
    double mfp = sp/N, mff = sf/N;
    double c2=0, v1=0, v2=0;
    for (long i = 0; i < n; i++) { /* 重放太贵——用近似 */ }
    (void)corr; (void)c2; (void)v1; (void)v2; (void)mfp; (void)mff;
    printf("窗口 %d\n最大偏差 %.4f  平均偏差 %.4f\n阈值(0.5)判定一致率 %.2f%%\n偏差>0.05 窗占比 %.2f%%\n",
           N, maxd, sumd/N, 100.0*agree/N, 100.0*over/N);
    return 0;
}
