/*
 * test_ref.c —— Mac/Linux 主机验证: C 实现 vs onnxruntime 黄金基准
 * 读 golden_x.f32 (N*512 float) 与 golden_prob.f32 (N float), 逐窗比对
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "silero_vad.h"

int main(int argc, char **argv)
{
    FILE *fx = fopen(argc > 1 ? argv[1] : "golden_x.f32", "rb");
    FILE *fp = fopen(argc > 2 ? argv[2] : "golden_prob.f32", "rb");
    if (!fx || !fp) { perror("golden"); return 1; }

    fseek(fx, 0, SEEK_END); long nx = ftell(fx) / 4; rewind(fx);
    int n = nx / SV_FRAME;
    float *X = malloc(nx * 4), *P = malloc(n * 4);
    if (fread(X, 4, nx, fx) != (size_t)nx || fread(P, 4, n, fp) != (size_t)n) return 2;

    sv_state_t st; sv_reset(&st);
    float maxd = 0.0f; int maxi = -1;
    printf("%4s %12s %12s %12s\n", "win", "C", "ORT", "|diff|");
    for (int i = 0; i < n; i++) {
        float p = sv_process(&st, X + (size_t)i * SV_FRAME);
        float d = fabsf(p - P[i]);
        if (d > maxd) { maxd = d; maxi = i; }
        if (i < 5 || i == n - 1 || d > 1e-3)
            printf("%4d %12.6f %12.6f %12.2e\n", i, p, P[i], d);
    }
    printf("\n窗口数 %d, 最大偏差 %.3e (第 %d 窗)\n", n, maxd, maxi);
    printf(maxd < 1e-3 ? "PASS: C 实现与 ORT 基准一致 (<1e-3)\n"
                        : "FAIL: 偏差超限, 需逐级排查\n");
    return maxd < 1e-3 ? 0 : 3;
}
