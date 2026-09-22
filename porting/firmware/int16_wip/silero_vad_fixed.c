/*
 * silero_vad_fixed.c —— Silero VAD v4 的 int16 定点实现(无 FPU 依赖)
 * 与 silero_vad.c(float)同构, 全整数运算:
 *   - 权重 int16 逐行缩放(gen_weights_fixed.py 生成)
 *   - 激活 int16 逐层 Q 格式(标定范围决定), conv/lstm 累加 int64
 *   - requant: out = ((acc>>16) * M) >> 15  (等价 acc*M>>31, 防 int64 溢出)
 *   - sqrt: 整数开方; log: ln 分解 + 512 项 LUT; sigmoid/tanh: 8192 项 LUT
 * 精度目标: 与 float 基准逐窗偏差 < 0.05(HIL 验收线 0.1)
 */
#include <string.h>
#if defined(SVF_DUMP) || defined(SVF_E6STREAM)
#include <stdio.h>
#endif
#ifdef SVF_DUMP
#include <stdio.h>
#endif
#include "silero_vad.h"
#include "silero_vad_fixed_consts.h"

extern const short svq_basis[258 * 256];
extern const short svq_filter7[7];
extern const int svq_ln_lut[513];
extern const short svq_sig_lut[8192];
extern const short svq_fl_proj_w[], svq_fl_dw_w[], svq_fl_pw_w[];
extern const int svq_fl_proj_b[], svq_fl_dw_b[], svq_fl_pw_b[], svq_fl_proj_M[], svq_fl_dw_M[], svq_fl_pw_M[];
extern const short svq_enc0_w[], svq_en3_proj_w[], svq_en3_dw_w[], svq_en3_pw_w[];
extern const int svq_enc0_b[], svq_en3_proj_b[], svq_en3_dw_b[], svq_en3_pw_b[];
extern const int svq_enc0_M[], svq_en3_proj_M[], svq_en3_dw_M[], svq_en3_pw_M[];
extern const short svq_enc4_w[], svq_en7_dw_w[], svq_en7_pw_w[];
extern const int svq_enc4_b[], svq_en7_dw_b[], svq_en7_pw_b[];
extern const int svq_enc4_M[], svq_en7_dw_M[], svq_en7_pw_M[];
extern const short svq_enc8_w[], svq_en11_proj_w[], svq_en11_dw_w[], svq_en11_pw_w[], svq_enc12_w[];
extern const int svq_enc8_b[], svq_en11_proj_b[], svq_en11_dw_b[], svq_en11_pw_b[], svq_enc12_b[];
extern const int svq_enc8_M[], svq_en11_proj_M[], svq_en11_dw_M[], svq_en11_pw_M[], svq_enc12_M[];
extern const short svq_l1_w[], svq_l1_r[], svq_l2_w[], svq_l2_r[];
extern const int svq_l1_b[], svq_l1_M[], svq_l2_b[], svq_l2_M[];
extern const short svq_dec_w[64];

static float g_last_prob_f;
#ifdef SVF_DUMP
extern int svf_dbg_win;
#define SVFD(arr, n) do { if (svf_dbg_win == 7) { FILE *df = fopen("f_" #arr ".f32", "wb");     short tmp[n]; for (int q = 0; q < n; q++) tmp[q] = arr[q];     fwrite(tmp, 2, n, df); fclose(df); } } while (0)
#endif

/* ---------- 整数开方(int64 -> int32) ---------- */
static unsigned int sqrt_64_seed(unsigned long long x);
static int isqrt64(long long v)
{
    /* 位扫描法(精确): 每次两位, 标准 integer sqrt, 无收敛问题 */
    if (v <= 0) return 0;
    unsigned long long x = (unsigned long long)v;
    unsigned long long res = 0;
    int bit = 62;
    while (bit >= 0) {
        unsigned long long t = ((res << 1) | 1) << bit;
        if (x >= t) { x -= t; res |= 1ULL << (bit >> 1); }
        bit -= 2;
    }
    return (int)res;   /* floor(sqrt(v)), 精确 */
}

/* ---------- sigmoid LUT: 输入 Q8(int), 输出 Q15 ---------- */
static int sig_q8(int x_q8)
{
    int i = x_q8 + 4096;
    if (i < 0) i = 0;
    if (i > 8191) i = 8191;
    return svq_sig_lut[i];
}
/* tanh Q15: tanh(x) = 2*sigmoid(2x) - 1, 输入 Q8 */
static int tanh_q8(int x_q8)
{
    int s = sig_q8(x_q8 + x_q8);
    return 2 * s - 32767;
}

/* ---------- ln(v), v 为正 int64, 输出 Q11 ---------- */
extern const long long SVQ_INV_A;   /* 由权重文件定义 */
static int ln_q11(long long v)
{
    if (v <= 0) return 0;
    int e = 0;
    while ((v >> (e + 1)) != 0) e++;           /* e = floor(log2(v)) */
    long long frac = v - (1LL << e);           /* [0, 2^e) */
    long long m31 = e > 31 ? (frac >> (e - 31)) : (frac << (31 - e));  /* [0,2^31) */
    int idx = (int)(m31 >> 23);                /* 512 级 */
    int fr = (int)(m31 & 0x7FFFFF);            /* 低 23 位做线性插值 */
    if (idx > 511) { idx = 511; fr = 0; }
    int l0 = svq_ln_lut[idx], l1 = svq_ln_lut[idx + 1];
    return SVQ_LNC + e * SVQ_LN2_Q11 + l0 + (int)(((long long)(l1 - l0) * fr) >> 23);
}

/* ---------- requant: (acc * M) >> 31, 防 int64 溢出的两步版 ---------- */
static inline int rq(long long acc, int M)
{
    return (int)((((acc >> 16) * (long long)M) + (1LL << 14)) >> 15);  /* 四舍五入 */
}
static inline short sat16(int v)
{
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (short)v;
}

/* ---------- 1x1 conv(定点): in [IC][T] int16, 逐行 M/bias ---------- */
static void conv1x1_q(const short *w, const int *b, const int *M, const short *in,
                      int IC, int OC, int T, int stride, short *out, int relu)
{
    int tout = (T - 1) / stride + 1;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < tout; t++) {
            const short *ip = in + (size_t)t * stride * IC;  /* 列布局见下: 实际 [C][T] */
            long long acc = b[c];
            const short *wp = w + (size_t)c * IC;
            for (int i = 0; i < IC; i++)
                acc += (long long)wp[i] * in[(size_t)i * T + t * stride];
            (void)ip;
            int v = rq(acc, M[c]);
            if (relu && v < 0) v = 0;
            out[(size_t)c * tout + t] = sat16(v);
        }
}

/* ---------- depthwise k5 pad2(定点, 逐通道行缩放) ---------- */
static void dwconv5_q(const short *w, const int *b, const int *M, const short *in,
                      int C, int T, short *out, int relu)
{
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            long long acc = b[c];
            const short *wp = w + c * 5;
            for (int k = 0; k < 5; k++) {
                int ti = t + k - 2;
                if (ti >= 0 && ti < T)
                    acc += (long long)wp[k] * in[(size_t)c * T + ti];
            }
            int v = rq(acc, M[c]);
            if (relu && v < 0) v = 0;
            out[(size_t)c * T + t] = sat16(v);
        }
}

/* ---------- LSTM 单步(定点) ----------
 * W/R: [256][IN] 行缩放 int16; bias/M: [256]
 * h: Q14 int16[64], c: Q11 int(钳位); x: int16 (Q=Qin) */
static void lstm_step_q(const short *Wq, const short *Rq, const int *b, const int *M,
                        int hshift, const short *x, int IN, short *h, int *c)
{
    short gate[4][64];
#ifdef SVF_DUMP
    static int dump_done = 0;
#endif
    for (int g = 0; g < 4; g++)
        for (int j = 0; j < 64; j++) {
            int row = g * 64 + j;
            const short *wp = Wq + (size_t)row * IN;
            const short *rp = Rq + (size_t)row * 64;
            long long acc = b[row];
            for (int i = 0; i < IN; i++) acc += (long long)wp[i] * x[i];
            for (int k = 0; k < 64; k++) {
                /* 先乘后移, 避免 h 预截断(反馈精度) */
                long long rh = (long long)rp[k] * h[k];
                acc += hshift >= 0 ? (rh >> hshift) : (rh << -hshift);
            }
            gate[g][j] = sat16(rq(acc, M[row]));
        }
#ifdef SVF_DUMP
    if (svf_dbg_win == 7 && !dump_done) {
        FILE *df = fopen("f_gates.s16", "wb");
        fwrite(gate, 2, 4*64, df); fclose(df); dump_done = 1;
    }
#endif
    for (int j = 0; j < 64; j++) {
        int i_ = sig_q8(gate[0][j]);
        int o_ = sig_q8(gate[1][j]);
        int f_ = sig_q8(gate[2][j]);
        int tg = tanh_q8(gate[3][j]);
        int cv = ((c[j] * f_ + 16384) >> 15) + ((i_ * tg + (1 << 18)) >> 19);  /* Q15xQ15 -> Q11, 舍入 */
        if (cv > SVQ_C_CLAMP) cv = SVQ_C_CLAMP;
        if (cv < -SVQ_C_CLAMP) cv = -SVQ_C_CLAMP;
        c[j] = cv;
        /* tanh(c): c Q11 -> Q8 */
        int cq8 = cv >> 3;
        if (cq8 > 127) cq8 = 127; else if (cq8 < -128) cq8 = -128;
        int hv = (o_ * tanh_q8(cq8) + (1 << 15)) >> 16;  /* Q15xQ15 -> Q14, 舍入 */
        h[j] = sat16(hv);
    }
}

typedef struct {
    short h[2][64];   /* Q14 */
    int c[2][64];     /* Q11, 钳位 ±16 */
} svf_state_t;

static svf_state_t g_st;

float sv_process_fixed(const short *x_pcm, void *stv)
{
    svf_state_t *st = (svf_state_t *)stv;
    /* ---------- 1. STFT ---------- */
    static int re[258][8], im[258][8];
    static int mag_q[129][8];
    for (int t = 0; t < 8; t++) {
        const short *win = x_pcm + t * 64 - 96;   /* reflect 96 由下方索引处理 */
        for (int f = 0; f < 258; f++) {
            const short *wp = svq_basis + (size_t)f * 256;
            long long acc = 0;
            for (int k = 0; k < 256; k++) {
                int idx = t * 64 + k - 96;        /* 原始索引 [-96, 512+96) */
                int s;
                if (idx < 0) s = x_pcm[96 - idx];          /* 左 reflect */
                else if (idx >= 512) s = x_pcm[510 - (idx - 512)]; /* 右 reflect */
                else s = x_pcm[idx];
                acc += (long long)wp[k] * s;
            }
            if (f < 129) re[f][t] = (int)(acc >> SVQ_STFT_SHIFT);
            else im[f - 129][t] = (int)(acc >> SVQ_STFT_SHIFT);
        }
    }
    for (int f = 0; f < 129; f++)
        for (int t = 0; t < 8; t++) {
            long long r = re[f][t], i2 = im[f][t];
            mag_q[f][t] = isqrt64(r * r + i2 * i2);
        }

#ifdef SVF_DUMP
    if (svf_dbg_win == 7) { FILE *df = fopen("f_re.i32", "wb"); fwrite(re, 4, 129*8, df); fclose(df);
        df = fopen("f_im.i32", "wb"); fwrite(im, 4, 129*8, df); fclose(df); }
#endif
    /* ---------- 2. lg + 归一化(Q11) ---------- */
    static int lg[129][8];
    for (int f = 0; f < 129; f++)
        for (int t = 0; t < 8; t++)
            lg[f][t] = (mag_q[f][t] == 0) ? 0 : ln_q11((long long)mag_q[f][t] + SVQ_INV_A);

    int m[8];
    for (int t = 0; t < 8; t++) {
        int s = 0;
        for (int f = 0; f < 129; f++) s += lg[f][t];
        m[t] = s / 129;
    }
    int mp[14] = { m[3], m[2], m[1], m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[6], m[5], m[4] };
    int base = 0;
    for (int k = 0; k < 8; k++) {
        long long acc = 0;
        for (int j = 0; j < 7; j++) acc += (long long)svq_filter7[j] * mp[k + j];
        base += rq(acc, SVQ_FILT_M);
    }
    base /= 8;

    /* ---------- feat: [mag | norm] 统一 Q9 int16 ---------- */
    static short feat[258][8];
    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++) {
            long long mm = (long long)mag_q[f][t] * SVQ_MAG2FEAT_M;
            feat[f][t] = sat16((int)(mm >> 31));
            int n = (lg[f][t] - base + 2) >> 2;  /* Q11 -> Q9, 舍入 */
            feat[129 + f][t] = sat16(n);
        }

#ifdef SVF_DUMP
    if (svf_dbg_win == 7) {
        FILE *df;
        df = fopen("f_feat.f32", "wb"); fwrite(&feat[0][0], 2, 258*8, df); fclose(df);
        df = fopen("f_magq.i32", "wb"); fwrite(&mag_q[0][0], 4, 129*8, df); fclose(df);
        df = fopen("f_lg.i32", "wb"); fwrite(&lg[0][0], 4, 129*8, df); fclose(df);
    }
#endif
    /* ---------- 3. first_layer 残差块 ---------- */
    static short proj[16][8], dwt[258][8], pw[16][8], act0[16][8];
    conv1x1_q(svq_fl_proj_w, svq_fl_proj_b, svq_fl_proj_M, &feat[0][0], 258, 16, 8, 1, &proj[0][0], 0);
    dwconv5_q(svq_fl_dw_w, svq_fl_dw_b, svq_fl_dw_M, &feat[0][0], 258, 8, &dwt[0][0], 1);
    conv1x1_q(svq_fl_pw_w, svq_fl_pw_b, svq_fl_pw_M, &dwt[0][0], 258, 16, 8, 1, &pw[0][0], 0);
    for (int c = 0; c < 16; c++)
        for (int t = 0; t < 8; t++) {
            int v = pw[c][t] + proj[c][t];
            act0[c][t] = sat16(v < 0 ? 0 : v);
        }

    /* ---------- 4. encoder: 8 -> 4 -> 2 -> 1 ---------- */
    static short e0[16][4], pj1[32][4], d1[16][4], w1[32][4], act1[32][4];
    static short e2[32][2], d2[32][2], w2[32][2], act2[32][2];
    static short e4[32][1], pj3[64][1], d3[32][1], w3[64][1], act3[64][1];
    static short e6[64][1];

    conv1x1_q(svq_enc0_w, svq_enc0_b, svq_enc0_M, &act0[0][0], 16, 16, 8, 2, &e0[0][0], 1);

    conv1x1_q(svq_en3_proj_w, svq_en3_proj_b, svq_en3_proj_M, &e0[0][0], 16, 32, 4, 1, &pj1[0][0], 0);
    dwconv5_q(svq_en3_dw_w, svq_en3_dw_b, svq_en3_dw_M, &e0[0][0], 16, 4, &d1[0][0], 1);
    conv1x1_q(svq_en3_pw_w, svq_en3_pw_b, svq_en3_pw_M, &d1[0][0], 16, 32, 4, 1, &w1[0][0], 0);
    for (int c = 0; c < 32; c++)
        for (int t = 0; t < 4; t++) {
            int v = w1[c][t] + pj1[c][t];
            act1[c][t] = sat16(v < 0 ? 0 : v);
        }

    conv1x1_q(svq_enc4_w, svq_enc4_b, svq_enc4_M, &act1[0][0], 32, 32, 4, 2, &e2[0][0], 1);

    dwconv5_q(svq_en7_dw_w, svq_en7_dw_b, svq_en7_dw_M, &e2[0][0], 32, 2, &d2[0][0], 1);
    conv1x1_q(svq_en7_pw_w, svq_en7_pw_b, svq_en7_pw_M, &d2[0][0], 32, 32, 2, 1, &w2[0][0], 0);
    for (int c = 0; c < 32; c++)
        for (int t = 0; t < 2; t++) {
            int v = w2[c][t] + ((e2[c][t] + 1) >> 1);   /* e2 Q10 -> Q9, 舍入 */
            act2[c][t] = sat16(v < 0 ? 0 : v);
        }

    conv1x1_q(svq_enc8_w, svq_enc8_b, svq_enc8_M, &act2[0][0], 32, 32, 2, 2, &e4[0][0], 1);

    conv1x1_q(svq_en11_proj_w, svq_en11_proj_b, svq_en11_proj_M, &e4[0][0], 32, 64, 1, 1, &pj3[0][0], 0);
    dwconv5_q(svq_en11_dw_w, svq_en11_dw_b, svq_en11_dw_M, &e4[0][0], 32, 1, &d3[0][0], 1);
    conv1x1_q(svq_en11_pw_w, svq_en11_pw_b, svq_en11_pw_M, &d3[0][0], 32, 64, 1, 1, &w3[0][0], 0);
    for (int c = 0; c < 64; c++) {
        int v = w3[c][0] + pj3[c][0];
        act3[c][0] = sat16(v < 0 ? 0 : v);
    }

    conv1x1_q(svq_enc12_w, svq_enc12_b, svq_enc12_M, &act3[0][0], 64, 64, 1, 1, &e6[0][0], 1);

#ifdef SVF_DUMP
    if (svf_dbg_win == 7) {
        FILE *df;
        df = fopen("f_act0.s16", "wb"); fwrite(&act0[0][0], 2, 16*8, df); fclose(df);
        df = fopen("f_e0.s16", "wb"); fwrite(&e0[0][0], 2, 16*4, df); fclose(df);
        df = fopen("f_act1.s16", "wb"); fwrite(&act1[0][0], 2, 32*4, df); fclose(df);
        df = fopen("f_e2.s16", "wb"); fwrite(&e2[0][0], 2, 32*2, df); fclose(df);
        df = fopen("f_act2.s16", "wb"); fwrite(&act2[0][0], 2, 32*2, df); fclose(df);
        df = fopen("f_e4.s16", "wb"); fwrite(&e4[0][0], 2, 32*1, df); fclose(df);
        df = fopen("f_act3.s16", "wb"); fwrite(&act3[0][0], 2, 64*1, df); fclose(df);
        df = fopen("f_e6.s16", "wb"); fwrite(&e6[0][0], 2, 64*1, df); fclose(df);
    }
#endif
#ifdef SVF_E6STREAM
    { static FILE *ef; if (!ef) ef = fopen("e6stream.s16", "wb"); fwrite(&e6[0][0], 2, 64, ef); }
#endif
    /* ---------- 5. LSTM x2 ---------- */
    short y1[64], y2[64];
    lstm_step_q(svq_l1_w, svq_l1_r, svq_l1_b, svq_l1_M, SVQ_L1_HSHIFT,
                e6[0], 64, st->h[0], st->c[0]);
    memcpy(y1, st->h[0], sizeof(y1));
    lstm_step_q(svq_l2_w, svq_l2_r, svq_l2_b, svq_l2_M, SVQ_L2_HSHIFT,
                y1, 64, st->h[1], st->c[1]);
    memcpy(y2, st->h[1], sizeof(y2));

#ifdef SVF_DUMP
    if (svf_dbg_win == 7) {
        FILE *df = fopen("f_h.s16", "wb"); fwrite(st->h, 2, 128, df); fclose(df);
        df = fopen("f_c.i32", "wb"); fwrite(st->c, 4, 128, df); fclose(df);
    }
#endif
    /* ---------- 6. decoder ---------- */
    long long acc = SVQ_DEC_B;
    for (int i = 0; i < 64; i++) {
        short r = y2[i] < 0 ? 0 : y2[i];
        acc += (long long)svq_dec_w[i] * r;
    }
    int pre_q8 = rq(acc, SVQ_DEC_M);
    int prob_q15 = sig_q8(pre_q8);
    g_last_prob_f = (float)prob_q15 / 32768.0f;
    return g_last_prob_f;
}

float sv_process_fixed_entry(const short *x_pcm)
{
    return sv_process_fixed(x_pcm, &g_st);
}

void sv_reset_fixed_global(void)
{
    memset(&g_st, 0, sizeof(g_st));
    g_last_prob_f = 0.0f;
}
