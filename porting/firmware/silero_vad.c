/*
 * Silero VAD v4 float 前向传播 —— 可移植 C99 参考实现
 * 结构(从 silero_vad_v4_float.onnx 逐节点转写, 时间维度经 ORT 数值校准):
 *   1. reflect pad(96,96) + Conv(258x1x256, stride64) => 8帧x258 (前129实部/后129虚部)
 *      mag = sqrt(re^2 + im^2)                          [数值验证误差 3e-6]
 *   2. 自适应归一化: lg = log(mag*2^20 + 1)
 *      m[t] = mean_f(lg[t]);  mp = [m[3], m[0..7], m[6]];  base = mean(conv7(mp))
 *      feat = concat(mag, lg - base)  —— 258 x 8
 *   3. first_layer 残差: proj(258->16) + [dw(k5,pad2,g258)+relu + pw(258->16)] 后相加再 relu
 *   4. encoder: s2(16->16) | 残差(16->32) | s2(32->32) | 残差(32->32,加块输入)
 *               | s2(32->32) | 残差(32->64) | conv(64->64); 时间 8->4->2->1
 *   5. 2 层 LSTM(input 64, hidden 64, PyTorch 门序 i,o,f,c), 每窗 1 帧
 *   6. relu -> 1x1(64->1) -> sigmoid => prob
 */
#include <math.h>
#include <string.h>
#include "silero_vad.h"
#include "silero_vad_consts.h"

extern const float sv_feature_extractor_forward_basis_buffer[]; /* [258*256] */
extern const float sv_adaptive_normalization_filter_[];         /* [7] */
extern const float sv_first_layer_0_proj_weight[], sv_first_layer_0_proj_bias[];
extern const float sv_first_layer_0_dw_conv_0_weight[], sv_first_layer_0_dw_conv_0_bias[];
extern const float sv_first_layer_0_pw_conv_0_weight[], sv_first_layer_0_pw_conv_0_bias[];
extern const float sv_Conv_349[], sv_Conv_350[];      /* enc0 16->16 s2 */
extern const float sv_encoder_3_0_proj_weight[], sv_encoder_3_0_proj_bias[];
extern const float sv_encoder_3_0_dw_conv_0_weight[], sv_encoder_3_0_dw_conv_0_bias[];
extern const float sv_encoder_3_0_pw_conv_0_weight[], sv_encoder_3_0_pw_conv_0_bias[];
extern const float sv_Conv_352[], sv_Conv_353[];      /* enc4 32->32 s2 */
extern const float sv_encoder_7_0_dw_conv_0_weight[], sv_encoder_7_0_dw_conv_0_bias[];
extern const float sv_encoder_7_0_pw_conv_0_weight[], sv_encoder_7_0_pw_conv_0_bias[];
extern const float sv_Conv_355[], sv_Conv_356[];      /* enc8 32->32 s2 */
extern const float sv_encoder_11_0_proj_weight[], sv_encoder_11_0_proj_bias[];
extern const float sv_encoder_11_0_dw_conv_0_weight[], sv_encoder_11_0_dw_conv_0_bias[];
extern const float sv_encoder_11_0_pw_conv_0_weight[], sv_encoder_11_0_pw_conv_0_bias[];
extern const float sv_Conv_358[], sv_Conv_359[];      /* enc12 64->64 s1 */
extern const float sv_LSTM_398[], sv_LSTM_399[], sv_LSTM_400[]; /* L1 W,R,B */
extern const float sv_LSTM_418[], sv_LSTM_419[], sv_LSTM_420[]; /* L2 W,R,B */
extern const float sv_decoder_decoder_1_weight[]; /* [64] */
extern const float sv_decoder_decoder_1_bias[];   /* [1] */

static float g_last_prob;



static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* 1x1 卷积(可带 stride), 输入输出 [C][T] 行主序 */
static void conv1x1(const float *w, const float *b, const float *in, int IC, int OC,
                    int T, int stride, float *out, int relu)
{
    int tout = (T - 1) / stride + 1;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < tout; t++) {
            float acc = b[c];
            for (int i = 0; i < IC; i++)
                acc += w[(size_t)c * IC + i] * in[(size_t)i * T + t * stride];
            out[(size_t)c * tout + t] = (relu && acc < 0.0f) ? 0.0f : acc;
        }
}

/* depthwise conv k5 pad2 stride1, relu 可选 */
static void dwconv5(const float *w, const float *b, const float *in, int C, int T,
                    float *out, int relu)
{
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            float acc = b[c];
            for (int k = 0; k < 5; k++) {
                int ti = t + k - 2;
                float v = (ti < 0 || ti >= T) ? 0.0f : in[(size_t)c * T + ti];
                acc += w[c * 5 + k] * v;
            }
            out[(size_t)c * T + t] = (relu && acc < 0.0f) ? 0.0f : acc;
        }
}

/* LSTM 单步: W[4*64][IN], R[4*64][64], B[512]=(Wb,Rb), 门序 i,o,f,c */
static void lstm_step(const float *W, const float *R, const float *B,
                      const float *x, float *h, float *c, int IN)
{
    float gi[64], go[64], gf[64], gc[64];
    for (int g = 0; g < 4; g++) {
        float *dst = (g == 0) ? gi : (g == 1) ? go : (g == 2) ? gf : gc;
        for (int j = 0; j < 64; j++) {
            float acc = B[g * 64 + j] + B[256 + g * 64 + j];
            const float *wp = W + (size_t)(g * 64 + j) * IN;
            for (int i = 0; i < IN; i++) acc += wp[i] * x[i];
            const float *rp = R + (size_t)(g * 64 + j) * 64;
            for (int hh = 0; hh < 64; hh++) acc += rp[hh] * h[hh];
            dst[j] = acc;
        }
    }
    for (int j = 0; j < 64; j++) {
        float i_ = sigmoidf_(gi[j]);
        float o_ = sigmoidf_(go[j]);
        float f_ = sigmoidf_(gf[j]);
        float c_ = f_ * c[j] + i_ * tanhf(gc[j]);
        c[j] = c_;
        h[j] = o_ * tanhf(c_);
    }
}

void sv_reset(sv_state_t *st)
{
    memset(st, 0, sizeof(*st));
    g_last_prob = 0.0f;
}

float sv_last_prob(void) { return g_last_prob; }

float sv_process(sv_state_t *st, const float *x)
{
    /* ---------- 1. 前端: reflect(96,96) + STFT(8 帧) ---------- */
    static float pad[704];
    static float stft[258][8];
    static float mag[129][8];
    static float lg[129][8];

    for (int i = 0; i < 96; i++) pad[i] = x[96 - i];
    memcpy(pad + 96, x, sizeof(float) * 512);
    for (int i = 0; i < 96; i++) pad[608 + i] = x[510 - i];

    for (int t = 0; t < 8; t++) {
        const float *win = pad + t * 64;
        for (int f = 0; f < 258; f++) {
            const float *wp = sv_feature_extractor_forward_basis_buffer + (size_t)f * 256;
            float acc = 0.0f;
            for (int k = 0; k < 256; k++) acc += wp[k] * win[k];
            stft[f][t] = acc;
        }
    }
    for (int f = 0; f < 129; f++)
        for (int t = 0; t < 8; t++) {
            float re = stft[f][t], im = stft[129 + f][t];
            mag[f][t] = sqrtf(re * re + im * im);
        }

    /* ---------- 2. 自适应归一化 ---------- */
    for (int f = 0; f < 129; f++)
        for (int t = 0; t < 8; t++)
            lg[f][t] = logf(mag[f][t] * SV_NORM_MUL + SV_NORM_ADD);

    float m[8];
    for (int t = 0; t < 8; t++) {
        float s = 0.0f;
        for (int f = 0; f < 129; f++) s += lg[f][t];
        m[t] = s / 129.0f;
    }
    /* m 的 reflect 补边 3+3(镜像反转), conv7 后取均值 => 全局基线标量 */
    float mp[14] = { m[3], m[2], m[1], m[0], m[1], m[2], m[3], m[4],
                     m[5], m[6], m[7], m[6], m[5], m[4] };
    float base = 0.0f;
    for (int k = 0; k < 8; k++) {
        float acc = 0.0f;
        for (int j = 0; j < 7; j++) acc += sv_adaptive_normalization_filter_[j] * mp[k + j];
        base += acc;
    }
    base /= 8.0f;

    static float feat[258][8];
    #ifdef SV_DUMP_E6
    if (g_dbg_win == 7) { FILE *df = fopen("c_feat_7.f32","wb"); fwrite(&feat[0][0],4,258*8,df); fwrite(&mag[0][0],4,129*8,df); fclose(df); }
#endif
    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++) {
            feat[f][t] = mag[f][t];
            feat[129 + f][t] = lg[f][t] - base;
        }

    /* ---------- 3. first_layer 残差块 (258->16) ---------- */
    static float proj[16][8], dwt[258][8], pw[16][8], act0[16][8];
    conv1x1(sv_first_layer_0_proj_weight, sv_first_layer_0_proj_bias,
            &feat[0][0], 258, 16, 8, 1, &proj[0][0], 0);
    dwconv5(sv_first_layer_0_dw_conv_0_weight, sv_first_layer_0_dw_conv_0_bias,
            &feat[0][0], 258, 8, &dwt[0][0], 1);
    conv1x1(sv_first_layer_0_pw_conv_0_weight, sv_first_layer_0_pw_conv_0_bias,
            &dwt[0][0], 258, 16, 8, 1, &pw[0][0], 0);
    for (int c = 0; c < 16; c++)
        for (int t = 0; t < 8; t++) {
            float v = pw[c][t] + proj[c][t];
            act0[c][t] = v < 0.0f ? 0.0f : v;
        }

    /* ---------- 4. encoder: 8 -> 4 -> 2 -> 1 ---------- */
    static float e0[16][4];
    static float pj1[32][4], d1[16][4], w1[32][4], act1[32][4];
    static float e2[32][2];
    static float d2[32][2], w2[32][2], act2[32][2];
    static float e4[32][1];
    static float pj3[64][1], d3[32][1], w3[64][1], act3[64][1];
    static float e6[64][1];

    conv1x1(sv_Conv_349, sv_Conv_350, &act0[0][0], 16, 16, 8, 2, &e0[0][0], 1);

    conv1x1(sv_encoder_3_0_proj_weight, sv_encoder_3_0_proj_bias,
            &e0[0][0], 16, 32, 4, 1, &pj1[0][0], 0);
    dwconv5(sv_encoder_3_0_dw_conv_0_weight, sv_encoder_3_0_dw_conv_0_bias,
            &e0[0][0], 16, 4, &d1[0][0], 1);
    conv1x1(sv_encoder_3_0_pw_conv_0_weight, sv_encoder_3_0_pw_conv_0_bias,
            &d1[0][0], 16, 32, 4, 1, &w1[0][0], 0);
    for (int c = 0; c < 32; c++)
        for (int t = 0; t < 4; t++) {
            float v = w1[c][t] + pj1[c][t];
            act1[c][t] = v < 0.0f ? 0.0f : v;
        }

    conv1x1(sv_Conv_352, sv_Conv_353, &act1[0][0], 32, 32, 4, 2, &e2[0][0], 1);

    dwconv5(sv_encoder_7_0_dw_conv_0_weight, sv_encoder_7_0_dw_conv_0_bias,
            &e2[0][0], 32, 2, &d2[0][0], 1);
    conv1x1(sv_encoder_7_0_pw_conv_0_weight, sv_encoder_7_0_pw_conv_0_bias,
            &d2[0][0], 32, 32, 2, 1, &w2[0][0], 0);
    for (int c = 0; c < 32; c++)
        for (int t = 0; t < 2; t++) {
            float v = w2[c][t] + e2[c][t];
            act2[c][t] = v < 0.0f ? 0.0f : v;
        }

    conv1x1(sv_Conv_355, sv_Conv_356, &act2[0][0], 32, 32, 2, 2, &e4[0][0], 1);

    conv1x1(sv_encoder_11_0_proj_weight, sv_encoder_11_0_proj_bias,
            &e4[0][0], 32, 64, 1, 1, &pj3[0][0], 0);
    dwconv5(sv_encoder_11_0_dw_conv_0_weight, sv_encoder_11_0_dw_conv_0_bias,
            &e4[0][0], 32, 1, &d3[0][0], 1);
    conv1x1(sv_encoder_11_0_pw_conv_0_weight, sv_encoder_11_0_pw_conv_0_bias,
            &d3[0][0], 32, 64, 1, 1, &w3[0][0], 0);
    for (int c = 0; c < 64; c++) {
        float v = w3[c][0] + pj3[c][0];
        act3[c][0] = v < 0.0f ? 0.0f : v;
    }

    conv1x1(sv_Conv_358, sv_Conv_359, &act3[0][0], 64, 64, 1, 1, &e6[0][0], 1);


    /* ---------- 5. LSTM x2 (每窗 1 帧) ---------- */
    float y1[64], y2[64];
    lstm_step(sv_LSTM_398, sv_LSTM_399, sv_LSTM_400, e6[0], st->lstm_h[0], st->lstm_c[0], 64);
    memcpy(y1, st->lstm_h[0], sizeof(y1));
    lstm_step(sv_LSTM_418, sv_LSTM_419, sv_LSTM_420, y1, st->lstm_h[1], st->lstm_c[1], 64);
    memcpy(y2, st->lstm_h[1], sizeof(y2));

    /* ---------- 6. decoder ---------- */
    float acc = sv_decoder_decoder_1_bias[0];
    for (int i = 0; i < 64; i++)
        acc += sv_decoder_decoder_1_weight[i] * (y2[i] < 0.0f ? 0.0f : y2[i]);
    g_last_prob = sigmoidf_(acc);
    return g_last_prob;
}
