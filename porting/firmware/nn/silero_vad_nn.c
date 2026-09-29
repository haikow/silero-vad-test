/*
 * silero_vad_nn.c —— Silero VAD v4 的 HiFi5 NNLib 加速实现(最终版)
 * 与 silero_vad.c 同接口(sv_process/sv_reset/sv_last_prob + sv_state_t),可直接替换编译。
 *
 * 加速点(占计算量 ~97%):
 *   STFT(258x256,s64)        -> xa_nn_conv1d_std_f32   [~528K MAC/窗, 热点之首]
 *   全部 1x1 卷积(proj/pw/s2) -> xa_nn_matmul_f32xf32_f32 (CHW 路径, 原生 [C][T] 布局)
 *   LSTM 门(4x[64x64+64xIN])  -> xa_nn_matXvec_f32xf32_f32 (一次产出 256 行 = 4 门)
 *   sigmoid/tanh              -> xa_nn_vec_sigmoid/tanh_f32_f32
 *   残差加                     -> xa_nn_elm_add_f32xf32_f32
 *   decoder FC                 -> xa_nn_dot_prod_f32xf32_f32
 * 保留标量(<3% 计算量,规避 NNLib depthwise 布局歧义):
 *   dwconv5(4 组, 共 ~20K MAC) + 幅值/对数/归一化前端 + decoder sigmoid
 *
 * 数据布局: 全程 [C][T] 行主序(与参考实现 silero_vad.c 完全一致,权重零重排零转置):
 *   - matmul: 权重 [OC][IC](row_stride=IC) / 输入 [IC][T](vec_offset=IC) / 输出 [OC][T]
 *     (out_offset=1, out_stride=vec_count 即 CHW 输出, 见 xa_nn_matmul_f32.c 源码注释)
 *   - conv1d: IC=1 时 CHW/HWC 同构, 输出 fmt=0 为 [8][258], 由前端索引直接吸收
 *   - LSTM 偏置在首次调用时合并(Wb+Rb -> [256], matXvec 只收一个 bias)
 *
 * 对齐: f32 内核参数检查仅要求 4 字节; 缓冲区加 aligned(16) 以命中 matmul 快路径
 *       (快路径还需权重 16B 对齐——可选优化: gen_weights.py 给数组加 aligned(16))。
 *
 * 编译: 仅 xt-clang(内核含 Xtensa TIE 代码; GCC 实测不可行, 见 NNLib优化方案.md §6)
 * 验证: 固件 [SHIL] 自检 vs golden_prob.f32(线 0.1); 数值应与参考实现一致(同 f32 运算,
 *       仅累加顺序不同, 偏差应在 1e-5 量级)
 */
#include <string.h>
#include <math.h>
#include "silero_vad.h"
#include "silero_vad_consts.h"
#include "xa_type_def.h"                     /* WORD32/FLOAT32/VOID 基础类型 */
#include "nnlib/xa_nnlib_kernels_api.h"

/* ---- 权重(silero_vad_weights.c, 布局原生匹配, 零重排) ---- */
extern const float sv_feature_extractor_forward_basis_buffer[]; /* [258*256] = [OC][KH][1] */
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
extern const float sv_LSTM_398[], sv_LSTM_399[], sv_LSTM_400[]; /* L1 W[256][64],R[256][64],B[512] */
extern const float sv_LSTM_418[], sv_LSTM_419[], sv_LSTM_420[]; /* L2 同构 */
extern const float sv_decoder_decoder_1_weight[]; /* [64] */
extern const float sv_decoder_decoder_1_bias[];   /* [1] */

#define AL16 __attribute__((aligned(16)))

static float g_last_prob;

/* conv1d 需要 bias 与 scratch 状态指针(NULL 会被参数检查拒绝) */
static const float s_zero_bias[258] = {0};
static long long s_conv1d_state[48] AL16;       /* xa_nn_conv_state_t 足量(384B) */

/* LSTM 合并偏置(Wb+Rb), 首次调用时生成 */
static float s_lstm_b1[256] AL16, s_lstm_b2[256] AL16;
static int s_packed = 0;

static void pack_all(void)
{
    for (int i = 0; i < 256; i++) {
        s_lstm_b1[i] = sv_LSTM_400[i] + sv_LSTM_400[256 + i];
        s_lstm_b2[i] = sv_LSTM_420[i] + sv_LSTM_420[256 + i];
    }
    s_packed = 1;
}

/* ---- dwconv5 标量(参考实现原样, 共 ~20K MAC, 占比 <3%) ---- */
static void dwconv5(const float *w, const float *b, const float *in, int C, int T,
                    float *out)
{
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            float acc = b[c];
            for (int k = 0; k < 5; k++) {
                int ti = t + k - 2;
                float v = (ti < 0 || ti >= T) ? 0.0f : in[(size_t)c * T + ti];
                acc += w[c * 5 + k] * v;
            }
            out[(size_t)c * T + t] = acc;
        }
}

/* ---- 1x1 卷积(NNLib matmul, CHW): in[IC][T] -> out[OC][T], relu 可选 ---- */
static void pw1x1(const float *w, const float *b, const float *in, int IC, int OC,
                  int T, float *out, int relu)
{
    xa_nn_matmul_f32xf32_f32(out, (FLOAT32 *)w, (FLOAT32 *)in, (FLOAT32 *)b,
                             OC /*rows*/, IC /*cols1*/, IC /*row_stride1*/,
                             T /*vec_count*/, IC /*vec_offset*/,
                             1 /*out_offset: CHW*/, T /*out_stride*/);
    if (relu) xa_nn_vec_relu_f32_f32(out, out, 0.0f, OC * T);
}

/* ---- 1x1 卷积 stride2: 满算后每通道行抽偶数列 ---- */
static void pw1x1_s2(const float *w, const float *b, const float *in, int IC, int OC,
                     int T, float *out, int relu, float *tmp /*[OC][T]*/)
{
    xa_nn_matmul_f32xf32_f32(tmp, (FLOAT32 *)w, (FLOAT32 *)in, (FLOAT32 *)b,
                             OC, IC, IC, T, IC, 1, T);
    int tout = (T - 1) / 2 + 1;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < tout; t++)
            out[(size_t)c * tout + t] = tmp[(size_t)c * T + 2 * t];
    if (relu) xa_nn_vec_relu_f32_f32(out, out, 0.0f, OC * tout);
}

/* ---- 残差块: out = relu(a + b) ---- */
static void res_relu(const float *a, const float *b, float *out, int n)
{
    xa_nn_elm_add_f32xf32_f32(out, a, b, n);
    xa_nn_vec_relu_f32_f32(out, out, 0.0f, n);
}

/* ---- LSTM 单步(NNLib): W[256][IN], R[256][64], 门序 i,o,f,c ---- */
static void lstm_step_nn(const float *W, const float *R, const float *bsum,
                         const float *x, float *h, float *c, int IN)
{
    float gates[256] AL16;      /* matXvec 一次产出 4 门 */
    float act[3][64] AL16;      /* i, o, f 激活结果 */
    float tg[64] AL16, tc[64] AL16;

    xa_nn_matXvec_f32xf32_f32(gates, W, R, x, h, bsum,
                              256 /*rows*/, IN /*cols1*/, 64 /*cols2*/,
                              IN /*row_stride1*/, 64 /*row_stride2*/);

    xa_nn_vec_sigmoid_f32_f32(act[0], gates + 0, 64);        /* i */
    xa_nn_vec_sigmoid_f32_f32(act[1], gates + 64, 64);       /* o */
    xa_nn_vec_sigmoid_f32_f32(act[2], gates + 128, 64);      /* f */
    xa_nn_vec_tanh_f32_f32(tg, gates + 192, 64);             /* g */

    for (int j = 0; j < 64; j++)
        c[j] = act[2][j] * c[j] + act[0][j] * tg[j];

    xa_nn_vec_tanh_f32_f32(tc, c, 64);
    for (int j = 0; j < 64; j++)
        h[j] = act[1][j] * tc[j];
}

void sv_reset(sv_state_t *st)
{
    if (!s_packed) pack_all();
    memset(st, 0, sizeof(*st));
    g_last_prob = 0.0f;
}

float sv_last_prob(void) { return g_last_prob; }

float sv_process(sv_state_t *st, const float *x)
{
    if (!s_packed) pack_all();

    /* ---------- 1. reflect(96,96) + STFT(8 帧) [热点 -> NNLib conv1d] ---------- */
    static float pad[704] AL16;          /* [704][1][1](IC=1, CHW/HWC 同构) */
    static float stft[8][258] AL16;      /* conv1d fmt=0 输出 [out_H=8][OC=258] */
    static float mag[129][8], lg[129][8];

    for (int i = 0; i < 96; i++) pad[i] = x[96 - i];
    memcpy(pad + 96, x, sizeof(float) * 512);
    for (int i = 0; i < 96; i++) pad[608 + i] = x[510 - i];

    xa_nn_conv1d_std_f32(&stft[0][0], pad,
                         (FLOAT32 *)sv_feature_extractor_forward_basis_buffer,
                         (FLOAT32 *)s_zero_bias,
                         704 /*input_height=time*/, 1 /*input_width*/, 1 /*in_ch*/,
                         256 /*kernel_height*/, 258 /*out_channels*/,
                         64 /*y_stride*/, 0 /*y_padding*/, 8 /*out_height*/,
                         0 /*out_data_format: [H][C]*/, (VOID *)s_conv1d_state);

    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++) {
            float re = stft[t][f], im = stft[t][129 + f];
            mag[f][t] = sqrtf(re * re + im * im);
        }

    /* ---------- 2. 自适应归一化(标量小算, 与参考实现一致) ---------- */
    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++)
            lg[f][t] = logf(mag[f][t] * SV_NORM_MUL + SV_NORM_ADD);

    float m[8];
    for (int t = 0; t < 8; t++) {
        float s = 0.0f;
        for (int f = 0; f < 129; f++) s += lg[f][t];
        m[t] = s / 129.0f;
    }
    float mp[14] = { m[3], m[2], m[1], m[0], m[1], m[2], m[3], m[4],
                     m[5], m[6], m[7], m[6], m[5], m[4] };
    float base = 0.0f;
    for (int k = 0; k < 8; k++) {
        float acc = 0.0f;
        for (int j = 0; j < 7; j++) acc += sv_adaptive_normalization_filter_[j] * mp[k + j];
        base += acc;
    }
    base /= 8.0f;

    static float feat[258][8] AL16;      /* [C][T] 全网主布局 */
    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++) {
            feat[f][t] = mag[f][t];
            feat[129 + f][t] = lg[f][t] - base;
        }

    /* ---------- 3. first_layer 残差块 (258->16) ---------- */
    static float proj[16][8] AL16, dwt[258][8] AL16, pw[16][8] AL16, act0[16][8] AL16;
    static float tmp[64][8] AL16;        /* stride2 满分辨率暂存(最大 [64][8]) */
    pw1x1(sv_first_layer_0_proj_weight, sv_first_layer_0_proj_bias,
          &feat[0][0], 258, 16, 8, &proj[0][0], 0);
    dwconv5(sv_first_layer_0_dw_conv_0_weight, sv_first_layer_0_dw_conv_0_bias,
            &feat[0][0], 258, 8, &dwt[0][0]);
    pw1x1(sv_first_layer_0_pw_conv_0_weight, sv_first_layer_0_pw_conv_0_bias,
          &dwt[0][0], 258, 16, 8, &pw[0][0], 0);
    res_relu(&pw[0][0], &proj[0][0], &act0[0][0], 16 * 8);

    /* ---------- 4. encoder: 8 -> 4 -> 2 -> 1 ---------- */
    static float e0[16][4] AL16;
    static float pj1[32][4] AL16, d1[16][4] AL16, w1[32][4] AL16, act1[32][4] AL16;
    static float e2[32][2] AL16;
    static float d2[32][2] AL16, w2[32][2] AL16, act2[32][2] AL16;
    static float e4[32][1] AL16;
    static float pj3[64][1] AL16, d3[32][1] AL16, w3[64][1] AL16, act3[64][1] AL16;
    static float e6[64][1] AL16;

    pw1x1_s2(sv_Conv_349, sv_Conv_350, &act0[0][0], 16, 16, 8, &e0[0][0], 1, &tmp[0][0]);

    pw1x1(sv_encoder_3_0_proj_weight, sv_encoder_3_0_proj_bias,
          &e0[0][0], 16, 32, 4, &pj1[0][0], 0);
    dwconv5(sv_encoder_3_0_dw_conv_0_weight, sv_encoder_3_0_dw_conv_0_bias,
            &e0[0][0], 16, 4, &d1[0][0]);
    pw1x1(sv_encoder_3_0_pw_conv_0_weight, sv_encoder_3_0_pw_conv_0_bias,
          &d1[0][0], 16, 32, 4, &w1[0][0], 0);
    res_relu(&w1[0][0], &pj1[0][0], &act1[0][0], 32 * 4);

    pw1x1_s2(sv_Conv_352, sv_Conv_353, &act1[0][0], 32, 32, 4, &e2[0][0], 1, &tmp[0][0]);

    dwconv5(sv_encoder_7_0_dw_conv_0_weight, sv_encoder_7_0_dw_conv_0_bias,
            &e2[0][0], 32, 2, &d2[0][0]);
    pw1x1(sv_encoder_7_0_pw_conv_0_weight, sv_encoder_7_0_pw_conv_0_bias,
          &d2[0][0], 32, 32, 2, &w2[0][0], 0);
    res_relu(&w2[0][0], &e2[0][0], &act2[0][0], 32 * 2);

    pw1x1_s2(sv_Conv_355, sv_Conv_356, &act2[0][0], 32, 32, 2, &e4[0][0], 1, &tmp[0][0]);

    pw1x1(sv_encoder_11_0_proj_weight, sv_encoder_11_0_proj_bias,
          &e4[0][0], 32, 64, 1, &pj3[0][0], 0);
    dwconv5(sv_encoder_11_0_dw_conv_0_weight, sv_encoder_11_0_dw_conv_0_bias,
            &e4[0][0], 32, 1, &d3[0][0]);
    pw1x1(sv_encoder_11_0_pw_conv_0_weight, sv_encoder_11_0_pw_conv_0_bias,
          &d3[0][0], 32, 64, 1, &w3[0][0], 0);
    res_relu(&w3[0][0], &pj3[0][0], &act3[0][0], 64 * 1);

    pw1x1(sv_Conv_358, sv_Conv_359, &act3[0][0], 64, 64, 1, &e6[0][0], 1);

    /* ---------- 5. LSTM x2 (每窗 1 帧) ---------- */
    float y1[64] AL16, y2[64] AL16;
    lstm_step_nn(sv_LSTM_398, sv_LSTM_399, s_lstm_b1,
                 e6[0], st->lstm_h[0], st->lstm_c[0], 64);
    memcpy(y1, st->lstm_h[0], sizeof(y1));
    lstm_step_nn(sv_LSTM_418, sv_LSTM_419, s_lstm_b2,
                 y1, st->lstm_h[1], st->lstm_c[1], 64);
    memcpy(y2, st->lstm_h[1], sizeof(y2));

    /* ---------- 6. decoder: relu + FC + sigmoid ---------- */
    float yr[64] AL16, dot = 0.0f;
    xa_nn_vec_relu_f32_f32(yr, y2, 0.0f, 64);
    xa_nn_dot_prod_f32xf32_f32(&dot, yr, sv_decoder_decoder_1_weight, 64, 1);
    g_last_prob = 1.0f / (1.0f + expf(-(dot + sv_decoder_decoder_1_bias[0])));
    return g_last_prob;
}
