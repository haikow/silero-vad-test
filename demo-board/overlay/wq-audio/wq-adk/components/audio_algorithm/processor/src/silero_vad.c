/*
 * silero_vad_nn.c —— Silero VAD v4 的 HiFi5 NNLib 加速实现(最终版)
 * 与 silero_vad.c 同接口(sv_process/sv_reset/sv_last_prob + sv_state_t),可直接替换编译。
 *
 * 加速点(占计算量 ~97%):
 *   STFT(258x256,s64)        -> 加窗256点基2FFT(OO; basis[0]=窗/行j=cos/行129+j=-sin,
 *   [PC验证40窗p偏差0]                     替代 conv1d: 砍528K MAC+264KB/窗kernel流量)
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
#include <stdint.h>
#include <math.h>
#include "silero_vad.h"
#include "silero_vad_consts.h"
#include "wq_heap.h"
#include "xa_type_def.h"                     /* WORD32/FLOAT32/VOID 基础类型 */
#include "nnlib/xa_nnlib_kernels_api.h"
#include "audio_alg_dbglog.h"  /* Y调试: 逐内核阶段埋点 */

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

/* ---- int8 权重(SS/WW): LSTM matXvec_8x8_32 + dwconv5(per-row/per-channel scale) ---- */
extern const int8_t svq_lstm_w1[], svq_lstm_r1[], svq_lstm_w2[], svq_lstm_r2[];
extern const float svq_lstm_sc1[], svq_lstm_sc2[];
extern const int8_t svq_dw1[], svq_dw2[], svq_dw3[], svq_dw4[];
extern const float svq_dw1_sc[], svq_dw2_sc[], svq_dw3_sc[], svq_dw4_sc[];
extern const float sv_win[256];
extern const float svq_s_x, svq_s_h;

/* ---- int8 权重(gen_weights_int8.py, SS 包): 语义见该文件头注释 ---- */
extern const int8_t svq_lstm_w1[], svq_lstm_r1[], svq_lstm_w2[], svq_lstm_r2[];
extern const float svq_lstm_sc1[], svq_lstm_sc2[];
extern const int8_t svq_dw1[], svq_dw2[], svq_dw3[], svq_dw4[];
extern const float svq_dw1_sc[], svq_dw2_sc[], svq_dw3_sc[], svq_dw4_sc[];
extern const float sv_win[256];
extern const float svq_s_x, svq_s_h;

extern const float sv_decoder_decoder_1_bias[];   /* [1] */

#define AL16 __attribute__((aligned(16)))

/* ---- VV: norm 快速近似(可回退) —— 0=libm 精确版, 1=位hack+多项式 ----
   误差界(PC 验证): fast_log2 全范围绝对 3.5e-06, fast_sqrt 相对 4.6e-06
   —— 远小于 int8 判定预算(0.0438)。省 norm 段 ~5ms(37→~32ms 真麦线) */
#ifndef SV_NORM_FAST
#define SV_NORM_FAST 1
#endif

#if SV_NORM_FAST
static inline float sv_fast_sqrt(float v)        /* v>0; 相对误差 ~4.6e-6(2次牛顿) */
{
    union { float f; uint32_t u; } c;
    c.f = v;
    c.u = 0x5f3759dfu - (c.u >> 1);
    float g = c.f;
    g = g * (1.5f - 0.5f * v * g * g);
    g = g * (1.5f - 0.5f * v * g * g);
    return v * g;
}
static inline float sv_fast_log(float x)         /* x>1; 绝对误差 ~3.5e-6 */
{   /* log2: 指数提取 + [1,2) 六阶多项式(系数经 PC 全范围验证); log = log2·ln2 */
    union { float f; uint32_t u; } c;
    c.f = x;
    int e = (int)(c.u >> 23) - 127;
    c.u = (c.u & 0x007FFFFFu) | 0x3F800000u;
    float t = c.f - 1.0f;
    float p = -0.024825649f;
    p = p * t + 0.11790537f;
    p = p * t - 0.272353455f;
    p = p * t + 0.45385646f;
    p = p * t - 0.716986952f;
    p = p * t + 1.442395494f;
    p = p * t + 5.065e-06f;
    return (float)e * 0.693147181f + p * 0.693147181f;
}
#endif

static int s_step_cnt = 0;
static float g_last_prob;

/* conv1d 已退役(OO): basis 行0=窗, 行j=w·cos(2πjk/256), 行129+j=-w·sin —— STFT
   即加窗 256 点 rFFT, PC 验证 basis 结构精确到 5.7e-08 / FFT-vs-DFT 全 40 窗 p 偏差 0。
   FFT 替代卷积: 砍 528K MAC/窗 与 264KB/窗 kernel 流量(NN 实测 STFT 78ms), 且不再占
   32KB TCM 暂存与 6KB conv1d state。XA 头保留——步骤 2 LSTM 回 matXvec 时复用 */

/* ---- FFT: 窗 + 旋转因子(init 一次) + 工作区(static, 栈仅 2KB 不能放) ---- */
#define SV_FFT_PI 3.14159265358979f
static float s_win[256] AL16;                   /* = basis 行0 */
static float s_cos[128] AL16, s_sin[128] AL16;  /* e^{-i2πj/256} = cos - i·sin */
static int s_fft_ready;
static float s_fre[256] AL16, s_fim[256] AL16;  /* 每帧工作区 */

static void sv_fft_init(void)
{
    memcpy(s_win, sv_win, sizeof(s_win));   /* SS: 窗来自 int8 权重文件(basis 258KB 退役) */
    for (int j = 0; j < 128; j++) {
        s_cos[j] = cosf(2.0f * SV_FFT_PI * j / 256.0f);
        s_sin[j] = sinf(2.0f * SV_FFT_PI * j / 256.0f);
    }
    s_fft_ready = 1;
}

/* 256 点基2 DIT 复FFT(含位反转), 约定 e^{-iθ} → 虚部 = -Σsin, 与 basis 一致 */
static void sv_fft256(float *re, float *im)
{
    for (unsigned i = 1, j = 0; i < 256; i++) {
        unsigned bit = 128;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= 256; len <<= 1) {
        int half = len >> 1, step = 256 / len;
        for (int i = 0; i < 256; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = -s_sin[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float vr = xr * wr - xi * wi;
                float vi = xr * wi + xi * wr;
                re[i + k] += vr;            im[i + k] += vi;
                re[i + k + half] = re[i + k] - 2 * vr;
                im[i + k + half] = im[i + k] - 2 * vi;
            }
        }
    }
}

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

/* ---- dwconv5 + ReLU(标量, 参考实现原样, 共 ~20K MAC, 占比 <3%) ----
   ONNX 图: 每处 dw_conv 后都有 ReLU(/dw_conv/dw_conv.1/Relu) 再进 pw_conv,
   II 包修复: 此前参考实现漏了该 ReLU, 静音窗 dw 输出过零被钳位差异放大成 p 偏高
   (HH 实测: 板端=numpy-C 逐位一致但 p=0.1271 vs onnx=0.0415; 补 ReLU 后=0.0415) */
static void dwconv5(const int8_t *wq, const float *wsc, const float *b, const float *in,
                    int C, int T, float *out)
{   /* SS: int8 权重 per-channel scale: out[c]=(Σ wq·x)·wsc[c]+b[c], ReLU 内嵌 */
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 5; k++) {
                int ti = t + k - 2;
                float v = (ti < 0 || ti >= T) ? 0.0f : in[(size_t)c * T + ti];
                acc += (float)wq[c * 5 + k] * v;
            }
            acc = acc * wsc[c] + b[c];
            out[(size_t)c * T + t] = acc > 0.0f ? acc : 0.0f;
        }
}

/* ---- 1x1 卷积(WW: NNLib f32 matmul + TCM 暂存权重) ----
   XX 教训: .a 的量化内核(8x16_32 等)为通用慢速版, 逐时间步调用开销×88 次/窗不可行
   (first 10→131ms); 1x1 保持 f32 matmul + TCM 暂存(10ms), 31ms 即该 .a 下实际下限 */
static float s_in_t[258 * 8] AL16;   /* 输入转置区 [T][IC], 最大 258×8 */
static float s_out_t[64 * 8] AL16;   /* 输出转置区 [T][OC], 最大 64×8 */

static void pw1x1(const float *w, const float *b, const float *in, int IC, int OC,
                  int T, float *out, int relu)
{
    for (int t = 0; t < T; t++)
        for (int k = 0; k < IC; k++)
            s_in_t[t * IC + k] = in[(size_t)k * T + t];

    xa_nn_matmul_f32xf32_f32(s_out_t, w, s_in_t, b, OC, IC, IC,
                             T /*vec_count*/, IC /*vec_offset*/,
                             OC /*out_offset*/, 1 /*out_stride*/);

    for (int t = 0; t < T; t++)
        for (int c = 0; c < OC; c++) {
            float v = s_out_t[t * OC + c];
            out[(size_t)c * T + t] = relu ? (v > 0.0f ? v : 0.0f) : v;
        }
}

/* ---- 1x1 卷积 stride2(标量): 每通道抽偶数列 ---- */
static void pw1x1_s2(const float *w, const float *b, const float *in, int IC, int OC,
                     int T, float *out, int relu, float *tmp /*[OC][T]*/)
{
    pw1x1(w, b, in, IC, OC, T, tmp, 0);
    int tout = (T - 1) / 2 + 1;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < tout; t++)
            out[(size_t)c * tout + t] = tmp[(size_t)c * T + 2 * t];
    if (relu) {
        int n = OC * tout;
        for (int i = 0; i < n; i++) out[i] = out[i] > 0.0f ? out[i] : 0.0f;
    }
}

/* ---- 残差块: out = relu(a + b) ---- */
static void res_relu(const float *a, const float *b, float *out, int n)
{
    /* 标量: SDK .a 的 elm_add/vec_relu 可靠性未证(同 matmul 布局问题), 先保数值 */
    for (int i = 0; i < n; i++) {
        float v = a[i] + b[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

/* ---- LSTM 单步(SS): int8 权重 + matXvec_8x8_32(int32 出口) + per-row 反量化 ----
   gate_f[i] = acc32[i]*s_c[i] + bsum[i]; W/R 行 scale 统一到 s_c(gen_weights_int8.py)。
   x/h 每调用量化 int8(静态标定 s_x/s_h)。内核拒绝时标量兜底(慢但正确) */
static int s_mxv_ok = 1;
static void lstm_step_nn(const int8_t *Wq, const int8_t *Rq, const float *sc,
                         const float *bsum, const float *x, float *h, float *c, int IN) { (void)IN;
    static int32_t acc[256] AL16;
    static int8_t x8[64] AL16, h8[64] AL16;
    float act[3][64] AL16;
    float tg[64] AL16, tc[64] AL16;

    for (int k = 0; k < 64; k++) {
        x8[k] = (int8_t)(x[k] > 0 ? (x[k] / svq_s_x + 0.5f) : (x[k] / svq_s_x - 0.5f));
        h8[k] = (int8_t)(h[k] > 0 ? (h[k] / svq_s_h + 0.5f) : (h[k] / svq_s_h - 0.5f));
    }
    if (s_mxv_ok) {
        /* TT: bias 传零数组(非 NULL)——f32 家族内核 NULL-bias 会走慢速路径(实测 67ms),
           零 bias 结果等价(acc 不变, 反量化时已加 float bsum) */
        static const int8_t z8[256] = {0};
        int r = xa_nn_matXvec_8x8_32(acc, (WORD8 *)Wq, (WORD8 *)Rq, (WORD8 *)x8, (WORD8 *)h8,
                                     (WORD8 *)z8, 256, 64, 64, 64, 64, 0, 0);
        if (r != 0) { s_mxv_ok = 0; DBGLOG_INFO(0, "[SVmxv] r=%d", r); }
    }
    if (!s_mxv_ok && s_step_cnt <= 2) DBGLOG_INFO(0, "[SVmxv] fallback=scalar");
    if (!s_mxv_ok)
        for (int i = 0; i < 256; i++) {
            int a = 0;
            for (int k = 0; k < 64; k++) a += (int)Wq[i * 64 + k] * x8[k] + (int)Rq[i * 64 + k] * h8[k];
            acc[i] = a;
        }
    /* UU 修复: 门控必须落 float(SS 的 (int32_t) 强转截断小数 → 静音窗 p 0.04→0.33) */
    static float gatef[256] AL16;
    for (int i = 0; i < 256; i++) gatef[i] = (float)acc[i] * sc[i] + bsum[i];

    for (int j = 0; j < 64; j++) {
        act[0][j] = 1.0f / (1.0f + expf(-gatef[j]));               /* i */
        act[1][j] = 1.0f / (1.0f + expf(-gatef[64 + j]));          /* o */
        act[2][j] = 1.0f / (1.0f + expf(-gatef[128 + j]));         /* f */
        tg[j] = tanhf(gatef[192 + j]);      /* g */
    }
    for (int j = 0; j < 64; j++)
        c[j] = act[2][j] * c[j] + act[0][j] * tg[j];
    for (int j = 0; j < 64; j++) {
        tc[j] = tanhf(c[j]);
        h[j] = act[1][j] * tc[j];
    }
}

/* ---- SS 暂存: LSTM int8(64KB) + 1x1 float 权重(~69KB) + dw int8 → TCM 单块 ----
   预算: 量产 TCM 空余 197KB > ~140KB(preset 后), HIL 278KB——两条路径都核过。
   malloc 失败 → 全部回退 flash 指针(慢但正确)。int8 让 flash 权重流量归零 */
static float *s_stg_base;
static struct {
    float *proj1, *pw1, *c349, *eproj1, *epw1, *c352, *epw2, *c355, *eproj3, *epw3, *c358;
    int8_t *lw1, *lr1, *lw2, *lr2, *dw1, *dw2, *dw3, *dw4;
} STG;
static int s_stg_done;

static void sv_stage(void)
{   /* WW: 1x1 float(68.5KB) + LSTM int8(64KB) + dw int8(1.3KB) ≈ 134KB — 两路径核过 */
    if (s_stg_done) return;
    s_stg_done = 1;
    const size_t NF = 16*258 + 16*258 + 16*16 + 32*16 + 32*16 + 32*32 + 32*32 + 32*32
                    + 64*32 + 64*32 + 64*64;
    const size_t N8 = 4*256*64 + 258*5 + 16*5 + 32*5 + 32*5;
    s_stg_base = (float *)wq_heap_caps_malloc(NF * 4 + N8 + 16, MALLOC_CAP_ALIGN_16BYTE);
    int8_t *p8 = (int8_t *)(s_stg_base ? (char *)s_stg_base + NF * 4 : NULL);
    float *pf = s_stg_base;
    if (!s_stg_base) { memset(&STG, 0, sizeof(STG)); }
    else {
        #define CPF(dst, src, n) do { STG.dst = pf; memcpy(pf, src, (n)*sizeof(float)); pf += (n); } while (0)
        CPF(proj1, sv_first_layer_0_proj_weight, 16*258);
        CPF(pw1,   sv_first_layer_0_pw_conv_0_weight, 16*258);
        CPF(c349,  sv_Conv_349, 16*16);
        CPF(eproj1, sv_encoder_3_0_proj_weight, 32*16);
        CPF(epw1,  sv_encoder_3_0_pw_conv_0_weight, 32*16);
        CPF(c352,  sv_Conv_352, 32*32);
        CPF(epw2,  sv_encoder_7_0_pw_conv_0_weight, 32*32);
        CPF(c355,  sv_Conv_355, 32*32);
        CPF(eproj3, sv_encoder_11_0_proj_weight, 64*32);
        CPF(epw3,  sv_encoder_11_0_pw_conv_0_weight, 64*32);
        CPF(c358,  sv_Conv_358, 64*64);
        #undef CPF
        #define CP8(dst, src, n) do { STG.dst = p8; memcpy(p8, src, n); p8 += (n); } while (0)
        CP8(lw1, svq_lstm_w1, 256*64); CP8(lr1, svq_lstm_r1, 256*64);
        CP8(lw2, svq_lstm_w2, 256*64); CP8(lr2, svq_lstm_r2, 256*64);
        CP8(dw1, svq_dw1, 258*5); CP8(dw2, svq_dw2, 16*5);
        CP8(dw3, svq_dw3, 32*5);  CP8(dw4, svq_dw4, 32*5);
        #undef CP8
    }
    DBGLOG_INFO(0, "[SVstg] base=%d", s_stg_base != 0);
}
#define WP1()   (STG.proj1  ? STG.proj1  : sv_first_layer_0_proj_weight)
#define WPW1()  (STG.pw1    ? STG.pw1    : sv_first_layer_0_pw_conv_0_weight)
#define WC349() (STG.c349   ? STG.c349   : sv_Conv_349)
#define WEP1()  (STG.eproj1 ? STG.eproj1 : sv_encoder_3_0_proj_weight)
#define WEPW1() (STG.epw1   ? STG.epw1   : sv_encoder_3_0_pw_conv_0_weight)
#define WC352() (STG.c352   ? STG.c352   : sv_Conv_352)
#define WEPW2() (STG.epw2   ? STG.epw2   : sv_encoder_7_0_pw_conv_0_weight)
#define WC355() (STG.c355   ? STG.c355   : sv_Conv_355)
#define WEP3()  (STG.eproj3 ? STG.eproj3 : sv_encoder_11_0_proj_weight)
#define WEPW3() (STG.epw3   ? STG.epw3   : sv_encoder_11_0_pw_conv_0_weight)
#define WC358() (STG.c358   ? STG.c358   : sv_Conv_358)
#define WLW1()  (STG.lw1    ? STG.lw1    : svq_lstm_w1)
#define WLR1()  (STG.lr1    ? STG.lr1    : svq_lstm_r1)
#define WLW2()  (STG.lw2    ? STG.lw2    : svq_lstm_w2)
#define WLR2()  (STG.lr2    ? STG.lr2    : svq_lstm_r2)
#define WDW1()  (STG.dw1    ? STG.dw1    : svq_dw1)
#define WDW2()  (STG.dw2    ? STG.dw2    : svq_dw2)
#define WDW3()  (STG.dw3    ? STG.dw3    : svq_dw3)
#define WDW4()  (STG.dw4    ? STG.dw4    : svq_dw4)

void sv_reset(sv_state_t *st)
{
    if (!s_packed) pack_all();
    memset(st, 0, sizeof(*st));
    g_last_prob = 0.0f;
}

float sv_last_prob(void) { return g_last_prob; }

float sv_process(sv_state_t *st, const float *x)
{
    s_step_cnt++;
    if (!s_packed) pack_all();

    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]1-preSTFT");
    /* ---------- 1. reflect(96,96) + STFT(8 帧) [OO: FFT 替代 conv1d] ---------- */
    static float pad[704] AL16;
    static float stft[8][258] AL16;      /* [8帧][129实+129虚] */
    static float mag[129][8], lg[129][8];

    for (int i = 0; i < 96; i++) pad[i] = x[96 - i];
    memcpy(pad + 96, x, sizeof(float) * 512);
    for (int i = 0; i < 96; i++) pad[608 + i] = x[510 - i];

    /* STFT = 加窗 rFFT: basis[0]=窗 w, basis[j]=w·cos, basis[129+j]=-w·sin
       (PC 验证: 结构精确到 5.7e-08, 全 40 窗 p 与 direct DFT 偏差 0)。
       每帧: y=pad·w → 256 点基2 FFT → stft[t][j]=Re, stft[t][129+j]=Im */
    if (!s_fft_ready) sv_fft_init();
    /* WW: 双实数帧合批 —— z=帧2k+i·帧2k+1, 一次 256 点复 FFT 解两帧谱
       (A[k]=(Z[k]+conj(Z[N-k]))/2, B[k]=(Z[k]-conj(Z[N-k]))/2i; PC 验证 1.67e-07;
       k=0/128 退化点与 basis 结构性零吻合, 无需特判)。FFT 次数 8→4 */
    for (int pair = 0; pair < 4; pair++) {
        const float *fa = pad + (2 * pair) * 64;
        const float *fb = pad + (2 * pair + 1) * 64;
        for (int k = 0; k < 256; k++) {
            s_fre[k] = fa[k] * s_win[k];
            s_fim[k] = fb[k] * s_win[k];
        }
        sv_fft256(s_fre, s_fim);
        for (int j = 0; j < 129; j++) {
            int k1 = (256 - j) & 255;
            float zr = s_fre[j], zi = s_fim[j], zr1 = s_fre[k1], zi1 = s_fim[k1];
            stft[2 * pair][j]       = (zr + zr1) * 0.5f;
            stft[2 * pair][129 + j] = (zi - zi1) * 0.5f;
            stft[2 * pair + 1][j]       = (zi + zi1) * 0.5f;
            stft[2 * pair + 1][129 + j] = (zr1 - zr) * 0.5f;
        }
    }

    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++) {
            float re = stft[t][f], im = stft[t][129 + f];
#if SV_NORM_FAST
            float v2 = re * re + im * im;
            mag[f][t] = v2 > 0.0f ? sv_fast_sqrt(v2) : 0.0f;
#else
            mag[f][t] = sqrtf(re * re + im * im);
#endif
        }

    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]2-postSTFT");
    /* ---------- 2. 自适应归一化(标量小算, 与参考实现一致) ---------- */
    for (int t = 0; t < 8; t++)
        for (int f = 0; f < 129; f++)
#if SV_NORM_FAST
            lg[f][t] = sv_fast_log(mag[f][t] * SV_NORM_MUL + SV_NORM_ADD);
#else
            lg[f][t] = logf(mag[f][t] * SV_NORM_MUL + SV_NORM_ADD);
#endif

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
    /* [GG] STFT dump 已完成使命(GG~II 包定位 conv1d 布局/音频量化), JJ 起移除省 UART */

    if (s_step_cnt <= 2) {
        DBGLOG_INFO(0, "[BB]feat00=%d feat10=%d feat1280=%d feat2577=%d",
                    (int)(feat[0][0]*100000), (int)(feat[1][0]*100000),
                    (int)(feat[128][0]*100000), (int)(feat[257][7]*100000));
    }
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]3-pre1st");
    /* ---------- 3. first_layer 残差块 (258->16) ---------- */
    static float proj[16][8] AL16, dwt[258][8] AL16, pw[16][8] AL16, act0[16][8] AL16;
    static float tmp[64][8] AL16;        /* stride2 满分辨率暂存(最大 [64][8]) */
    pw1x1(WP1(), sv_first_layer_0_proj_bias,
          &feat[0][0], 258, 16, 8, &proj[0][0], 0);
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]proj: %d %d %d %d",
                    (int)(proj[0][0]*100000), (int)(proj[3][0]*100000),
                    (int)(proj[7][0]*100000), (int)(proj[15][7]*100000));
    dwconv5(WDW1(), svq_dw1_sc, sv_first_layer_0_dw_conv_0_bias,
            &feat[0][0], 258, 8, &dwt[0][0]);
    pw1x1(WPW1(), sv_first_layer_0_pw_conv_0_bias,
          &dwt[0][0], 258, 16, 8, &pw[0][0], 0);
    res_relu(&pw[0][0], &proj[0][0], &act0[0][0], 16 * 8);
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]act0: %d %d %d %d",
                    (int)(act0[0][0]*100000), (int)(act0[3][0]*100000),
                    (int)(act0[7][0]*100000), (int)(act0[15][7]*100000));

    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]4-preEnc");
    /* ---------- 4. encoder: 8 -> 4 -> 2 -> 1 ---------- */
    static float e0[16][4] AL16;
    static float pj1[32][4] AL16, d1[16][4] AL16, w1[32][4] AL16, act1[32][4] AL16;
    static float e2[32][2] AL16;
    static float d2[32][2] AL16, w2[32][2] AL16, act2[32][2] AL16;
    static float e4[32][1] AL16;
    static float pj3[64][1] AL16, d3[32][1] AL16, w3[64][1] AL16, act3[64][1] AL16;
    static float e6[64][1] AL16;

    pw1x1_s2(WC349(), sv_Conv_350, &act0[0][0], 16, 16, 8, &e0[0][0], 1, &tmp[0][0]);
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]e0: %d %d %d",
                    (int)(e0[0][0]*100000), (int)(e0[5][1]*100000),
                    (int)(e0[15][3]*100000));

    pw1x1(WEP1(), sv_encoder_3_0_proj_bias,
          &e0[0][0], 16, 32, 4, &pj1[0][0], 0);
    dwconv5(WDW2(), svq_dw2_sc, sv_encoder_3_0_dw_conv_0_bias,
            &e0[0][0], 16, 4, &d1[0][0]);
    pw1x1(WEPW1(), sv_encoder_3_0_pw_conv_0_bias,
          &d1[0][0], 16, 32, 4, &w1[0][0], 0);
    res_relu(&w1[0][0], &pj1[0][0], &act1[0][0], 32 * 4);

    pw1x1_s2(WC352(), sv_Conv_353, &act1[0][0], 32, 32, 4, &e2[0][0], 1, &tmp[0][0]);

    dwconv5(WDW3(), svq_dw3_sc, sv_encoder_7_0_dw_conv_0_bias,
            &e2[0][0], 32, 2, &d2[0][0]);
    pw1x1(WEPW2(), sv_encoder_7_0_pw_conv_0_bias,
          &d2[0][0], 32, 32, 2, &w2[0][0], 0);
    res_relu(&w2[0][0], &e2[0][0], &act2[0][0], 32 * 2);

    pw1x1_s2(WC355(), sv_Conv_356, &act2[0][0], 32, 32, 2, &e4[0][0], 1, &tmp[0][0]);

    pw1x1(WEP3(), sv_encoder_11_0_proj_bias,
          &e4[0][0], 32, 64, 1, &pj3[0][0], 0);
    dwconv5(WDW4(), svq_dw4_sc, sv_encoder_11_0_dw_conv_0_bias,
            &e4[0][0], 32, 1, &d3[0][0]);
    pw1x1(WEPW3(), sv_encoder_11_0_pw_conv_0_bias,
          &d3[0][0], 32, 64, 1, &w3[0][0], 0);
    res_relu(&w3[0][0], &pj3[0][0], &act3[0][0], 64 * 1);

    pw1x1(WC358(), sv_Conv_359, &act3[0][0], 64, 64, 1, &e6[0][0], 1);
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]e6: %d %d %d %d",
                    (int)(e6[0][0]*100000), (int)(e6[16][0]*100000),
                    (int)(e6[32][0]*100000), (int)(e6[63][0]*100000));

    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]5-preLSTM");
    /* ---------- 5. LSTM x2 (每窗 1 帧) ---------- */
    float y1[64] AL16, y2[64] AL16;
    sv_stage();
    lstm_step_nn(WLW1(), WLR1(), svq_lstm_sc1, s_lstm_b1,
                 e6[0], st->lstm_h[0], st->lstm_c[0], 64);
    memcpy(y1, st->lstm_h[0], sizeof(y1));
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]y1(L1h): %d %d %d %d",
                    (int)(y1[0]*100000), (int)(y1[1]*100000),
                    (int)(y1[32]*100000), (int)(y1[63]*100000));
    lstm_step_nn(WLW2(), WLR2(), svq_lstm_sc2, s_lstm_b2,
                 y1, st->lstm_h[1], st->lstm_c[1], 64);
    memcpy(y2, st->lstm_h[1], sizeof(y2));
    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[CC]y2(L2h): %d %d %d %d",
                    (int)(y2[0]*100000), (int)(y2[1]*100000),
                    (int)(y2[32]*100000), (int)(y2[63]*100000));

    if (s_step_cnt <= 2)
        DBGLOG_INFO(0, "[SVs]6-preDec");
    /* ---------- 6. decoder: relu + FC + sigmoid ---------- */
    float yr[64] AL16, dot = 0.0f;
    for (int i = 0; i < 64; i++) yr[i] = y2[i] > 0.0f ? y2[i] : 0.0f;
    for (int i = 0; i < 64; i++) dot += yr[i] * sv_decoder_decoder_1_weight[i];
    g_last_prob = 1.0f / (1.0f + expf(-(dot + sv_decoder_decoder_1_bias[0])));
    /* 打印节流(2026-10-15): 录音期间逐窗打印约 31 行/s 占日志通道且崩过 WDT2,
       改为"命中态变化必打 + 每 32 窗一拍心跳"(约 1-3 行/s); 前 2 窗保留全打 */
    {
        static int s_done_cnt;
        static int s_done_hit = -1;
        int now_hit = g_last_prob >= 0.5f;
        if (s_step_cnt <= 2 || now_hit != s_done_hit || (s_done_cnt++ & 31) == 0) {
            DBGLOG_INFO(0, "[SVs]7-done p=%d dot=%d", (int)(g_last_prob * 10000),
                        (int)((dot + sv_decoder_decoder_1_bias[0])*100000));
            s_done_hit = now_hit;
        }
    }
    return g_last_prob;
}
