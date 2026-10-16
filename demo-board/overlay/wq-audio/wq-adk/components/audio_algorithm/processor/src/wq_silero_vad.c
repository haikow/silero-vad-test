/*
 * wq_silero_vad.c —— 物奇 libwq_sw_vad 五函数接口的 Silero VAD 实现
 * 详见 wq_silero_vad.h 头注释。
 */
#include <string.h>
#include <math.h>
#include "wq_silero_vad.h"
#include "silero_vad.h"
#include "audio_alg_dbglog.h"
#include "dfs.h"

#define WQ_SILERO_ACC 512

/* ---- 录音态降载(2026-10-15, WDT2 修复; 2026-10-16 v6 修正) ----
 * 实测(17:43 日志): 本地录音并发 opus 编码时窗完成率仅 62%(19.5/s vs 实时 31.25/s),
 * 全速 Silero(31ms/窗 ≈ 97% 实时预算) + opus + 管线 → dcore 超支, 30s 后 WDT2 咬死
 * (原厂轻量 VAD <1ms/窗 无此问题)。
 * v5 教训: 曾用"2s 静音退出降载"——但说话句间停顿常超 2s, 录音中途退回全速又超支,
 * 板验(10-16 日志)录音期 run too long 44ms×146 次, 吞吐贴地 1.0×无余量。
 * v6 定案: **命中后降载持续整个流实例生命周期, 不再退出**。每个 TVAD 候选窗都是新
 * 实例(wq_sw_vad_init 清零 hd→全速), 首响时延不受影响; 命中后的录音/监测期一律
 * 每 REC_SKIP 窗推理 1 窗, 端点判定粒度 192ms 仍可接受(厂商停录判定 700ms)。
 * v7 自适应跳窗: v6 固定 1/6 把语音期判定密度也压掉了(录音中停录判定变钝)。
 * 改为按能量自适应: 噪声底 EMA 跟踪, 静音窗(≤2.5×底噪)跳 6 窗省预算;
 * 活跃窗(>2.5×底噪, 语音/持续噪声)跳 3 窗保判定密度(v5 验证过的安全档)。
 * 预算账: 说话期均摊 6.5ms/帧+opus~5ms ≈ 12ms < 20ms(实测 v5 就是这个档跑通的);
 * 静音期 3.2ms/帧, 余量更大 */
#define WQ_SILERO_SKIP_QUIET 6
#define WQ_SILERO_SKIP_ACTIVE 3
#define WQ_SILERO_NF_UP 0.05f    /* 噪声底 EMA 上升系数 */
#define WQ_SILERO_NF_DN 0.005f   /* 下降系数(慢降, 防语音把底噪顶高) */
#define WQ_SILERO_NF_RATIO 2.5f  /* 活跃判定: RMS > 2.5×噪声底 */

/* demo 板 DFS 降频修复: 厂商 demo 电源管理会把 dcore 降到 16/32MHz,
   Silero 需 192M 才能实时(31ms/窗)。v7 起改会话级: init 时申请 192M,
   流销毁时由上层 sw_vad.c 调 wq_sw_vad_exit() 释放——空闲回到厂商
   原生 DFS(TVAD 硬件常开微安级), 只有候选窗/录音期上 192M。
   安全性: 录音流(含 opus)有自己的 DFS floor(64M)请求, 释放我们的
   192M 不会饿着 opus(DFS 引用计数各自独立) */
static dfs_object_t *s_dfs_obj;

static void sv_dfs_hold(void)
{
    if (s_dfs_obj) return;
    s_dfs_obj = dfs_request(WQ_FREQ_192M, 0);
}

void wq_sw_vad_exit(void)
{
    if (s_dfs_obj) {
        dfs_release(s_dfs_obj);
        s_dfs_obj = NULL;
    }
}

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
    sv_dfs_hold();
    wq_silero_hd_t *hd = (wq_silero_hd_t *)vad_hd;
    memset(hd, 0, sizeof(*hd));
    return hd;
}

#define SILERO_STUB 0   /* 诊断构建A: 1=跳过推理直接判无语音(定位 WDT 崩溃用) */

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
            /* v7: 每窗(无论是否推理)先算 RMS, 维护噪声底 EMA, 决定跳窗档位 */
            float e = 0.0f;
            for (int i = 0; i < WQ_SILERO_ACC; i++) e += hd->acc[i] * hd->acc[i];
            float rms = sqrtf(e / WQ_SILERO_ACC);
            if (rms > hd->noise_floor)
                hd->noise_floor += WQ_SILERO_NF_UP * (rms - hd->noise_floor);
            else
                hd->noise_floor += WQ_SILERO_NF_DN * (rms - hd->noise_floor);
            int active = rms > (hd->noise_floor * WQ_SILERO_NF_RATIO);
            int skip = active ? WQ_SILERO_SKIP_ACTIVE : WQ_SILERO_SKIP_QUIET;

            /* 录音态降载: 命中后按档位跳窗, 跳过的窗沿用上一窗判定。
               不设退出条件(v5 教训); 降载随 hd 生命周期结束——下一候选窗
               是新实例, 自动恢复全速保首响。v7 自适应: 静音 1/6 省预算,
               活跃(语音出现)回 1/3 保停录判定密度 */
            int infer = 1;
            if (hd->rec_phase && hd->skip_cnt < (skip - 1)) {
                infer = 0;
                hd->skip_cnt++;
            } else {
                hd->skip_cnt = 0;
            }
            if (infer) {
#if SILERO_STUB
                hd->last_prob = 0.0f;   /* stub: 不推理, 概率恒 0 */
#else
                hd->last_prob = sv_process(&hd->sv, hd->acc);
#endif
                hd->windows++;
                if (hd->last_prob >= WQ_SILERO_THRESHOLD) {
                    hit = 1;
                    hd->rec_phase = 1;          /* 命中=录音将启动, 进降载态 */
                }
            }
            /* v7 采集档: 逐窗特征行(p/RMS/底噪/档位), 状态变化或 32 窗心跳一条,
               供 PC 端负样本特征工程(吹气/摩擦过滤规则标定)。正常使用可忽略 */
            {
                static int s_feat_cnt;
                static int s_feat_hit = -1;
                int now_hit = (hd->last_prob >= WQ_SILERO_THRESHOLD);
                if (now_hit != s_feat_hit || (s_feat_cnt++ & 31) == 0) {
                    DBGLOG_INFO(0, "[SVf] p=%d rms=%d nf=%d skip=%d\n",
                                (int)(hd->last_prob * 10000),
                                (int)(rms * 100000), (int)(hd->noise_floor * 100000), skip);
                    s_feat_hit = now_hit;
                }
            }
            hd->acc_n = 0;      /* 无重叠: 下一窗从零攒 */
        }
    }
    /* 契约: 上层 sw_vad.c 的迟滞按"连续调用"计数(continue_hit_cnt=10),
       要求每次调用都回报当前命中态; 推理仍按 512 采样窗执行, 此处把最近
       一窗的判定平摊到两次窗完成之间的每一帧返回。仅窗完成帧返回会导致
       连续 1 最多 2 次(320 帧/512 窗), cb(1) 永不触发 → 录音无法启动 */
    if (hd->windows > 0 && hit == 0 && hd->last_prob >= WQ_SILERO_THRESHOLD) {
        hit = 1;
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
