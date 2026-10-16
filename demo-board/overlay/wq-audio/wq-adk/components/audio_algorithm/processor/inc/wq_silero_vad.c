/*
 * wq_silero_vad.c —— 物奇 libwq_sw_vad 五函数接口的 Silero VAD 实现
 * 详见 wq_silero_vad.h 头注释。
 */
#include <string.h>
#include "wq_silero_vad.h"
#include "silero_vad.h"
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
 * REC_SKIP=6: Silero 均摊 3.2ms/帧 + opus ~5ms + 管线 ≈ 11ms < 20ms 预算, 留 45% 余量 */
#define WQ_SILERO_REC_SKIP 6

/* demo 板 DFS 降频修复: 厂商 demo 电源管理会把 dcore 降到 16/32MHz,
   Silero 需 192M 才能实时(31ms/窗)。init 时申请 192M 并持有(不释放);
   量产如需省电, 可改为录音会话期间持有 */
static dfs_object_t *s_dfs_obj;

static void sv_dfs_hold(void)
{
    if (s_dfs_obj) return;
    s_dfs_obj = dfs_request(WQ_FREQ_192M, 0);
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
            /* 录音态降载: 命中后每 REC_SKIP 窗才推理 1 窗, 跳过的窗沿用上一窗判定。
               不设退出条件(v5 教训: 句间停顿>2s 退出降载→录音中途回全速→超支)。
               降载随 hd 生命周期结束——下一候选窗是新实例, 自动恢复全速保首响 */
            int infer = 1;
            if (hd->rec_phase && hd->skip_cnt < (WQ_SILERO_REC_SKIP - 1)) {
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
