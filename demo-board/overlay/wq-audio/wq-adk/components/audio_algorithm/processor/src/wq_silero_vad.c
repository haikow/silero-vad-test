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
#define WQ_SILERO_NF_UP 0.02f    /* 噪声底 EMA 上升系数(v8: 0.05→0.02, 说话首窗勿顶高底噪) */
#define WQ_SILERO_NF_DN 0.005f   /* 下降系数(慢降, 防语音把底噪顶高) */
#define WQ_SILERO_NF_RATIO 2.5f  /* 活跃判定: RMS > 2.5×噪声底 */

/* ---- v8 双门限 + 瞬态否决(2026-10-19, 板上 [SVf] 数据驱动) ----
 * 问题1(轻声延迟 2.5-3.3s): p 爬升后在 0.5 单阈值附近震荡(0.55/0.49/0.81/0.48),
 *   每次跌破 0.5 厂商"连续 10 帧确认"清零重凑 → 迟迟凑不满。
 *   修: 双门限——进入用 0.5(误触发不升), 进入后维持用 0.35(p 已确立是人声,
 *   音节间隙/尾音的轻微回落不清零迟滞)。
 * 问题2(大力吹气触发): rms 脉冲骤降指纹(81021→412 降197×, 37148→236 降157×;
 *   正常说话 rms 平稳, 峰后降幅 <5×)。修: rms>8000 的峰窗后 2 窗内骤降 20×
 *   → 判气流/碰撞瞬态, 否决随后 4 窗(128ms)的命中。大声喊 rms 持续高无骤降,
 *   不误杀; 万一误杀也只延迟 128ms 照常触发。 */
#define WQ_SILERO_TH_ENTER 0.45f
#define WQ_SILERO_TH_HOLD  0.35f
#define WQ_SILERO_PK_RMS   8000    /* 瞬态判定: rms 峰阈值(万分之一单位) */
#define WQ_SILERO_PK_DROP  20      /* 峰后 2 窗内降到 峰/20 以下 = 骤降 */
#define WQ_SILERO_VETO_WIN 4       /* 瞬态后否决窗数(128ms) */
#define WQ_SILERO_QUIET_N 12        /* v10 能量停录: 连续安静窗数(~380ms)后无视p尾巴 */
#define WQ_SILERO_CAP_WIN 3750      /* v11 录音硬顶: 连续无安静间隙窗数(3750×32ms=120s) */

/* demo 板 DFS 降频修复: 厂商 demo 电源管理会把 dcore 降到 16/32MHz,
   Silero 需 192M 才能实时(31ms/窗)。
   v11b 回退永久持有(v7 会话级被 10-20 14:50 日志证伪): 候选窗每次超时
   都销毁/重建流, exit→释放 192M→重建→再申请的间隙里 dcore 掉到 64M
   甚至 16M(日志 5×"set to 32/32/64"+1×"16/16/16"), 若恰逢录音/媒体
   流并发, Silero 每窗 93-372ms → 124 条 run too long → WDT2 崩在
   xa_nn_matmul_f32。流生命周期 ≠ 会话生命周期, 释放粒度太细。
   永久持有的功耗代价(空闲 ~百 mA 级)先接受, 灵敏度/稳定性优先;
   功耗版待做"延迟释放防抖"或与厂商对齐流生命周期后再上 */
static dfs_object_t *s_dfs_obj;

static void sv_dfs_hold(void)
{
    if (s_dfs_obj) return;
    s_dfs_obj = dfs_request(WQ_FREQ_192M, 0);
}

void wq_sw_vad_exit(void)
{
    /* v11b: 不再释放——见上方注释。保留空函数(swid sw_vad.c 调用点不动) */
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

            /* v8: 瞬态(吹气/碰撞)否决——rms 峰后骤降指纹。峰窗记峰, 后 2 窗内
               掉到峰/20 以下 → 否决随后 4 窗(气流脉冲把 p 顶出的尖峰不算数) */
            int rms_i = (int)(rms * 100000);
            if (rms_i > WQ_SILERO_PK_RMS) {
                hd->pk_rms = rms_i;
                hd->pk_age = 0;
            } else if (hd->pk_rms > WQ_SILERO_PK_RMS) {
                hd->pk_age++;
                if (hd->pk_age <= 2 && rms_i * WQ_SILERO_PK_DROP < hd->pk_rms) {
                    if (hd->veto_cnt < WQ_SILERO_VETO_WIN)
                        hd->veto_cnt = WQ_SILERO_VETO_WIN;
                    hd->pk_rms = 0;         /* 峰已消费 */
                }
                if (hd->pk_age > 2) hd->pk_rms = 0;
            }
            if (hd->veto_cnt > 0) hd->veto_cnt--;
            int veto = hd->veto_cnt > 0;

            /* v10 能量门控停录(v11 修正执行顺序): 修"没人声也一直录"。
               实测(10-19 18:20 日志) 25% 窗口 rms<500(能量已静) 但 p≥0.35
               (Silero 衰减尾巴/环境持续声把保持线托住) → 厂商 35 帧停录凑不齐。
               v10 教训(10-20 14:50 日志): 归零写在 infer 前, 被 sv_process
               输出覆盖 → 门控从未生效 → 音乐场景录音停不了 → 音乐解码+opus+
               Silero 三重负载 24s 持续超支 → WDT2 崩在 xa_nn_matmul_f32。
               v11: 计数在 infer 前累计(每窗都算), 归零移到 infer 之后覆盖。 */
            if (hd->rec_phase) {
                float nf_i = hd->noise_floor * 100000.0f;
                float quiet_thr = (nf_i * 1.6f < 600.0f) ? (nf_i * 1.6f) : 600.0f;
                if ((float)rms_i < quiet_thr) {
                    if (hd->quiet_cnt < 255) hd->quiet_cnt++;
                } else {
                    hd->quiet_cnt = 0;
                }
            } else {
                hd->quiet_cnt = 0;
            }
            int quiet_kill = hd->rec_phase && hd->quiet_cnt >= WQ_SILERO_QUIET_N;

            /* v11 录音硬顶保底: 音乐场景 rms 高(能量门控不触发)+Silero 判音乐
               p 0.5-0.9(停录凑不齐)→录音无限拖长→音乐解码+opus+Silero 三重
               负载持续超支→WDT2(10-20 14:50 日志 rec6, 124 条 run too long
               后崩在 xa_nn_matmul_f32)。任何连续 120s 无一次 380ms 安静间隙
               的"语音"(只有音乐/持续声做得到)强制判停, 并在下次安静前不再
               重新命中——同时阻断音乐反复触发录音的循环。正常人声 120s 内
               必有句间安静, 最坏代价是把超长独白切成两段 */
            if (!quiet_kill) {
                if (hd->nosil_cnt < 60000) hd->nosil_cnt++;
            } else {
                hd->nosil_cnt = 0;
            }
            int cap_kill = hd->rec_phase && hd->nosil_cnt > WQ_SILERO_CAP_WIN;

            /* v8 双门限: 进入 0.5(误触发不升); rec_phase 后维持 0.35(音节间隙
               轻微回落不清零厂商迟滞——轻声贴线震荡 3.3s 的根因) */
            float thr = hd->rec_phase ? WQ_SILERO_TH_HOLD : WQ_SILERO_TH_ENTER;

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
                /* v11: 能量门控在推理之后覆盖(v10 顺序 bug 修正)——
                   连续安静窗数达标后, 无论模型尾巴输出什么都判静音;
                   120s 无安静间隙硬顶(音乐场景)同理强制判停 */
                if (quiet_kill || cap_kill)
                    hd->last_prob = 0.0f;
                hd->windows++;
                if (!veto && hd->last_prob >= thr) {
                    hit = 1;
                    hd->rec_phase = 1;          /* 命中=录音将启动, 进降载态 */
                }
            }
            /* v7 采集档: 逐窗特征行(p/RMS/底噪/档位), 状态变化或 32 窗心跳一条,
               供 PC 端负样本特征工程(吹气/摩擦过滤规则标定)。正常使用可忽略 */
            {
                static int s_feat_cnt;
                static int s_feat_hit = -1;
                int now_hit = (hd->last_prob >= thr);
                if (now_hit != s_feat_hit || (s_feat_cnt++ & 31) == 0) {
                    DBGLOG_INFO(0, "[SVf] p=%d rms=%d nf=%d skip=%d v=%d\n",
                                (int)(hd->last_prob * 10000),
                                rms_i, (int)(hd->noise_floor * 100000), skip, veto);
                    s_feat_hit = now_hit;
                }
            }
            hd->acc_n = 0;      /* 无重叠: 下一窗从零攒 */
        }
    }
    /* 契约: 上层 sw_vad.c 的迟滞按"连续调用"计数(continue_hit_cnt=10),
       要求每次调用都回报当前命中态; 推理仍按 512 采样窗执行, 此处把最近
       一窗的判定平摊到两次窗完成之间的每一帧返回。仅窗完成帧返回会导致
       连续 1 最多 2 次(320 帧/512 窗), cb(1) 永不触发 → 录音无法启动。
       v8: 平摊判定同用双门限 + 瞬态否决 */
    {
        float thr = hd->rec_phase ? WQ_SILERO_TH_HOLD : WQ_SILERO_TH_ENTER;
        if (hd->windows > 0 && hit == 0 && hd->veto_cnt == 0 &&
            hd->last_prob >= thr) {
            hit = 1;
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
