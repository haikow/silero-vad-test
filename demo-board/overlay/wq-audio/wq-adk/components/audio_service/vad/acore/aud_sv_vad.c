/****************************************************************************

Copyright(c) 2024 by WuQi Technologies. ALL RIGHTS RESERVED.

This Information is proprietary to WuQi Technologies and MAY NOT
be copied by any method or incorporated into another program without
the express written consent of WuQi. This Information or any portion
thereof remains the property of WuQi. The Information contained herein
is believed to be accurate and WuQi assumes no responsibility or
liability for its use in any way and conveys no license or title under
any patent or copyright and makes no representation or warranty that this
Information is free from patent or copyright infringement.

****************************************************************************/
/*
 * INCLUDE FILES
 ****************************************************************************
 */
#include "types.h"
#include "string.h"
#include "assert.h"
#include "cpu.h"
#include "FreeRTOS.h"
#include "timers.h"

#include "os_timer.h"
#include "as_ipc_tr.h"
#include "aud_sv_api.h"
#include "aud_sv_audio_config_define.h"
#include "storage_controller.h"
#include "aud_kv_def.h"
#include "ro_cfg.h"
#include "aud_sv_stream.h"
#include "aud_res_mic.h"
#include "aud_sv_vad.h"
#include "as_sh_info.h"
#include "m_datapath.h"
#include "multicore_data_pack.h"
#include "anc_mgr.h"
#include "cores_task_cfg.h"
#include "aud_sv_task.h"
#include "aud_sv_rpc.h"
#include "aud_sv_sync.h"
#include "feat_res_mgnt.h"
#include "wq_heap.h"
#include "wq_rtc.h"
#include "webrtc_core.h"
#include "aud_sv_stream_hold.h"
#include "aud_sv_stream_mgr.h"
#include "dcore_streams.h"

#if defined(CONFIG_AUDIO_VAD_ENABLE)

#if (CONFIG_RING_ALLOCATE_CFG != 2 && CONFIG_RING_ALLOCATE_CFG != 3)
#error "It is necessary to set RING_ALLOCATE_CFG to 2 or 3."
#endif

#if (CONFIG_PIPELINE_STREAM_MAX_NUM <= STREAM_ID_RECORD_BASE)
#error "It is necessary to set PIPELINE_STREAM_MAX_NUM greater than STREAM_ID_RECORD_BASE"
#endif

#if (CONFIG_PIPELINE_STREAM_MAX_NUM <= STREAM_ID_VAD)
#error "It is necessary to set PIPELINE_STREAM_MAX_NUM greater than STREAM_ID_VAD"
#endif

/*
 * TYPE DEFINITIONS
 ****************************************************************************
 */
#define AUD_SV_GVAD_TIMEOUT_MS    500
#if defined(CONFIG_APP_VAD_GVAD_MODE)
#define AUD_SV_GVAD_MODE CONFIG_APP_VAD_GVAD_MODE
#else
#define AUD_SV_GVAD_MODE 0   // 0-3, 3 is very aggressive mode
#endif
#if defined(CONFIG_APP_VAD_FRONT_CONFIRM_MS)
#define AUD_SV_VAD_FRONT_CONFIRM_MS CONFIG_APP_VAD_FRONT_CONFIRM_MS
#else
#define AUD_SV_VAD_FRONT_CONFIRM_MS 0
#endif
#if defined(CONFIG_APP_VAD_HIT_CONFIRM_MS)
#define AUD_SV_VAD_HIT_CONFIRM_MS CONFIG_APP_VAD_HIT_CONFIRM_MS
#else
#define AUD_SV_VAD_HIT_CONFIRM_MS 120
#endif
#if AUD_SV_GVAD_MODE < 0 || AUD_SV_GVAD_MODE > 3
#error "GVAD mode must be between 0 and 3"
#endif
#if AUD_SV_VAD_FRONT_CONFIRM_MS < 0 || AUD_SV_VAD_FRONT_CONFIRM_MS > 500
#error "Front VAD confirmation must be between 0 and 500 ms"
#endif
#define AUD_SV_VAD_FRONT_SAMPLES_PER_MS 16U
#define AUD_SV_VAD_FRONT_GAP_MS 50U
#define AUD_SV_KWS_TIMEOUT_MS     5000
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
#ifndef CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS
#define CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS 2000
#endif
#ifndef CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS
#define CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS 500
#endif
#if CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS < 1000 || \
    CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS > AUD_SV_KWS_TIMEOUT_MS || \
    CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS < 0 || \
    CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS > 1000
#error "Invalid VG4 candidate timeout/cooldown"
#endif
#endif
#define AUD_SV_VAD_RETRY_MS       30
#define AUD_SV_VAD_RETRY_NONE     0
#define AUD_SV_VAD_RETRY_NORMAL   1
#define AUD_SV_VAD_RETRY_NO_WAIT  2
#define AUD_SV_VAD_TRIGGER_DLY_MS 800
#define AUD_SV_TVAD_BUF_LEN       11
#if defined(CONFIG_BUILD_SERIES_BBB)
#if CONFIG_AUD_RES_MIC_SAMPLE_RATE == 48000
#define AUD_SV_VAD_RING_SAVE_PKT_NUM 15
#else
#define AUD_SV_VAD_RING_SAVE_PKT_NUM 20
#endif
#define AUD_SV_GVAD_BUF_LEN 6
#else
#define AUD_SV_VAD_RING_SAVE_PKT_NUM 11
#endif

/*
 * TYPES
 ****************************************************************************
 */
// vad state
typedef enum {
    VAD_ST_IDLE,
    VAD_ST_OPENING,   // opening mic
    VAD_ST_CLOSING,   // vad stream stop
    VAD_ST_PENDING_CLOSING,   // mic closing before fully pending
    VAD_ST_OPEN,      // mic open done
    VAD_ST_START,     // vad stream start
    VAD_ST_PENDING,   // fully closed and pending by voice

    VAD_ST_MAX
} vad_state_t;

// vad dp task event
typedef enum {
    VAD_EVT_OPEN,           // open mic and tvad/gvad
    VAD_EVT_OPEN_DONE,      // start mic
    VAD_EVT_MIC_PACKET,     // mic packet
    VAD_EVT_TVAD_DETECT,    // tvad detected
    VAD_EVT_GVAD_DETECT,    // acore gvad detected
    VAD_EVT_GVAD_TIMEOUT,   // acore gvad detect timeout
    VAD_EVT_CREATE,         // start dcore vad stream
    VAD_EVT_VOICE_HIT,      // dcore gvad hit
    VAD_EVT_KWS_HIT,        // dcore kws hit
    VAD_EVT_KWS_TIMEOUT,    // dcore kws timeout
    VAD_EVT_DESTROY,        // stop dcore vad stream
    VAD_EVT_CLOSE,          // stop mic and tvad/gvad
    VAD_EVT_CLOSE_DONE,     // close mic and tvad/gvad
    VAD_EVT_RESTART,        // restart dcore vad proc
    VAD_EVT_PEND,           // pend by voice
    VAD_EVT_RESUME,         // resume by voice
    VAD_EVT_RECORD_MONITOR, // keep GVAD active while KWS is paused for recording
    VAD_EVT_RECORD_MONITOR_ACK, // DCORE accepted/rejected the KWS update
    VAD_EVT_RECORD_MONITOR_RETRY, // retry IPC/guard work without changing desired state
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    VAD_EVT_CANDIDATE_TIMEOUT,
    VAD_EVT_COOLDOWN_DONE,
#endif

    VAD_EVT_MAX
} vad_evt_t;

typedef struct {
    vad_evt_t evt;

    uint32_t seq;
    uint32_t ipc_seq;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    uint32_t gvad_window_seq;
#endif
    int32_t result;
    uint8_t enable;
    uint8_t hit_status;
    const uint8_t *data;
    uint32_t data_len;
} vad_msg_t;

typedef struct {
    //anc share info pointer
    volatile as_sh_info_anc_stream_info_t *p_anc;
    volatile as_sh_info_mic_stream_info_t *p_mic;

    wq_aud_vad_voice_hit_cb voice_cb;
    wq_aud_vad_kws_hit_cb kws_cb;
    wq_aud_vad_close_done_cb close_done_cb;
    char *gvad_hdl;
    int gvad_ret;
    uint32_t gvad_front_samples;
    uint32_t gvad_front_last_ms;
    timer_id_t kws_timer;
    timer_id_t retry_timer;
#if CONFIG_AUD_RES_MIC_SAMPLE_RATE != 16000
    int16_t *mic_16k_buff;
#endif
#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
    int16_t *pcm_data_buffer;
#endif
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    timer_id_t gvad_timer;
    uint32_t gvad_window_seq;
    uint32_t retry_gvad_timeout_window_seq;
    TickType_t gvad_timeout_deadline_tick;
    uint32_t tvad_buffer[AUD_SV_TVAD_BUF_LEN];
#if defined(CONFIG_BUILD_SERIES_BBB)
    uint32_t gvad_buffer[AUD_SV_GVAD_BUF_LEN];
#endif
#endif
    vad_state_t state;
    stream_cmd_reason_e manager_close_reason;
    uint32_t evt_seq;
    uint32_t latest_evt_seq[VAD_EVT_MAX];
    uint32_t retry_evt_seq[VAD_EVT_MAX];
    uint32_t public_intent_seq;
    uint32_t stream_intent_seq;
    uint32_t record_monitor_ipc_seq;
    uint32_t record_monitor_pending_seq;
    uint32_t kws_timeout_seq;
    TickType_t kws_timeout_deadline_tick;
    vad_evt_t stream_intent_evt;
    volatile uint32_t retry_evt_mask;
    int8_t cur_save_pkt_num;
    uint8_t retry_voice_hit_status;
    uint8_t retry_kws_hit_status;
    uint8_t retry_record_monitor_enable;
    uint8_t retry_record_monitor_ack_enable;
    uint32_t retry_record_monitor_ack_ipc_seq;
    int32_t retry_record_monitor_ack_result;
    volatile uint8_t gvad_enable;
    uint8_t stream_start : 1;
    uint8_t kws_trigger  : 1;
    uint8_t music_start  : 1;
    uint8_t pend_start   : 1;
    uint8_t pend_stop    : 1;
    uint8_t gvad_detect_pending : 1;
    uint8_t pend_by_stream : 1;
    uint8_t stream_guard_held : 1;
    uint8_t stream_mic_owned : 1;
    uint8_t manager_close_pending : 1;
    uint8_t retry_timer_active : 1;
    uint8_t public_open_desired : 1;
    uint8_t record_monitor_desired : 1;
    uint8_t record_monitor_applied : 1;
    uint8_t record_monitor_inflight : 1;
    uint8_t record_monitor_pending_enable : 1;
    uint8_t record_monitor_exit_pending : 1;
    uint8_t record_monitor_unhit_notify_pending : 1;
    uint8_t voice_active : 1;
    uint8_t kws_timer_armed : 1;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    uint32_t candidate_epoch;
    uint32_t retry_voice_epoch;
    uint32_t record_monitor_request_seq;
    vad_evt_t kws_timer_event;
    uint8_t candidate_origin : 1;
    uint8_t candidate_hit_seen : 1;
    uint8_t candidate_cleanup : 1;
    uint8_t cooldown_active : 1;
    uint8_t record_monitor_request_pending : 1;
    uint8_t record_monitor_request_enable : 1;
#endif
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    uint8_t gvad_timer_armed : 1;
#endif
} aud_sv_vad_env_t;

static WQ_RET aud_sv_vad_read_ring_pkt(uint16_t data_len, bool force_read);
static void aud_sv_vad_data_mic_hook(const uint8_t *p_data, uint32_t length);
static void aud_sv_vad_evt_callback(AUD_SV_EVT_E evt, const aud_sv_evt_param_t *p_param);
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
static void aud_sv_vad_tvad_open(void);
static void aud_sv_vad_tvad_start(void);
static void aud_sv_vad_tvad_mask(void);
static void aud_sv_vad_tvad_unmask(void);
static void aud_sv_vad_tvad_rearm(const char *reason);
static void aud_sv_vad_tvad_close(void);
#endif
static void aud_sv_vad_gvad_enable(bool enable);
static void aud_sv_vad_retry_schedule(vad_evt_t evt);
static void aud_sv_vad_record_monitor_kick(void);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
static bool aud_sv_vad_record_monitor_holds_kws_off(void);
static WQ_RET aud_sv_vad_record_monitor_notify_unhit(void);
static void aud_sv_vad_notice_app_cb(uint8_t result);
#endif

/*
 * VARIABLES
 ****************************************************************************
 */
static aud_sv_vad_env_t vad_env = {0};

LOG_STREAM_SEC evt_str[VAD_EVT_MAX][13] = {
    "open",           // VAD_EVT_OPEN
    "open_done",      // VAD_EVT_OPEN_DONE
    "mic_packet",     // VAD_EVT_MIC_PACKET
    "tvad_detect",    // VAD_EVT_TVAD_DETECT
    "gvad_detect",    // VAD_EVT_GVAD_DETECT
    "gvad_timeout",   // VAD_EVT_GVAD_TIMEOUT
    "create",         // VAD_EVT_CREATE
    "voice_hit",      // VAD_EVT_VOICE_HIT
    "kws_hit",        // VAD_EVT_KWS_HIT
    "kws_timeout",    // VAD_EVT_KWS_TIMEOUT
    "destroy",        // VAD_EVT_DESTROY
    "close",          // VAD_EVT_CLOSE
    "close_done",     // VAD_EVT_CLOSE_DONE
    "restart",        // VAD_EVT_RESTART
    "pend",           // VAD_EVT_PEND
    "resume",         // VAD_EVT_RESUME
    "record_mon",     // VAD_EVT_RECORD_MONITOR
    "record_ack",     // VAD_EVT_RECORD_MONITOR_ACK
    "record_retry",   // VAD_EVT_RECORD_MONITOR_RETRY
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    "cand_timeout",   // VAD_EVT_CANDIDATE_TIMEOUT
    "cool_done",      // VAD_EVT_COOLDOWN_DONE
#endif
};

LOG_STREAM_SEC st_str[VAD_ST_MAX][8] = {
    "idle",      // VAD_ST_IDLE
    "opening",   // VAD_ST_OPENING
    "closing",   // VAD_ST_CLOSING
    "pendcls",   // VAD_ST_PENDING_CLOSING
    "open",      // VAD_ST_OPEN
    "start",     // VAD_ST_START
    "pending",   // VAD_ST_PENDING
};

/*
 * FUNCTIONS
 ****************************************************************************
 */
/* Only candidate decision state is reset; PCM/ring ownership is unchanged. */
static void aud_sv_vad_front_reset(void)
{
    if (vad_env.gvad_front_samples != 0U) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG3 front reset samples:%d\n",
                                vad_env.gvad_front_samples);
    }
    vad_env.gvad_front_samples = 0;
    vad_env.gvad_front_last_ms = 0;
}

/* Count audio actually passed to the 16 kHz GMM, not callbacks or wall time.
 * GMM hangover may remain positive after a transient; this is not a separate
 * speech classifier. No per-frame logging or additional wakeup is needed. */
static bool aud_sv_vad_front_confirm(int result, uint32_t samples, uint32_t now_ms)
{
#if AUD_SV_VAD_FRONT_CONFIRM_MS > 0
    const uint32_t target = AUD_SV_VAD_FRONT_CONFIRM_MS * AUD_SV_VAD_FRONT_SAMPLES_PER_MS;
    uint32_t previous;

    if (result <= 0 || samples == 0U) {
        vad_env.gvad_front_samples = 0;
        vad_env.gvad_front_last_ms = 0;
        return false;
    }
    if (vad_env.gvad_front_samples != 0U
        && (uint32_t)(now_ms - vad_env.gvad_front_last_ms) > AUD_SV_VAD_FRONT_GAP_MS) {
        vad_env.gvad_front_samples = 0;
    }
    vad_env.gvad_front_last_ms = now_ms;
    previous = vad_env.gvad_front_samples;
    if (samples >= target - previous) {
        vad_env.gvad_front_samples = target;
    } else {
        vad_env.gvad_front_samples += samples;
    }
    if (previous < target && vad_env.gvad_front_samples == target) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG3 front ready samples:%d confirm:%dms\n",
                                target, AUD_SV_VAD_FRONT_CONFIRM_MS);
    }
    return vad_env.gvad_front_samples == target;
#else
    UNUSED(samples);
    UNUSED(now_ms);
    return result > 0;
#endif
}

static WQ_RET aud_sv_vad_status_update_seq(vad_cmd_t cmd, uint8_t param, uint32_t seq,
                                           int32_t result)
{
    aud_sv_ipc_vad_param_msg_t msg = {0};

    msg.ipc_req_hdr.ipc_msg_id = AS_IPC_TR_MSG_ID_VAD_STATUS;
    msg.cmd = cmd;
    msg.seq = seq;
    msg.result = result;
    msg.param = param;
    return as_ipc_tr_send_msg(WQ_CORES_DCORE, (ipc_req_t *)&msg,
                              sizeof(aud_sv_ipc_vad_param_msg_t));
}

static WQ_RET aud_sv_vad_status_update(vad_cmd_t cmd, uint8_t param)
{
    return aud_sv_vad_status_update_seq(cmd, param, 0, WQ_RET_OK);
}

static const vad_evt_t retry_evt_priority[] = {
    VAD_EVT_CLOSE_DONE, VAD_EVT_OPEN_DONE,    VAD_EVT_CLOSE,       VAD_EVT_PEND,
    VAD_EVT_RESUME,     VAD_EVT_OPEN,         VAD_EVT_DESTROY,     VAD_EVT_CREATE,
    VAD_EVT_KWS_TIMEOUT, VAD_EVT_GVAD_TIMEOUT, VAD_EVT_TVAD_DETECT, VAD_EVT_VOICE_HIT,
    VAD_EVT_KWS_HIT,    VAD_EVT_RESTART,      VAD_EVT_RECORD_MONITOR_ACK,
    VAD_EVT_RECORD_MONITOR_RETRY, VAD_EVT_RECORD_MONITOR,
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    VAD_EVT_CANDIDATE_TIMEOUT, VAD_EVT_COOLDOWN_DONE,
#endif
};

static uint32_t aud_sv_vad_next_seq_locked(void)
{
    vad_env.evt_seq++;
    if (!vad_env.evt_seq) {
        vad_env.evt_seq++;
    }

    return vad_env.evt_seq;
}

#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
/* Caller owns the critical section; no timer commands or logging here. */
static void aud_sv_vad_candidate_cancel_locked(void)
{
    vad_env.candidate_epoch = aud_sv_vad_next_seq_locked();
    vad_env.latest_evt_seq[VAD_EVT_CANDIDATE_TIMEOUT] = vad_env.candidate_epoch;
    vad_env.latest_evt_seq[VAD_EVT_COOLDOWN_DONE] = vad_env.candidate_epoch;
    vad_env.retry_evt_mask &= ~(BIT(VAD_EVT_CANDIDATE_TIMEOUT) | BIT(VAD_EVT_COOLDOWN_DONE));
    if (vad_env.kws_timer_event != VAD_EVT_KWS_TIMEOUT) {
        vad_env.kws_timer_armed = 0;
    }
    vad_env.candidate_origin = 0;
    vad_env.candidate_hit_seen = 0;
    if (!vad_env.stream_start && !vad_env.stream_guard_held && !vad_env.stream_mic_owned) {
        vad_env.candidate_cleanup = 0;
    }
    vad_env.cooldown_active = 0;
}

static bool aud_sv_vad_front_should_process(void)
{
    return vad_env.gvad_enable && vad_env.gvad_hdl && vad_env.public_open_desired
        && vad_env.state == VAD_ST_OPEN && !vad_env.stream_start
        && vad_env.stream_intent_evt != VAD_EVT_PEND
        && !vad_env.candidate_cleanup && !vad_env.cooldown_active;
}
#endif

static bool aud_sv_vad_event_is_current_locked(const vad_msg_t *p_evt)
{
    if (!p_evt->seq) {
        return true;
    }
    if (vad_env.latest_evt_seq[p_evt->evt] != p_evt->seq) {
        return false;
    }

    switch (p_evt->evt) {
    case VAD_EVT_OPEN:
        return vad_env.public_open_desired
            && vad_env.public_intent_seq == p_evt->seq;
    case VAD_EVT_CLOSE:
        return !vad_env.public_open_desired
            && vad_env.public_intent_seq == p_evt->seq;
    case VAD_EVT_PEND:
    case VAD_EVT_RESUME:
        return vad_env.stream_intent_evt == p_evt->evt
            && vad_env.stream_intent_seq == p_evt->seq;
    default:
        return true;
    }
}

static bool aud_sv_vad_event_is_current(const vad_msg_t *p_evt)
{
    bool current;

    cpu_critical_enter();
    current = aud_sv_vad_event_is_current_locked(p_evt);
    cpu_critical_exit();

    return current;
}

static uint32_t aud_sv_vad_event_begin(vad_evt_t evt)
{
    uint32_t seq;

    cpu_critical_enter();
    seq = aud_sv_vad_next_seq_locked();
    vad_env.latest_evt_seq[evt] = seq;
    vad_env.retry_evt_mask &= ~BIT(evt);
    switch (evt) {
    case VAD_EVT_OPEN:
        vad_env.public_open_desired = 1;
        vad_env.public_intent_seq = seq;
        vad_env.retry_evt_mask &= ~BIT(VAD_EVT_CLOSE);
        break;
    case VAD_EVT_CLOSE:
        vad_env.public_open_desired = 0;
        vad_env.public_intent_seq = seq;
        vad_env.retry_evt_mask &= ~BIT(VAD_EVT_OPEN);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        aud_sv_vad_candidate_cancel_locked();
#endif
        break;
    case VAD_EVT_PEND:
    case VAD_EVT_RESUME:
        vad_env.stream_intent_evt = evt;
        vad_env.stream_intent_seq = seq;
        vad_env.retry_evt_mask &= ~(BIT(VAD_EVT_PEND) | BIT(VAD_EVT_RESUME));
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        if (evt == VAD_EVT_PEND) {
            aud_sv_vad_candidate_cancel_locked();
        }
#endif
        break;
    default:
        break;
    }
    cpu_critical_exit();

    return seq;
}

static uint32_t aud_sv_vad_retry_seq_get(vad_evt_t evt)
{
    uint32_t seq = 0;

    cpu_critical_enter();
    switch (evt) {
    case VAD_EVT_OPEN:
        if (vad_env.public_open_desired) {
            seq = vad_env.public_intent_seq;
        }
        break;
    case VAD_EVT_CLOSE:
        if (!vad_env.public_open_desired) {
            seq = vad_env.public_intent_seq;
        }
        break;
    case VAD_EVT_PEND:
    case VAD_EVT_RESUME:
        if (vad_env.stream_intent_evt == evt) {
            seq = vad_env.stream_intent_seq;
        }
        break;
    default:
        seq = aud_sv_vad_next_seq_locked();
        vad_env.latest_evt_seq[evt] = seq;
        vad_env.retry_evt_mask &= ~BIT(evt);
        break;
    }
    cpu_critical_exit();

    return seq;
}

static bool aud_sv_vad_retry_pending_set(const vad_msg_t *p_evt)
{
    bool set = false;

    cpu_critical_enter();
    if (aud_sv_vad_event_is_current_locked(p_evt)) {
        vad_env.retry_evt_seq[p_evt->evt] = p_evt->seq;
        if (p_evt->evt == VAD_EVT_VOICE_HIT) {
            vad_env.retry_voice_hit_status = p_evt->hit_status;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
            vad_env.retry_voice_epoch = p_evt->ipc_seq;
#endif
        } else if (p_evt->evt == VAD_EVT_KWS_HIT) {
            vad_env.retry_kws_hit_status = p_evt->hit_status;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
        } else if (p_evt->evt == VAD_EVT_GVAD_TIMEOUT) {
            vad_env.retry_gvad_timeout_window_seq = p_evt->gvad_window_seq;
#endif
        } else if (p_evt->evt == VAD_EVT_RECORD_MONITOR) {
            vad_env.retry_record_monitor_enable = p_evt->enable;
        } else if (p_evt->evt == VAD_EVT_RECORD_MONITOR_ACK) {
            vad_env.retry_record_monitor_ack_enable = p_evt->enable;
            vad_env.retry_record_monitor_ack_ipc_seq = p_evt->ipc_seq;
            vad_env.retry_record_monitor_ack_result = p_evt->result;
        }
        vad_env.retry_evt_mask |= BIT(p_evt->evt);
        set = true;
    }
    cpu_critical_exit();

    return set;
}

static bool aud_sv_vad_retry_pending_get(vad_msg_t *p_evt)
{
    uint32_t mask;

    cpu_critical_enter();
    mask = vad_env.retry_evt_mask;
    for (uint8_t i = 0; i < sizeof(retry_evt_priority) / sizeof(retry_evt_priority[0]); i++) {
        if (mask & BIT(retry_evt_priority[i])) {
            p_evt->evt = retry_evt_priority[i];
            p_evt->seq = vad_env.retry_evt_seq[p_evt->evt];
            if (p_evt->evt == VAD_EVT_VOICE_HIT) {
                p_evt->hit_status = vad_env.retry_voice_hit_status;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
                p_evt->ipc_seq = vad_env.retry_voice_epoch;
#endif
            } else if (p_evt->evt == VAD_EVT_KWS_HIT) {
                p_evt->hit_status = vad_env.retry_kws_hit_status;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
            } else if (p_evt->evt == VAD_EVT_GVAD_TIMEOUT) {
                p_evt->gvad_window_seq = vad_env.retry_gvad_timeout_window_seq;
#endif
            } else if (p_evt->evt == VAD_EVT_RECORD_MONITOR) {
                p_evt->enable = vad_env.retry_record_monitor_enable;
            } else if (p_evt->evt == VAD_EVT_RECORD_MONITOR_ACK) {
                p_evt->enable = vad_env.retry_record_monitor_ack_enable;
                p_evt->ipc_seq = vad_env.retry_record_monitor_ack_ipc_seq;
                p_evt->result = vad_env.retry_record_monitor_ack_result;
            }
            cpu_critical_exit();
            return true;
        }
    }
    cpu_critical_exit();

    return false;
}

static void aud_sv_vad_retry_sent_clear(const vad_msg_t *p_evt)
{
    cpu_critical_enter();
    if ((vad_env.retry_evt_mask & BIT(p_evt->evt))
        && vad_env.retry_evt_seq[p_evt->evt] == p_evt->seq) {
        vad_env.retry_evt_mask &= ~BIT(p_evt->evt);
    }
    cpu_critical_exit();
}

static void aud_sv_vad_retry_timer_stop_if_idle(bool in_isr);
static void aud_sv_vad_retry_timer_stop_no_wait(bool in_isr);

static bool aud_sv_vad_retry_timer_start(bool in_isr, bool no_wait)
{
    bool start_timer = false;
    bool stop_timer = false;
    bool ret;

    cpu_critical_enter();
    if (vad_env.retry_evt_mask && !vad_env.retry_timer_active) {
        vad_env.retry_timer_active = 1;
        start_timer = true;
    }
    cpu_critical_exit();

    if (!start_timer) {
        return true;
    }

    if (in_isr) {
        ret = os_start_timer_from_isr(vad_env.retry_timer, AUD_SV_VAD_RETRY_MS);
    } else if (no_wait) {
        TickType_t period = AUD_SV_VAD_RETRY_MS / portTICK_PERIOD_MS;

        period = period ? period : 1;
        ret = xTimerChangePeriod((TimerHandle_t)vad_env.retry_timer, period, 0) == pdPASS;
    } else {
        ret = os_start_timer(vad_env.retry_timer, AUD_SV_VAD_RETRY_MS);
    }
    cpu_critical_enter();
    if (ret) {
        /* A matching event may have been delivered while the start command
         * was being queued. Stop again so command ordering converges. */
        vad_env.retry_timer_active = 1;
        stop_timer = !vad_env.retry_evt_mask;
    } else {
        vad_env.retry_timer_active = 0;
    }
    cpu_critical_exit();
    if (!ret) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] retry timer start fail\n");
    } else if (stop_timer) {
        if (no_wait) {
            aud_sv_vad_retry_timer_stop_no_wait(in_isr);
        } else {
            aud_sv_vad_retry_timer_stop_if_idle(in_isr);
        }
    }

    return ret;
}

static void aud_sv_vad_retry_timer_stop_if_idle(bool in_isr)
{
    bool restart_timer = false;
    bool stop_timer = false;
    bool ret;

    cpu_critical_enter();
    if (!vad_env.retry_evt_mask && vad_env.retry_timer_active) {
        stop_timer = true;
    }
    cpu_critical_exit();

    if (!stop_timer) {
        return;
    }

    ret = in_isr ? os_stop_timer_from_isr(vad_env.retry_timer)
                 : os_stop_timer(vad_env.retry_timer);
    cpu_critical_enter();
    if (ret) {
        /* Keep the active flag set until the stop command is queued. If an
         * ISR records work meanwhile, restart after that stop command. */
        restart_timer = vad_env.retry_evt_mask != 0;
        vad_env.retry_timer_active = 0;
    }
    cpu_critical_exit();
    if (!ret) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] retry timer stop fail\n");
    } else if (restart_timer) {
        (void)aud_sv_vad_retry_timer_start(in_isr, false);
    }
}

static void aud_sv_vad_retry_timer_stop_no_wait(bool in_isr)
{
    bool restart_timer = false;
    bool stop_timer = false;
    bool ret;

    cpu_critical_enter();
    if (!vad_env.retry_evt_mask && vad_env.retry_timer_active) {
        stop_timer = true;
    }
    cpu_critical_exit();

    if (!stop_timer) {
        return;
    }

    ret = in_isr ? os_stop_timer_from_isr(vad_env.retry_timer)
                 : xTimerStop((TimerHandle_t)vad_env.retry_timer, 0) == pdPASS;
    cpu_critical_enter();
    if (ret) {
        restart_timer = vad_env.retry_evt_mask != 0;
        vad_env.retry_timer_active = 0;
    }
    cpu_critical_exit();
    if (!ret) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] retry timer no-wait stop fail\n");
    } else if (restart_timer) {
        (void)aud_sv_vad_retry_timer_start(in_isr, true);
    }
}

static bool aud_sv_vad_retry_post_one(bool in_isr)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    if (!aud_sv_vad_retry_pending_get(p_dmsg)) {
        return false;
    }
    if (!aud_sv_task_send_msg(AS_MSG_VAD, &msg, in_isr, sizeof(*p_dmsg))) {
        return false;
    }

    aud_sv_vad_retry_sent_clear(p_dmsg);
    return true;
}

static bool aud_sv_vad_control_post_seq_internal(const vad_msg_t *p_evt, bool in_isr,
                                                 uint8_t retry_mode, uint32_t seq,
                                                 bool accept_deferred)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);
    bool retry_saved = false;

    memcpy(p_dmsg, p_evt, sizeof(*p_dmsg));
    p_dmsg->seq = seq ? seq : aud_sv_vad_event_begin(p_dmsg->evt);
    if (aud_sv_task_send_msg(AS_MSG_VAD, &msg, in_isr, sizeof(*p_dmsg))) {
        aud_sv_vad_retry_sent_clear(p_dmsg);
        if (retry_mode != AUD_SV_VAD_RETRY_NO_WAIT) {
            aud_sv_vad_retry_timer_stop_if_idle(in_isr);
        }
        return true;
    }

    if (retry_mode != AUD_SV_VAD_RETRY_NONE) {
        retry_saved = aud_sv_vad_retry_pending_set(p_dmsg);
        if (retry_saved
            && aud_sv_vad_retry_timer_start(in_isr,
                                            retry_mode == AUD_SV_VAD_RETRY_NO_WAIT)) {
            return accept_deferred;
        }
    }
    if (accept_deferred && retry_saved) {
        /* The caller must not commit a desired state that has no delivery path. */
        aud_sv_vad_retry_sent_clear(p_dmsg);
    }

    return false;
}

static bool aud_sv_vad_control_post_seq(const vad_msg_t *p_evt, bool in_isr,
                                        uint8_t retry_mode, uint32_t seq)
{
    return aud_sv_vad_control_post_seq_internal(p_evt, in_isr, retry_mode, seq, false);
}

static bool aud_sv_vad_control_post(const vad_msg_t *p_evt, bool in_isr, uint8_t retry_mode)
{
    return aud_sv_vad_control_post_seq(p_evt, in_isr, retry_mode, 0);
}

static bool aud_sv_vad_control_post_durable(const vad_msg_t *p_evt)
{
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    uint32_t seq;
    uint32_t previous_seq;
    bool previous_pending;
    bool previous_enable;
    bool accepted;
    bool restore_retry = false;

    /* Serialize accepted recording intent with the timeout release claim. */
    cpu_critical_enter();
    if (p_evt->enable && vad_env.candidate_cleanup) {
        cpu_critical_exit();
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 recording intent waits for release\n");
        return false;
    }
    previous_seq = vad_env.record_monitor_request_seq;
    previous_pending = vad_env.record_monitor_request_pending;
    previous_enable = vad_env.record_monitor_request_enable;
    seq = aud_sv_vad_next_seq_locked();
    vad_env.latest_evt_seq[p_evt->evt] = seq;
    vad_env.retry_evt_mask &= ~BIT(p_evt->evt);
    vad_env.record_monitor_request_pending = 1;
    /* A replacement disable cannot expose an accepted enable until the new
     * request has a delivery path. Failed replacements restore/replay it. */
    vad_env.record_monitor_request_enable = p_evt->enable != 0
        || (previous_pending && previous_enable);
    vad_env.record_monitor_request_seq = seq;
    cpu_critical_exit();
    accepted = aud_sv_vad_control_post_seq_internal(p_evt, false, AUD_SV_VAD_RETRY_NORMAL,
                                                    seq, true);
    cpu_critical_enter();
    if (vad_env.record_monitor_request_seq == seq) {
        if (accepted) {
            vad_env.record_monitor_request_enable = p_evt->enable != 0;
        } else {
            vad_env.record_monitor_request_pending = previous_pending;
            vad_env.record_monitor_request_enable = previous_enable;
            vad_env.record_monitor_request_seq = previous_seq;
            if (previous_pending) {
                vad_env.latest_evt_seq[p_evt->evt] = previous_seq;
                vad_env.retry_evt_seq[p_evt->evt] = previous_seq;
                vad_env.retry_record_monitor_enable = previous_enable;
                vad_env.retry_evt_mask |= BIT(p_evt->evt);
                restore_retry = true;
            }
        }
    }
    cpu_critical_exit();
    if (restore_retry) {
        (void)aud_sv_vad_retry_timer_start(false, false);
    }
    return accepted;
#else
    return aud_sv_vad_control_post_seq_internal(p_evt, false, AUD_SV_VAD_RETRY_NORMAL, 0,
                                                true);
#endif
}

static void aud_sv_vad_retry_schedule(vad_evt_t evt)
{
    vad_msg_t retry_evt = {0};

    retry_evt.evt = evt;
    if (evt == VAD_EVT_RECORD_MONITOR) {
        retry_evt.enable = vad_env.record_monitor_desired;
    }
    retry_evt.seq = aud_sv_vad_retry_seq_get(evt);
    if (retry_evt.seq && aud_sv_vad_retry_pending_set(&retry_evt)) {
        (void)aud_sv_vad_retry_timer_start(in_irq(), false);
    }
}

static void aud_sv_vad_retry_cancel(void)
{
    aud_sv_vad_retry_timer_stop_if_idle(in_irq());
}

static void aud_sv_vad_retry_invalidate_stream_events(void)
{
    static const vad_evt_t stream_events[] = {
        VAD_EVT_CREATE, VAD_EVT_DESTROY, VAD_EVT_KWS_TIMEOUT, VAD_EVT_TVAD_DETECT,
        VAD_EVT_GVAD_TIMEOUT, VAD_EVT_VOICE_HIT, VAD_EVT_KWS_HIT, VAD_EVT_RESTART,
    };

    cpu_critical_enter();
    for (uint8_t i = 0; i < sizeof(stream_events) / sizeof(stream_events[0]); i++) {
        vad_evt_t evt = stream_events[i];

        vad_env.latest_evt_seq[evt] = aud_sv_vad_next_seq_locked();
        vad_env.retry_evt_mask &= ~BIT(evt);
    }
    cpu_critical_exit();

    aud_sv_vad_retry_timer_stop_if_idle(in_irq());
}

static void aud_sv_vad_kws_timer_stop(void)
{
    (void)os_stop_timer(vad_env.kws_timer);

    cpu_critical_enter();
    vad_env.latest_evt_seq[VAD_EVT_KWS_TIMEOUT] = aud_sv_vad_next_seq_locked();
    vad_env.retry_evt_mask &= ~BIT(VAD_EVT_KWS_TIMEOUT);
    vad_env.kws_timer_armed = 0;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    vad_env.latest_evt_seq[VAD_EVT_CANDIDATE_TIMEOUT] = aud_sv_vad_next_seq_locked();
    vad_env.latest_evt_seq[VAD_EVT_COOLDOWN_DONE] = aud_sv_vad_next_seq_locked();
    vad_env.retry_evt_mask &= ~(BIT(VAD_EVT_CANDIDATE_TIMEOUT) | BIT(VAD_EVT_COOLDOWN_DONE));
#endif
    cpu_critical_exit();

    aud_sv_vad_retry_timer_stop_if_idle(in_irq());
}

#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
static bool aud_sv_vad_deadline_start(vad_evt_t evt, uint32_t duration_ms)
{
    uint32_t seq;
    TickType_t ticks = duration_ms / portTICK_PERIOD_MS;

    ticks = ticks ? ticks : 1;
    aud_sv_vad_kws_timer_stop();
    seq = aud_sv_vad_event_begin(evt);
    cpu_critical_enter();
    vad_env.kws_timer_event = evt;
    vad_env.kws_timeout_seq = seq;
    vad_env.kws_timeout_deadline_tick = xTaskGetTickCount() + ticks;
    vad_env.kws_timer_armed = 1;
    cpu_critical_exit();
    if (!os_start_timer(vad_env.kws_timer, duration_ms)) {
        aud_sv_vad_kws_timer_stop();
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] VG4 deadline fail evt:%d ms:%d\n", evt, duration_ms);
        return false;
    }
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 deadline evt:%d ms:%d seq:%d\n", evt, duration_ms, seq);
    return true;
}
#endif

static bool aud_sv_vad_kws_timer_start(void)
{
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    return aud_sv_vad_deadline_start(VAD_EVT_KWS_TIMEOUT, AUD_SV_KWS_TIMEOUT_MS);
#else
    uint32_t seq;

    aud_sv_vad_kws_timer_stop();
    seq = aud_sv_vad_event_begin(VAD_EVT_KWS_TIMEOUT);
    cpu_critical_enter();
    vad_env.kws_timeout_seq = seq;
    vad_env.kws_timeout_deadline_tick =
        xTaskGetTickCount() + AUD_SV_KWS_TIMEOUT_MS / portTICK_PERIOD_MS;
    vad_env.kws_timer_armed = 1;
    cpu_critical_exit();

    if (!os_start_timer(vad_env.kws_timer, AUD_SV_KWS_TIMEOUT_MS)) {
        aud_sv_vad_kws_timer_stop();
        return false;
    }

    return true;
#endif
}

#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
static bool aud_sv_vad_candidate_protected(void)
{
    return aud_sv_vad_record_monitor_holds_kws_off() || vad_env.music_start
        || wq_aud_res_mic_other_user_active(WQ_AUD_RES_MIC_USER_VAD)
        || (vad_env.record_monitor_request_pending && vad_env.record_monitor_request_enable);
}

static void aud_sv_vad_candidate_begin(void)
{
    cpu_critical_enter();
    if (!vad_env.candidate_origin && !vad_env.stream_start && !vad_env.stream_guard_held
        && !vad_env.stream_mic_owned && !vad_env.p_mic->mic_stream_start
        && !aud_sv_vad_candidate_protected() && vad_env.public_open_desired
        && !vad_env.cooldown_active) {
        vad_env.candidate_epoch = aud_sv_vad_next_seq_locked();
        vad_env.candidate_origin = 1;
        vad_env.candidate_hit_seen = 0;
        vad_env.candidate_cleanup = 0;
    }
    cpu_critical_exit();
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 candidate begin eligible:%d epoch:%d\n",
                            vad_env.candidate_origin, vad_env.candidate_epoch);
}

static void aud_sv_vad_candidate_release_done(void)
{
    bool cooldown;

    cpu_critical_enter();
    cooldown = vad_env.candidate_cleanup && vad_env.candidate_origin
        && !vad_env.candidate_hit_seen && vad_env.public_open_desired
        && vad_env.stream_intent_evt != VAD_EVT_PEND && !aud_sv_vad_candidate_protected();
    aud_sv_vad_candidate_cancel_locked();
    vad_env.cooldown_active = cooldown;
    cpu_critical_exit();
    aud_sv_vad_front_reset();
    if (cooldown && CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS > 0
        && aud_sv_vad_deadline_start(VAD_EVT_COOLDOWN_DONE,
                                     CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS)) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 cooldown start ms:%d\n",
                                CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS);
    } else {
        vad_env.cooldown_active = 0;
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 listening resumed cooldown:%d\n", cooldown);
    }
}
#endif

#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
static uint32_t aud_sv_vad_gvad_window_next_locked(void)
{
    vad_env.gvad_window_seq++;
    if (!vad_env.gvad_window_seq) {
        vad_env.gvad_window_seq++;
    }

    return vad_env.gvad_window_seq;
}

static void aud_sv_vad_gvad_window_stop(void)
{
    (void)os_stop_timer(vad_env.gvad_timer);

    if (vad_env.gvad_enable) {
        aud_sv_vad_gvad_enable(false);
    } else {
        vad_env.gvad_ret = 0;
        vad_env.gvad_detect_pending = 0;
    }

    cpu_critical_enter();
    (void)aud_sv_vad_gvad_window_next_locked();
    vad_env.gvad_timer_armed = 0;
    vad_env.retry_evt_mask &= ~BIT(VAD_EVT_GVAD_TIMEOUT);
    cpu_critical_exit();

    aud_sv_vad_retry_timer_stop_if_idle(false);
}

static bool aud_sv_vad_gvad_window_start(void)
{
    uint32_t window_seq;

    aud_sv_vad_gvad_window_stop();
    cpu_critical_enter();
    window_seq = aud_sv_vad_gvad_window_next_locked();
    vad_env.gvad_timeout_deadline_tick =
        xTaskGetTickCount() + AUD_SV_GVAD_TIMEOUT_MS / portTICK_PERIOD_MS;
    vad_env.gvad_timer_armed = 1;
    cpu_critical_exit();

    if (!os_start_timer(vad_env.gvad_timer, AUD_SV_GVAD_TIMEOUT_MS)) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] gvad window timer start fail\n");
        aud_sv_vad_gvad_window_stop();
        return false;
    }
    aud_sv_vad_gvad_enable(true);
    DBGLOG_LIB_AUD_SV_INFO("[VAD] gvad window arm seq:%d\n", window_seq);
    return true;
}

static bool aud_sv_vad_gvad_window_is_current(uint32_t window_seq)
{
    bool current;

    cpu_critical_enter();
    current = vad_env.gvad_enable && window_seq && window_seq == vad_env.gvad_window_seq;
    cpu_critical_exit();

    return current;
}
#endif

static WQ_RET aud_sv_vad_create_stream(uint32_t sample_rate)
{
    WQ_RET ret;

    aud_sv_vad_front_reset();
    if (vad_env.stream_start) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] stream already created\n");
        return WQ_RET_AGAIN;
    }

    /* Mark ownership before each asynchronous resource request so an unknown
     * result is cleaned up by the next close instead of dropping the guard. */
    vad_env.stream_guard_held = 1;
#if defined(CONFIG_DCORE_GVAD_ENABLE)
    ret = wq_feat_res_start(WQ_FEAT_RES_USR_GVAD, NULL);
#else
    ret = wq_feat_res_start(WQ_FEAT_RES_USR_VAD, NULL);
#endif
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] stream guard create fail:%d\n", ret);
        return ret;
    }

    ret = (WQ_RET)dsp_create_stream(SM_VAD, 0, sample_rate, 0);
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] dcore stream create fail:%d\n", ret);
        return ret;
    }
    if (ret == WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_WARNING("[VAD] dcore stream already active, resync ownership\n");
    }

    vad_env.stream_mic_owned = 1;
    ret = wq_aud_res_mic_stream_create(WQ_AUD_RES_MIC_USER_VAD);
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] mic stream create fail:%d\n", ret);
        return ret;
    }

    if (!vad_env.record_monitor_desired) {
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        bool timer_started = false;

        if (vad_env.candidate_origin && !vad_env.candidate_hit_seen
            && !aud_sv_vad_candidate_protected()) {
            timer_started = aud_sv_vad_deadline_start(VAD_EVT_CANDIDATE_TIMEOUT,
                                                       CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS);
        }
        if (!timer_started) {
            timer_started = aud_sv_vad_kws_timer_start();
        }
        if (!timer_started) {
            /* A live consumer without either deadline must be released, not
             * repeatedly treated as an incomplete create. */
            vad_env.stream_start = 1;
            cpu_critical_enter();
            if (!vad_env.candidate_hit_seen && !vad_env.voice_active
                && !aud_sv_vad_record_monitor_holds_kws_off()
                && !(vad_env.record_monitor_request_pending
                     && vad_env.record_monitor_request_enable)) {
                vad_env.candidate_cleanup = 1;
            }
            cpu_critical_exit();
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] VG4 timers failed, release consumer\n");
            if (vad_env.candidate_cleanup) {
                return WQ_RET_FAIL;
            }
            /* Already accepted voice/recording work owns the consumer now. */
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] VG4 timer failure protected by active work\n");
        }
#else
        if (!aud_sv_vad_kws_timer_start()) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] kws timer start fail\n");
            return WQ_RET_FAIL;
        }
#endif
    } else {
        aud_sv_vad_kws_timer_stop();
    }

    vad_env.stream_start = 1;
    vad_env.record_monitor_applied = 0;
    vad_env.record_monitor_inflight = 0;
    vad_env.record_monitor_exit_pending = 0;
    vad_env.cur_save_pkt_num = 0;
    DBGLOG_LIB_AUD_SV_INFO("[VAD] stream create, sample_rate:%d\n", sample_rate);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 front paused, dcore owns detection\n");
#endif
    if (vad_env.record_monitor_desired) {
        aud_sv_vad_record_monitor_kick();
    }
    return WQ_RET_OK;
}

static WQ_RET aud_sv_vad_destory_stream(void)
{
    WQ_RET ret;

    aud_sv_vad_front_reset();
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    aud_sv_vad_gvad_window_stop();
#endif
    aud_sv_vad_kws_timer_stop();
    aud_sv_vad_retry_invalidate_stream_events();

    if (!vad_env.stream_start && !vad_env.stream_guard_held
        && !vad_env.stream_mic_owned && !vad_env.kws_trigger) {
        vad_env.record_monitor_applied = 0;
        vad_env.record_monitor_inflight = 0;
        vad_env.record_monitor_exit_pending = 0;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        aud_sv_vad_candidate_release_done();
#endif
        DBGLOG_LIB_AUD_SV_INFO("[VAD] stream already destoryed\n");
        return WQ_RET_AGAIN;
    }

    /* OK means this call removed the DCORE consumer. AGAIN means it was
     * already absent, which is the same target state and must not be retried. */
    ret = (WQ_RET)dsp_destory_stream(SM_VAD);
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] dcore stream destory fail:%d\n", ret);
        return ret;
    }
    if (ret == WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_WARNING("[VAD] dcore stream already inactive, resync ownership\n");
    }

    if (vad_env.stream_mic_owned) {
        ret = wq_aud_res_mic_stream_destory(WQ_AUD_RES_MIC_USER_VAD);
        if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] mic stream destory fail:%d\n", ret);
            return ret;
        }
    }

    if (vad_env.stream_guard_held) {
#if defined(CONFIG_DCORE_GVAD_ENABLE)
        ret = wq_feat_res_stop(WQ_FEAT_RES_USR_GVAD);
#else
        ret = wq_feat_res_stop(WQ_FEAT_RES_USR_VAD);
#endif
        if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] stream guard destory fail:%d\n", ret);
            return ret;
        }
    }
#if defined(CONFIG_DCORE_GVAD_ENABLE)
    if (vad_env.kws_trigger) {
        ret = wq_feat_res_stop(WQ_FEAT_RES_USR_VAD);
        if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] kws guard destory fail:%d\n", ret);
            return ret;
        }
    }
#endif

    vad_env.kws_trigger = 0;
    vad_env.stream_start = 0;
    vad_env.stream_guard_held = 0;
    vad_env.stream_mic_owned = 0;
    vad_env.record_monitor_applied = 0;
    vad_env.record_monitor_inflight = 0;
    vad_env.record_monitor_exit_pending = 0;
    vad_env.voice_active = 0;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    aud_sv_vad_candidate_release_done();
#endif
    DBGLOG_LIB_AUD_SV_INFO("[VAD] stream destory\n");
    return WQ_RET_OK;
}

static bool aud_sv_vad_record_monitor_holds_kws_off(void)
{
    return vad_env.record_monitor_desired || vad_env.record_monitor_applied
        || vad_env.record_monitor_exit_pending
        || (vad_env.record_monitor_inflight && vad_env.record_monitor_pending_enable);
}

#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
static void aud_sv_vad_candidate_timeout_handle(const vad_msg_t *p_evt)
{
    bool claimed = false;
    bool fallback = false;
    WQ_RET ret;

    cpu_critical_enter();
    if (aud_sv_vad_event_is_current_locked(p_evt) && vad_env.candidate_origin
        && vad_env.state == VAD_ST_START && vad_env.stream_start
        && !vad_env.candidate_cleanup && vad_env.public_open_desired
        && vad_env.stream_intent_evt != VAD_EVT_PEND
        && (int32_t)(xTaskGetTickCount() - vad_env.kws_timeout_deadline_tick) >= 0) {
        if (!vad_env.candidate_hit_seen && !vad_env.voice_active
            && !aud_sv_vad_candidate_protected()) {
            /* This is the release linearization point. An already received
             * hit or accepted recording request wins; later arrivals retry. */
            vad_env.candidate_cleanup = 1;
            claimed = true;
        } else {
            vad_env.candidate_origin = 0;
            fallback = !aud_sv_vad_record_monitor_holds_kws_off() && !vad_env.voice_active;
        }
    }
    cpu_critical_exit();
    if (fallback && !aud_sv_vad_kws_timer_start()) {
        cpu_critical_enter();
        if (!vad_env.candidate_hit_seen && !vad_env.voice_active
            && !aud_sv_vad_record_monitor_holds_kws_off()
            && !(vad_env.record_monitor_request_pending
                 && vad_env.record_monitor_request_enable)) {
            vad_env.candidate_cleanup = 1;
            claimed = true;
        }
        cpu_critical_exit();
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] VG4 fallback timer failed cleanup:%d\n", claimed);
    }
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 candidate timeout claim:%d hit:%d fallback:%d\n",
                            claimed, vad_env.candidate_hit_seen, fallback);
    if (!claimed) {
        return;
    }
    aud_sv_vad_notice_app_cb(VAD_HIT_TIMEOUT);
    ret = aud_sv_vad_destory_stream();
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
        return;
    }
    vad_env.state = VAD_ST_OPEN;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    aud_sv_vad_tvad_rearm("candidate timeout");
#endif
    aud_sv_vad_record_monitor_notify_unhit();
}

static void aud_sv_vad_candidate_cooldown_done(const vad_msg_t *p_evt)
{
    bool resume = false;

    cpu_critical_enter();
    if (aud_sv_vad_event_is_current_locked(p_evt) && vad_env.cooldown_active
        && (int32_t)(xTaskGetTickCount() - vad_env.kws_timeout_deadline_tick) >= 0
        && vad_env.public_open_desired && vad_env.stream_intent_evt != VAD_EVT_PEND
        && vad_env.state == VAD_ST_OPEN && !vad_env.stream_start) {
        vad_env.cooldown_active = 0;
        resume = true;
    }
    cpu_critical_exit();
    if (resume) {
        aud_sv_vad_front_reset();
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
        aud_sv_vad_tvad_rearm("candidate cooldown done");
#endif
    }
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 cooldown done resume:%d\n", resume);
}

static bool aud_sv_vad_candidate_voice_receive(vad_msg_t *p_evt)
{
    bool accept;

    cpu_critical_enter();
    accept = !vad_env.candidate_cleanup && !vad_env.cooldown_active
        && vad_env.public_open_desired;
    /* This tags the ACORE queue lifetime, not the source DCORE lifetime:
     * unsolicited voice IPC does not carry a producer generation. */
    p_evt->ipc_seq = vad_env.candidate_epoch;
    if (accept && p_evt->hit_status && (vad_env.stream_start || vad_env.stream_guard_held)) {
        /* Latch at IPC ingress, before a full task queue can delay the hit. */
        vad_env.candidate_hit_seen = 1;
        vad_env.latest_evt_seq[VAD_EVT_CANDIDATE_TIMEOUT] = aud_sv_vad_next_seq_locked();
        vad_env.retry_evt_mask &= ~BIT(VAD_EVT_CANDIDATE_TIMEOUT);
        if (vad_env.kws_timer_event == VAD_EVT_CANDIDATE_TIMEOUT) {
            vad_env.kws_timer_armed = 0;
        }
    }
    cpu_critical_exit();
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 voice ingress accept:%d hit:%d epoch:%d\n",
                            accept, p_evt->hit_status, p_evt->ipc_seq);
    return accept;
}
#endif

static void aud_sv_vad_retry_clear_event(vad_evt_t evt)
{
    cpu_critical_enter();
    vad_env.retry_evt_mask &= ~BIT(evt);
    cpu_critical_exit();
}

static WQ_RET aud_sv_vad_record_monitor_release_kws_guard(void)
{
#if defined(CONFIG_DCORE_GVAD_ENABLE)
    WQ_RET ret;

    if (vad_env.kws_trigger) {
        ret = wq_feat_res_stop(WQ_FEAT_RES_USR_VAD);
        if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] record monitor release kws guard fail:%d\n",
                                    ret);
            aud_sv_vad_retry_schedule(VAD_EVT_RECORD_MONITOR_RETRY);
            return ret;
        }
        vad_env.kws_trigger = 0;
    }
#endif

    return WQ_RET_OK;
}

static WQ_RET aud_sv_vad_record_monitor_notify_unhit(void)
{
    WQ_RET ret;

    if (!vad_env.record_monitor_unhit_notify_pending) {
        return WQ_RET_AGAIN;
    }

    if (vad_env.voice_cb) {
        ret = vad_env.voice_cb(0);
    } else {
        ret = WQ_RET_AGAIN;
    }
    if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
        vad_env.record_monitor_unhit_notify_pending = 0;
        aud_sv_vad_retry_clear_event(VAD_EVT_RECORD_MONITOR_RETRY);
    } else {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] record monitor unhit notify fail:%d\n", ret);
        aud_sv_vad_retry_schedule(VAD_EVT_RECORD_MONITOR_RETRY);
    }

    return ret;
}

static WQ_RET aud_sv_vad_record_monitor_exit_stream(void)
{
    WQ_RET ret = aud_sv_vad_destory_stream();

    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] record monitor stream release fail:%d\n", ret);
        aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
        return ret;
    }

    aud_sv_vad_retry_clear_event(VAD_EVT_RECORD_MONITOR_RETRY);
    /* state is made OPEN before re-arm so the helper can enforce the gate. */
    vad_env.state = VAD_ST_OPEN;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    aud_sv_vad_tvad_rearm("record monitor exit");
#endif
    DBGLOG_LIB_AUD_SV_INFO("[VAD] record monitor exit quiet, stream released\n");
    aud_sv_vad_record_monitor_notify_unhit();
    return WQ_RET_OK;
}

static WQ_RET aud_sv_vad_record_monitor_send_pending(void)
{
    WQ_RET ret;

    ret = aud_sv_vad_status_update_seq(VAD_CMD_KWS_ENABLE,
                                       vad_env.record_monitor_pending_enable ? 0 : 1,
                                       vad_env.record_monitor_pending_seq, WQ_RET_OK);
    DBGLOG_LIB_AUD_SV_INFO(
        "[VAD] record monitor ipc seq:%d enable:%d send:%d\n",
        vad_env.record_monitor_pending_seq, vad_env.record_monitor_pending_enable, ret);
    aud_sv_vad_retry_schedule(VAD_EVT_RECORD_MONITOR_RETRY);
    return ret;
}

static void aud_sv_vad_record_monitor_begin_disable(void)
{
    vad_env.record_monitor_ipc_seq++;
    if (!vad_env.record_monitor_ipc_seq) {
        vad_env.record_monitor_ipc_seq++;
    }
    vad_env.record_monitor_pending_seq = vad_env.record_monitor_ipc_seq;
    vad_env.record_monitor_pending_enable = 1;
    vad_env.record_monitor_inflight = 1;
    (void)aud_sv_vad_record_monitor_send_pending();
}

static void aud_sv_vad_record_monitor_kick(void)
{
    WQ_RET ret;

    if (!vad_env.stream_start) {
        if (vad_env.record_monitor_unhit_notify_pending) {
            ret = aud_sv_vad_record_monitor_notify_unhit();
            if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                return;
            }
        }
        vad_env.record_monitor_applied = 0;
        vad_env.record_monitor_inflight = 0;
        vad_env.record_monitor_exit_pending = 0;
        aud_sv_vad_retry_clear_event(VAD_EVT_RECORD_MONITOR_RETRY);
        return;
    }

    if (vad_env.record_monitor_desired) {
        vad_env.record_monitor_exit_pending = 0;
        aud_sv_vad_kws_timer_stop();
        if (vad_env.record_monitor_inflight) {
            (void)aud_sv_vad_record_monitor_send_pending();
        } else if (vad_env.record_monitor_applied) {
            (void)aud_sv_vad_record_monitor_release_kws_guard();
        } else {
            aud_sv_vad_record_monitor_begin_disable();
        }
        return;
    }

    if (!vad_env.record_monitor_exit_pending) {
        return;
    }
    aud_sv_vad_kws_timer_stop();
    if (vad_env.record_monitor_inflight) {
        (void)aud_sv_vad_record_monitor_send_pending();
        return;
    }
    if (vad_env.state != VAD_ST_START) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] record monitor exit deferred state:%s\n",
                               st_str[vad_env.state]);
        return;
    }
    if (!vad_env.voice_active) {
        (void)aud_sv_vad_record_monitor_exit_stream();
        return;
    }
    if (!vad_env.record_monitor_applied) {
        aud_sv_vad_record_monitor_begin_disable();
        return;
    }

    (void)aud_sv_vad_record_monitor_release_kws_guard();
    DBGLOG_LIB_AUD_SV_INFO("[VAD] record monitor exit waits for voice unhit\n");
}

/**
 * Switch the recording consumer hold independently from the keyword stage.
 * KWS may be compiled out, while this state still controls whether DCORE
 * remains alive or the flow returns to TVAD listening after an unhit.
 */
static void aud_sv_vad_record_monitor_apply(bool enable)
{
    bool was_active = aud_sv_vad_record_monitor_holds_kws_off();

    vad_env.record_monitor_desired = enable;
    if (enable) {
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        vad_env.candidate_origin = 0;
        vad_env.cooldown_active = 0;
        aud_sv_vad_front_reset();
#endif
        vad_env.record_monitor_exit_pending = 0;
        aud_sv_vad_kws_timer_stop();
    } else if (was_active && vad_env.stream_start) {
        vad_env.record_monitor_exit_pending = 1;
        aud_sv_vad_kws_timer_stop();
    }

    DBGLOG_LIB_AUD_SV_INFO(
        "[VAD] record monitor desired:%d applied:%d inflight:%d exit:%d\n", enable,
        vad_env.record_monitor_applied, vad_env.record_monitor_inflight,
        vad_env.record_monitor_exit_pending);
    aud_sv_vad_record_monitor_kick();
}

static void aud_sv_vad_record_monitor_ack_apply(const vad_msg_t *p_evt)
{
    bool ack_enable = p_evt->enable != 0;

    if (!vad_env.record_monitor_inflight
        || p_evt->ipc_seq != vad_env.record_monitor_pending_seq
        || ack_enable != (vad_env.record_monitor_pending_enable != 0)) {
        DBGLOG_LIB_AUD_SV_WARNING(
            "[VAD] stale record monitor ack seq:%d enable:%d pending:%d/%d/%d\n",
            p_evt->ipc_seq, ack_enable, vad_env.record_monitor_inflight,
            vad_env.record_monitor_pending_seq, vad_env.record_monitor_pending_enable);
        return;
    }

    aud_sv_vad_retry_clear_event(VAD_EVT_RECORD_MONITOR_RETRY);
    vad_env.record_monitor_inflight = 0;
    if (p_evt->result == WQ_RET_OK || p_evt->result == WQ_RET_AGAIN) {
        vad_env.record_monitor_applied = ack_enable;
        DBGLOG_LIB_AUD_SV_INFO("[VAD] record monitor ack seq:%d applied:%d ret:%d\n",
                               p_evt->ipc_seq, ack_enable, p_evt->result);
        if (ack_enable) {
            (void)aud_sv_vad_record_monitor_release_kws_guard();
        }
        aud_sv_vad_record_monitor_kick();
    } else {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] record monitor ack fail seq:%d enable:%d ret:%d\n",
                                p_evt->ipc_seq, ack_enable, p_evt->result);
        aud_sv_vad_retry_schedule(VAD_EVT_RECORD_MONITOR_RETRY);
    }
}

static bool aud_sv_vad_stream_prepare(void)
{
    WQ_RET ret;

    if (vad_env.p_mic->mic_stream_start || vad_env.music_start) {
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        vad_env.candidate_origin = 0;
#endif
        DBGLOG_LIB_AUD_SV_INFO("[VAD] stream prepare, mic_stream:%d music:%d\n",
                               vad_env.p_mic->mic_stream_start, vad_env.music_start);
        ret = aud_sv_vad_create_stream(16000);
        if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
            DBGLOG_LIB_AUD_SV_ERROR("[VAD] stream prepare fail:%d\n", ret);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
            if (vad_env.candidate_cleanup) {
                aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
                return true;
            }
#endif
            aud_sv_vad_retry_schedule(VAD_EVT_CREATE);
            return false;
        }

        return true;
    }

    return false;
}

static void aud_sv_vad_stream_cb(stream_cb_cmd_e cmd)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    DBGLOG_LIB_AUD_SV_INFO("[VAD] stream cb %d\n", cmd);
    switch (cmd) {
    case STREAM_CB_CMD_PENDING:
        p_dmsg->evt = VAD_EVT_PEND;
        (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NO_WAIT);
        break;
    case STREAM_CB_CMD_RESUME:
        p_dmsg->evt = VAD_EVT_RESUME;
        (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NO_WAIT);
        break;
    default:
        break;
    }
}

static WQ_RET aud_sv_vad_open(void)
{
    WQ_RET ret = WQ_RET_FAIL;
    int gvad_ret;
    int handle_size;
    char *new_handle;

    /* The GMM library has one global instance. Do not overwrite an existing
     * handle, and validate its settings before opening any physical source. */
    if (vad_env.gvad_hdl != NULL) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] refuse open with existing GMM handle\n");
        return WQ_RET_FAIL;
    }
    aud_sv_vad_front_reset();
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    cpu_critical_enter();
    aud_sv_vad_candidate_cancel_locked();
    cpu_critical_exit();
#endif
    handle_size = wq_gmmvad_get_handle_size();
    if (handle_size <= 0) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] GMM handle size invalid:%d\n", handle_size);
        return WQ_RET_FAIL;
    }
    new_handle = wq_heap_caps_malloc((uint32_t)handle_size, MALLOC_CAP_NONE);
    if (new_handle == NULL) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] GMM allocation fail size:%d\n", handle_size);
        return WQ_RET_FAIL;
    }
    gvad_ret = wq_gmmvad_init(new_handle);
    if (gvad_ret != 0) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] GMM init fail:%d\n", gvad_ret);
        goto gmm_failed;
    }
    gvad_ret = wq_gmmvad_set_mode(AUD_SV_GVAD_MODE);
    if (gvad_ret != 0) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] GMM set mode:%d fail:%d\n", AUD_SV_GVAD_MODE,
                                gvad_ret);
        goto gmm_failed;
    }

    ret = aud_sv_stream_create(STREAM_VAD_ID, aud_sv_vad_stream_cb);
    if (WQ_RET_OK != ret) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] open fail, ret:%d\n", ret);
        goto gmm_failed;
    }
    vad_env.gvad_hdl = new_handle;
#if defined(CONFIG_AUDIO_VAD_INBAND_RING_COEXIST)
    aud_sv_stream_hold_req(AUD_SV_HOLD_USER_VAD, AUD_SV_HOLD_STOP_CALL_UP, NULL);
#else
    aud_sv_stream_hold_req(AUD_SV_HOLD_USER_VAD, AUD_SV_HOLD_STOP_VOICE, NULL);
#endif /* CONFIG_AUDIO_VAD_INBAND_RING_COEXIST */
    aud_sv_evt_callback_register(AUD_SV_EVT_USR_VAD, BIT(AUD_SV_EVT_VOICE),
                                 aud_sv_vad_evt_callback);

    wq_aud_res_mic_open(WQ_AUD_RES_MIC_USER_VAD, NULL, NULL);
#if CONFIG_AUD_RES_MIC_SAMPLE_RATE != 16000
    vad_env.mic_16k_buff = wq_heap_caps_malloc(
        16 * vad_env.p_mic->mic_blk_ms * 2 * vad_env.p_mic->use_mic_num, MALLOC_CAP_NONE);
    assert(vad_env.mic_16k_buff);
#endif
    aud_sv_datamgr_mic_data_hook_register(aud_sv_vad_data_mic_hook);
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    aud_sv_vad_tvad_open();
#elif defined(CONFIG_AUDIO_VAD_WAKEUP_GVAD)
    aud_sv_vad_gvad_enable(true);
#endif
#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
    vad_env.pcm_data_buffer = (int16_t *)wq_heap_caps_malloc(
        32 * vad_env.p_mic->mic_blk_ms * sizeof(int16_t), MALLOC_CAP_NONE);
    assert(vad_env.pcm_data_buffer);
#endif

    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG3 config mode:%d front:%dms hit:%dms\n",
                            AUD_SV_GVAD_MODE, AUD_SV_VAD_FRONT_CONFIRM_MS,
                            AUD_SV_VAD_HIT_CONFIRM_MS);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 candidate poweropt timeout:%dms cooldown:%dms\n",
                            CONFIG_APP_VAD_INVALID_CANDIDATE_TIMEOUT_MS,
                            CONFIG_APP_VAD_INVALID_CANDIDATE_COOLDOWN_MS);
#endif
    DBGLOG_LIB_AUD_SV_INFO("[VAD] open\n");
    return WQ_RET_OK;

gmm_failed:
    /* Clear the library's global pointer before releasing the caller-owned
     * memory. A manager BUSY result retains its original pending/resume flow. */
    wq_gmmvad_free();
    wq_heap_caps_free(new_handle);
    aud_sv_vad_front_reset();
    return ret;
}

static WQ_RET aud_sv_vad_close(stream_cmd_reason_e reason)
{
    WQ_RET ret;

    aud_sv_vad_front_reset();
    /* Do not release the physical source or local processing while ownership
     * of the DCORE consumer is still unknown. */
    ret = aud_sv_vad_destory_stream();
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] close dcore stream fail:%d reason:%d\n", ret, reason);
        return ret;
    }
    vad_env.record_monitor_unhit_notify_pending = 0;

    ret = wq_aud_res_mic_close(WQ_AUD_RES_MIC_USER_VAD);
    if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] close physical mic fail:%d reason:%d\n", ret, reason);
        return ret;
    }

    aud_sv_datamgr_mic_data_hook_register(NULL);
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    aud_sv_vad_tvad_close();
#endif
    wq_gmmvad_free();
    wq_heap_caps_free(vad_env.gvad_hdl);
    vad_env.gvad_hdl = NULL;
#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
    wq_heap_caps_free(vad_env.pcm_data_buffer);
    vad_env.pcm_data_buffer = NULL;
#endif
#if CONFIG_AUD_RES_MIC_SAMPLE_RATE != 16000
    if (vad_env.mic_16k_buff) {
        wq_heap_caps_free(vad_env.mic_16k_buff);
        vad_env.mic_16k_buff = NULL;
    }
#endif

    vad_env.manager_close_pending = 1;
    vad_env.manager_close_reason = reason;
    ret = (WQ_RET)aud_sv_stream_destory(STREAM_VAD_ID, reason);
    if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
        vad_env.manager_close_pending = 0;
    } else {
        /* The physical close is already in flight. CLOSE_DONE will retry the
         * manager update before the state machine completes the transition. */
        DBGLOG_LIB_AUD_SV_ERROR("[VAD] defer service stream close:%d reason:%d\n", ret, reason);
    }

    if (reason != STREAM_CMD_REASON_PENDING) {
        aud_sv_evt_callback_register(AUD_SV_EVT_USR_VAD, BIT(AUD_SV_EVT_VOICE), NULL);
    }
#if defined(CONFIG_AUDIO_VAD_INBAND_RING_COEXIST)
    aud_sv_stream_hold_req(AUD_SV_HOLD_USER_VAD, AUD_SV_HOLD_RESUME_CALL_UP, NULL);
#else
    aud_sv_stream_hold_req(AUD_SV_HOLD_USER_VAD, AUD_SV_HOLD_RESUME_VOICE, NULL);
#endif /* CONFIG_AUDIO_VAD_INBAND_RING_COEXIST */

    DBGLOG_LIB_AUD_SV_INFO("[VAD] close, reason:%d\n", reason);
    return WQ_RET_OK;
}

static WQ_RET aud_sv_vad_manager_close_finish(void)
{
    WQ_RET ret;

    if (!vad_env.manager_close_pending) {
        return WQ_RET_AGAIN;
    }

    ret = (WQ_RET)aud_sv_stream_destory(STREAM_VAD_ID, vad_env.manager_close_reason);
    if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
        vad_env.manager_close_pending = 0;
    }

    return ret;
}

static void aud_sv_vad_notice_app_cb(uint8_t result)
{
    DBGLOG_LIB_AUD_SV_INFO("[VAD] notice app result:%d cb:%x\n", result, vad_env.kws_cb);

    if (vad_env.kws_cb) {
        vad_env.kws_cb(result);
    }
}

static void aud_sv_vad_feat_event_cb(aud_sv_anc_mgr_event_t event, uint32_t param)
{
    UNUSED(param);

    switch (event) {
    case AUD_SV_ANC_MGR_EVENT_MUSIC_START:
        DBGLOG_LIB_AUD_SV_INFO("[VAD] music start\n");
        vad_env.music_start = 1;
        break;
    case AUD_SV_ANC_MGR_EVENT_MUSIC_STOP:
        DBGLOG_LIB_AUD_SV_INFO("[VAD] music stop\n");
        vad_env.music_start = 0;
        break;
    default:
        break;
    }
}

#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
static int16_t *wq_madc_repair_data_proc(int16_t *data, int16_t data_len)
{
    assert((data_len == 320) || (data_len == 640));   //10ms or 20ms

    int frame_len = data_len == 320 ? 160 : 320;
    int end_idx = data_len == 320 ? 316 : 636;

    memcpy(&vad_env.pcm_data_buffer[frame_len], data, frame_len * 2);

    for (int j = 0; j < (frame_len * 2); ++j) {
        if (vad_env.pcm_data_buffer[j] == 0x7FFF) {
            if (j <= end_idx) {
                int a = vad_env.pcm_data_buffer[j - 1];
                int b = vad_env.pcm_data_buffer[j + 1];

                int c = (a + b) / 2;

                vad_env.pcm_data_buffer[j] = c;
            } else {
                DBGLOG_AUD_DEV_INFO("[VAD] drop data touch the end %d , end_idx %d \n", j, end_idx);
            }
        }
    }

    return vad_env.pcm_data_buffer;
}

static void wq_madc_repair_data_proc_after(int16_t data_len)
{
    assert((data_len == 320) || (data_len == 640));   //10ms or 20ms

    int frame_len = data_len == 320 ? 160 : 320;

    memmove(vad_env.pcm_data_buffer, &vad_env.pcm_data_buffer[frame_len], frame_len * 2);
}
#endif

// static void aud_sv_vad_flush_ring_pkt(void)
// {
//     uint16_t mic_length;

//     if (!vad_env.p_mic->mic_stream_start) {
//         mic_length = vad_env.p_mic->use_mic_num * vad_env.p_mic->mic_blk_ms * 16 * 2;

//         while (aud_sv_vad_read_ring_pkt(mic_length, true) == WQ_RET_OK) {
//             // Keep reading until the ring buffer is empty
//         }
//         vad_env.cur_save_pkt_num = 0;
//         DBGLOG_LIB_AUD_SV_INFO("[VAD] mic flush done\n");
//     } else {
//         DBGLOG_LIB_AUD_SV_INFO("[VAD] mic flush fail, mic stream exits\n");
//     }
// }

static WQ_RET aud_sv_vad_read_ring_pkt(uint16_t length, bool force_read)
    IRAM_TEXT(aud_sv_vad_read_ring_pkt);
static WQ_RET aud_sv_vad_read_ring_pkt(uint16_t data_len, bool force_read)
{
    uint16_t read_pkt_len;
    uint16_t len_without_pad = DATAPATH_PKT_PCMHDR_LEN + vad_env.p_mic->use_mic_num * data_len;
    uint16_t pad_num;

    if (!force_read && (vad_env.cur_save_pkt_num < AUD_SV_VAD_RING_SAVE_PKT_NUM)) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] pkt skip num:%d\n", vad_env.cur_save_pkt_num);
        vad_env.cur_save_pkt_num++;
        return WQ_RET_OK;
    }

    if (len_without_pad % DATAPATH_PKT_UP_ALIGN_LEN == 0) {
        pad_num = 0;
    } else {
        pad_num = DATAPATH_PKT_UP_ALIGN_LEN - len_without_pad % DATAPATH_PKT_UP_ALIGN_LEN;
    }
    read_pkt_len = len_without_pad + pad_num;

    return multicore_datapath_read(WQ_CORES_DCORE, DP_RID_DFA, NULL, read_pkt_len);
}

#if CONFIG_AUD_RES_MIC_SAMPLE_RATE != 16000
#if CONFIG_AUD_RES_MIC_SAMPLE_RATE == 32000
static void aud_sv_vad_mic_32k_to_16k(int16_t *data, uint16_t len)
{
    assert(!(len % 2));
    assert(vad_env.mic_16k_buff);

    for (uint16_t i = 0; i < len / 2; i++) {
        vad_env.mic_16k_buff[i] = data[i * 2];
    }
}
#endif

#if CONFIG_AUD_RES_MIC_SAMPLE_RATE == 48000
static void aud_sv_vad_mic_48k_to_16k(int16_t *data, uint16_t len)
{
    assert(!(len % 3));
    assert(vad_env.mic_16k_buff);

    for (uint16_t i = 0; i < len / 3; i++) {
        vad_env.mic_16k_buff[i] = data[i * 3];
    }
}
#endif

static void aud_sv_vad_mic_convert(int16_t *data, uint16_t len)
{
    UNUSED(data);
    UNUSED(len);

#if (CONFIG_AUD_RES_MIC_SAMPLE_RATE == 32000)
    aud_sv_vad_mic_32k_to_16k(data, len);
#elif (CONFIG_AUD_RES_MIC_SAMPLE_RATE == 48000)
    aud_sv_vad_mic_48k_to_16k(data, len);
#endif
}
#endif

static void aud_sv_vad_gvad_detect_hdl(int16_t *data, uint16_t length)
{
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    if (!aud_sv_vad_front_should_process()) {
        aud_sv_vad_front_reset();
        if (!vad_env.p_mic->mic_stream_start) {
            (void)aud_sv_vad_read_ring_pkt(length, false);
        }
        return;
    }
#endif
#if 0
    DBGLOG_LIB_AUD_SV_INFO("[VAD] dbg save:%d start:%d length:%d\n", vad_env.cur_save_pkt_num,
                           vad_env.p_mic->mic_stream_start, length);
#endif
    uint8_t vad_wake_up_mic_idx = wq_aud_res_mic_index_get(WQ_AUD_RES_MIC_TYPE_VAD_WAKE_UP);
    assert(vad_wake_up_mic_idx < RO_RES_MIC_MAX);

#if CONFIG_AUD_RES_MIC_SAMPLE_RATE != 16000
    aud_sv_vad_mic_convert(data, length);
    uint16_t data_sp = (length / 3) >> 1;
    int16_t *vad_data = vad_env.mic_16k_buff + data_sp * vad_wake_up_mic_idx;
#else
    uint16_t data_sp = length >> 1;
    int16_t *vad_data = data + data_sp * vad_wake_up_mic_idx;
#endif
    int ret = 0;
    uint32_t start_us = wq_rtc_get_time_us();
    uint32_t proc_us;
    uint32_t end_us;

#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
    if (vad_wake_up_mic_idx == vad_env.p_mic->madc_mic_idx) {
        int16_t *repair_data = wq_madc_repair_data_proc((int16_t *)vad_data, data_sp * 2);
        proc_us = wq_rtc_get_time_us();
        ret = wq_gmmvad_process(repair_data, data_sp);
        wq_madc_repair_data_proc_after(data_sp * 2);
        end_us = wq_rtc_get_time_us();
    } else {
        proc_us = wq_rtc_get_time_us();
        ret = wq_gmmvad_process((int16_t *)vad_data, data_sp);
        end_us = wq_rtc_get_time_us();
    }
#else
    proc_us = wq_rtc_get_time_us();
    ret = wq_gmmvad_process((int16_t *)vad_data, data_sp);
    end_us = wq_rtc_get_time_us();
#endif

    if (vad_env.gvad_enable) {
        if (vad_env.state == VAD_ST_OPEN) {
            if (ret != vad_env.gvad_ret) {
                DBGLOG_AUD_DEV_INFO(
                    "[VAD] gvad process num:%d diff:%dus proc:%dus %dus ret:%d save:%d\n", data_sp,
                    end_us - start_us, proc_us - start_us, end_us - proc_us, ret,
                    vad_env.cur_save_pkt_num);
                vad_env.gvad_ret = ret;
            }
            if (!vad_env.gvad_detect_pending
                && aud_sv_vad_front_confirm(ret, data_sp, wq_rtc_get_global_time_ms())) {
                dp_msg_t msg = {0};
                vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

                p_dmsg->evt = VAD_EVT_GVAD_DETECT;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
                cpu_critical_enter();
                p_dmsg->gvad_window_seq = vad_env.gvad_window_seq;
                cpu_critical_exit();
#endif
                if (aud_sv_task_send_msg(AS_MSG_VAD, &msg, false, sizeof(*p_dmsg))) {
                    vad_env.gvad_detect_pending = 1;
                    aud_sv_vad_front_reset();
                }
            }
        } else {
            vad_env.gvad_ret = 0;
            aud_sv_vad_front_reset();
        }
    } else {
        aud_sv_vad_front_reset();
    }

    if (!vad_env.p_mic->mic_stream_start) {
        // read ring in case of mic ring full
        aud_sv_vad_read_ring_pkt(length, false);
    }
}

static void aud_sv_vad_msg_hdl(void *p_msg)
{
    vad_msg_t *p_evt = (vad_msg_t *)p_msg;
    vad_state_t old_state = vad_env.state;
    bool evt_not_handle = false;
    WQ_RET ret;

    aud_sv_vad_retry_sent_clear(p_evt);
    if (!aud_sv_vad_event_is_current(p_evt)) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] drop stale evt:%s seq:%d\n",
                               evt_str[p_evt->evt], p_evt->seq);
        (void)aud_sv_vad_retry_timer_start(false, false);
        aud_sv_vad_retry_timer_stop_if_idle(false);
        return;
    }
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    if ((p_evt->evt == VAD_EVT_GVAD_DETECT || p_evt->evt == VAD_EVT_GVAD_TIMEOUT)
        && !aud_sv_vad_gvad_window_is_current(p_evt->gvad_window_seq)) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] drop stale %s window:%d current:%d\n",
                               evt_str[p_evt->evt], p_evt->gvad_window_seq,
                               vad_env.gvad_window_seq);
        (void)aud_sv_vad_retry_timer_start(false, false);
        aud_sv_vad_retry_timer_stop_if_idle(false);
        return;
    }
#endif
    if (p_evt->evt == VAD_EVT_GVAD_DETECT) {
        vad_env.gvad_detect_pending = 0;
    }
    if (p_evt->evt == VAD_EVT_RECORD_MONITOR) {
        aud_sv_vad_record_monitor_apply(p_evt->enable != 0);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        cpu_critical_enter();
        if (vad_env.record_monitor_request_seq == p_evt->seq) {
            vad_env.record_monitor_request_pending = 0;
        }
        cpu_critical_exit();
#endif
        goto msg_done;
    }
    if (p_evt->evt == VAD_EVT_RECORD_MONITOR_ACK) {
        aud_sv_vad_record_monitor_ack_apply(p_evt);
        goto msg_done;
    }
    if (p_evt->evt == VAD_EVT_RECORD_MONITOR_RETRY) {
        aud_sv_vad_record_monitor_kick();
        goto msg_done;
    }
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    if (p_evt->evt == VAD_EVT_CANDIDATE_TIMEOUT) {
        aud_sv_vad_candidate_timeout_handle(p_evt);
        goto msg_done;
    }
    if (p_evt->evt == VAD_EVT_COOLDOWN_DONE) {
        aud_sv_vad_candidate_cooldown_done(p_evt);
        goto msg_done;
    }
    if (p_evt->evt == VAD_EVT_VOICE_HIT
        && (vad_env.candidate_cleanup || vad_env.cooldown_active
            || p_evt->ipc_seq != vad_env.candidate_epoch)) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] VG4 drop retired voice epoch:%d current:%d\n",
                                p_evt->ipc_seq, vad_env.candidate_epoch);
        goto msg_done;
    }
#endif

    switch (vad_env.state) {
    case VAD_ST_IDLE:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            ret = aud_sv_vad_open();
            if (ret == WQ_RET_OK) {
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_OPENING;
            } else if (ret == WQ_RET_BUSY) {
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_PENDING;
                vad_env.pend_by_stream = 1;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] open request fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_OPEN);
            }

            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    case VAD_ST_OPENING:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            if (!vad_env.pend_by_stream) {
                vad_env.pend_stop = 0;
            }
            break;
        case VAD_EVT_OPEN_DONE:
            if (vad_env.pend_stop) {
                stream_cmd_reason_e reason = vad_env.pend_by_stream
                    ? STREAM_CMD_REASON_PENDING : STREAM_CMD_REASON_NORMAL;

                ret = aud_sv_vad_close(reason);
                if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_stop = 0;
                    vad_env.state = vad_env.pend_by_stream
                        ? VAD_ST_PENDING_CLOSING : VAD_ST_CLOSING;
                } else {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] deferred close fail:%d reason:%d\n",
                                            ret, reason);
                    vad_env.pend_stop = 0;
                    vad_env.state = vad_env.stream_start ? VAD_ST_START : VAD_ST_OPEN;
                    aud_sv_vad_retry_schedule(vad_env.pend_by_stream ? VAD_EVT_PEND
                                                                     : VAD_EVT_CLOSE);
                }
                break;
            }

            if (aud_sv_vad_stream_prepare()) {
                vad_env.state = VAD_ST_START;
            } else {
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
                aud_sv_vad_tvad_start();
#endif
                vad_env.state = VAD_ST_OPEN;
            }
            break;
        case VAD_EVT_CLOSE:
            vad_env.pend_stop = 1;
            vad_env.pend_by_stream = 0;
            break;
        case VAD_EVT_PEND:
            if (!vad_env.pend_stop) {
                vad_env.pend_stop = 1;
                vad_env.pend_by_stream = 1;
            }
            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    case VAD_ST_OPEN:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            break;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
        case VAD_EVT_TVAD_DETECT:
            if (aud_sv_vad_stream_prepare()) {
                vad_env.state = VAD_ST_START;
            } else {
                if (!aud_sv_vad_gvad_window_start()) {
                    aud_sv_vad_tvad_rearm("GVAD window start failure");
                }
            }
            break;
#endif /* defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD) */
        case VAD_EVT_MIC_PACKET:
            aud_sv_vad_gvad_detect_hdl((int16_t *)p_evt->data, p_evt->data_len);
            break;
        case VAD_EVT_GVAD_DETECT:
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
            if (!aud_sv_vad_front_should_process()) {
                break;
            }
            aud_sv_vad_candidate_begin();
#endif
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
            aud_sv_vad_gvad_window_stop();
#endif
            ret = aud_sv_vad_create_stream(16000);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_START;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] detect stream create fail:%d\n", ret);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
                if (vad_env.candidate_cleanup) {
                    vad_env.state = VAD_ST_START;
                    aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
                    break;
                }
#endif
                aud_sv_vad_retry_schedule(VAD_EVT_CREATE);
            }
            break;
        case VAD_EVT_CREATE:
            ret = aud_sv_vad_create_stream(16000);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
                aud_sv_vad_gvad_window_stop();
#endif
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_START;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] retry stream create fail:%d\n", ret);
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
                if (vad_env.candidate_cleanup) {
                    vad_env.state = VAD_ST_START;
                    aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
                    break;
                }
#endif
                aud_sv_vad_retry_schedule(VAD_EVT_CREATE);
            }
            break;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
        case VAD_EVT_GVAD_TIMEOUT:
            aud_sv_vad_gvad_window_stop();
            if (aud_sv_vad_stream_prepare()) {
                vad_env.state = VAD_ST_START;
            } else {
                vad_env.state = VAD_ST_OPEN;
                aud_sv_vad_tvad_rearm("GVAD window timeout");
            }
            break;
#endif
        case VAD_EVT_CLOSE:
            ret = aud_sv_vad_close(STREAM_CMD_REASON_NORMAL);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_by_stream = 0;
                vad_env.state = VAD_ST_CLOSING;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] close from open fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_CLOSE);
            }
            break;
        case VAD_EVT_PEND:
            ret = aud_sv_vad_close(STREAM_CMD_REASON_PENDING);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_start = 0;
                vad_env.pend_stop = 0;
                vad_env.pend_by_stream = 1;
                vad_env.state = VAD_ST_PENDING_CLOSING;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] pend from open fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_PEND);
            }
            break;
        default:
            evt_not_handle = true;
        }
        break;
    case VAD_ST_START:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            break;
        case VAD_EVT_MIC_PACKET:
            aud_sv_vad_gvad_detect_hdl((int16_t *)p_evt->data, p_evt->data_len);
            break;
        case VAD_EVT_VOICE_HIT:
            vad_env.voice_active = p_evt->hit_status != 0;
            if (!aud_sv_vad_record_monitor_holds_kws_off()) {
                if (p_evt->hit_status) {
                    aud_sv_vad_kws_timer_stop();
                    if (!vad_env.kws_trigger) {
                        wq_feat_res_start(WQ_FEAT_RES_USR_VAD, NULL);
                        vad_env.kws_trigger = 1;
                    }
                } else {
                    if (!aud_sv_vad_kws_timer_start()) {
                        DBGLOG_LIB_AUD_SV_ERROR("[VAD] kws timer restart fail\n");
                    }
                }
            }
            if (!p_evt->hit_status && vad_env.record_monitor_exit_pending) {
                vad_env.record_monitor_unhit_notify_pending = 1;
                (void)aud_sv_vad_record_monitor_exit_stream();
            } else if (vad_env.voice_cb) {
                (void)vad_env.voice_cb(p_evt->hit_status);
            }
            break;
        case VAD_EVT_KWS_HIT:
            if (!aud_sv_vad_record_monitor_holds_kws_off()) {
                aud_sv_vad_notice_app_cb(p_evt->hit_status);
            }
            break;
        case VAD_EVT_KWS_TIMEOUT:
            if (aud_sv_vad_record_monitor_holds_kws_off()) {
                DBGLOG_LIB_AUD_SV_INFO("[VAD] ignore queued kws timeout in record monitor\n");
                break;
            }
            aud_sv_vad_notice_app_cb(VAD_HIT_TIMEOUT);
            ret = aud_sv_vad_destory_stream();
            if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] timeout stream destory fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
                break;
            }
            aud_sv_vad_retry_cancel();
            vad_env.state = VAD_ST_OPEN;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
            aud_sv_vad_tvad_rearm("KWS timeout");
#endif
            aud_sv_vad_record_monitor_notify_unhit();
            break;
        case VAD_EVT_DESTROY:
            ret = aud_sv_vad_destory_stream();
            if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] retry stream destory fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_DESTROY);
                break;
            }
            aud_sv_vad_retry_cancel();
            vad_env.state = VAD_ST_OPEN;
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
            aud_sv_vad_tvad_rearm("stream destroy");
#endif
            aud_sv_vad_record_monitor_notify_unhit();
            break;
        case VAD_EVT_RESTART:
            aud_sv_vad_status_update(VAD_CMD_RESTART, 0);
            if (!aud_sv_vad_record_monitor_holds_kws_off()) {
                if (!aud_sv_vad_kws_timer_start()) {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] kws timer restart fail\n");
                }
            }
            break;
        case VAD_EVT_PEND:
            ret = aud_sv_vad_close(STREAM_CMD_REASON_PENDING);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_start = 0;
                vad_env.pend_stop = 0;
                vad_env.pend_by_stream = 1;
                vad_env.state = VAD_ST_PENDING_CLOSING;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] pend from start fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_PEND);
            }
            break;
        case VAD_EVT_CLOSE:
            ret = aud_sv_vad_close(STREAM_CMD_REASON_NORMAL);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_by_stream = 0;
                vad_env.state = VAD_ST_CLOSING;
            } else {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] close from start fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_CLOSE);
            }
            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    case VAD_ST_PENDING_CLOSING:
        switch (p_evt->evt) {
        case VAD_EVT_RESUME:
            if (!vad_env.pend_stop) {
                vad_env.pend_start = 1;
            }
            break;
        case VAD_EVT_OPEN:
            vad_env.pend_stop = 0;
            vad_env.pend_start = 1;
            break;
        case VAD_EVT_CLOSE:
            vad_env.pend_start = 0;
            vad_env.pend_stop = 1;
            break;
        case VAD_EVT_CLOSE_DONE:
            ret = aud_sv_vad_manager_close_finish();
            if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] pending manager close retry fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_CLOSE_DONE);
                break;
            }
            if (vad_env.pend_stop) {
                ret = (WQ_RET)aud_sv_stream_destory(STREAM_VAD_ID,
                                                    STREAM_CMD_REASON_NORMAL);
                if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] cancel pending close fail:%d\n", ret);
                    aud_sv_vad_retry_schedule(VAD_EVT_CLOSE_DONE);
                    break;
                }
                aud_sv_vad_retry_cancel();
                vad_env.pend_stop = 0;
                vad_env.pend_by_stream = 0;
                vad_env.state = VAD_ST_IDLE;
                aud_sv_evt_callback_register(AUD_SV_EVT_USR_VAD, BIT(AUD_SV_EVT_VOICE), NULL);
                if (vad_env.close_done_cb) {
                    vad_env.close_done_cb();
                }
            } else if (vad_env.pend_start) {
                ret = aud_sv_vad_open();
                if (ret == WQ_RET_OK) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_start = 0;
                    vad_env.pend_by_stream = 0;
                    vad_env.state = VAD_ST_OPENING;
                } else if (ret == WQ_RET_BUSY) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_start = 0;
                    vad_env.state = VAD_ST_PENDING;
                } else {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] resume after pending close fail:%d\n", ret);
                    aud_sv_vad_retry_schedule(VAD_EVT_CLOSE_DONE);
                }
            } else {
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_PENDING;
            }
            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    case VAD_ST_PENDING:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            vad_env.pend_stop = 0;
            break;
        case VAD_EVT_RESUME:
            if (vad_env.pend_stop) {
                ret = (WQ_RET)aud_sv_stream_destory(STREAM_VAD_ID,
                                                    STREAM_CMD_REASON_NORMAL);
                if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_stop = 0;
                    vad_env.pend_by_stream = 0;
                    vad_env.state = VAD_ST_IDLE;
                    aud_sv_evt_callback_register(AUD_SV_EVT_USR_VAD, BIT(AUD_SV_EVT_VOICE), NULL);
                    if (vad_env.close_done_cb) {
                        vad_env.close_done_cb();
                    }
                } else {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] discard pending resume fail:%d\n", ret);
                    aud_sv_vad_retry_schedule(VAD_EVT_CLOSE);
                }
                break;
            }

            ret = aud_sv_vad_open();
            if (ret == WQ_RET_OK) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_by_stream = 0;
                vad_env.state = VAD_ST_OPENING;
            } else if (ret != WQ_RET_BUSY) {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] resume from pending fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_RESUME);
            }
            break;
        case VAD_EVT_CLOSE:
            ret = (WQ_RET)aud_sv_stream_destory(STREAM_VAD_ID,
                                                STREAM_CMD_REASON_NORMAL);
            if (ret == WQ_RET_OK || ret == WQ_RET_AGAIN) {
                aud_sv_vad_retry_cancel();
                vad_env.pend_start = 0;
                vad_env.pend_stop = 0;
                vad_env.pend_by_stream = 0;
                vad_env.state = VAD_ST_IDLE;
                aud_sv_evt_callback_register(AUD_SV_EVT_USR_VAD, BIT(AUD_SV_EVT_VOICE), NULL);
                if (vad_env.close_done_cb) {
                    vad_env.close_done_cb();
                }
            } else {
                vad_env.pend_stop = 1;
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] close fully pending fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_CLOSE);
            }
            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    case VAD_ST_CLOSING:
        switch (p_evt->evt) {
        case VAD_EVT_OPEN:
            vad_env.pend_start = 1;
            break;
        case VAD_EVT_CLOSE:
            vad_env.pend_start = 0;
            break;
        case VAD_EVT_CLOSE_DONE:
            ret = aud_sv_vad_manager_close_finish();
            if (ret != WQ_RET_OK && ret != WQ_RET_AGAIN) {
                DBGLOG_LIB_AUD_SV_ERROR("[VAD] manager close retry fail:%d\n", ret);
                aud_sv_vad_retry_schedule(VAD_EVT_CLOSE_DONE);
                break;
            }
            if (vad_env.pend_start) {
                ret = aud_sv_vad_open();
                if (ret == WQ_RET_OK) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_start = 0;
                    vad_env.state = VAD_ST_OPENING;
                } else if (ret == WQ_RET_BUSY) {
                    aud_sv_vad_retry_cancel();
                    vad_env.pend_start = 0;
                    vad_env.pend_by_stream = 1;
                    vad_env.state = VAD_ST_PENDING;
                } else {
                    DBGLOG_LIB_AUD_SV_ERROR("[VAD] restart after close fail:%d\n", ret);
                    aud_sv_vad_retry_schedule(VAD_EVT_CLOSE_DONE);
                }
            } else {
                aud_sv_vad_retry_cancel();
                vad_env.state = VAD_ST_IDLE;
                if (vad_env.close_done_cb) {
                    vad_env.close_done_cb();
                }
            }
            break;
        default:
            evt_not_handle = true;
            break;
        }
        break;
    default:
        assert(0);
        break;
    }

msg_done:
    if (old_state != vad_env.state) {
        aud_sv_vad_front_reset();
    }
    if (evt_not_handle) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] msg hdl unexcept evt:%s state:%s\n", evt_str[p_evt->evt],
                               st_str[vad_env.state]);
    } else {
        if (p_evt->evt != VAD_EVT_MIC_PACKET) {
            DBGLOG_LIB_AUD_SV_INFO(
                "[VAD] msg hdl state:%s->%s evt:%s mic:%d music:%d pend_start:%d pend_stop:%d\n",
                st_str[old_state], st_str[vad_env.state], evt_str[p_evt->evt],
                vad_env.p_mic->mic_stream_start, vad_env.music_start, vad_env.pend_start,
                vad_env.pend_stop);
        }
    }
    (void)aud_sv_vad_retry_timer_start(false, false);
    aud_sv_vad_retry_timer_stop_if_idle(false);
}

#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
static void aud_sv_vad_tvad_process_cb(void *param)
{
    WQ_VAD_MODULE module = *(WQ_VAD_MODULE *)param;

    if (module != WQ_VAD_MODULE_TVAD) {
        return;
    }

    DBGLOG_AUD_DEV_INFO("[VAD] tvad detect irq\n");
    aud_sv_vad_tvad_mask();

    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    p_dmsg->evt = VAD_EVT_TVAD_DETECT;
    (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL);
}

static void aud_sv_vad_tvad_done_cb(uint8_t type, void *param)
{
    DBGLOG_AUD_DEV_INFO("[VAD] tvad done cb, type:%d param:%p\n", type, param);
    if (type == RECORD_VAD_CMD_OPEN) {
        aud_vad_open_param_t *vad_param = (aud_vad_open_param_t *)param;

        assert(vad_param->mode_cfg);
        wq_heap_caps_free(vad_param->mode_cfg);
    }
}

static void aud_sv_vad_tvad_open(void)
{
    audsys_record_msg_u vad_msg = {0};

    vad_msg.record_vad.cmd = RECORD_VAD_CMD_OPEN;
    vad_msg.record_vad.cfg.open.mode_cfg =
        wq_heap_caps_malloc(sizeof(wq_vad_mode_config_t), MALLOC_CAP_NONE);
    assert(vad_msg.record_vad.cfg.open.mode_cfg);
    memset(vad_msg.record_vad.cfg.open.mode_cfg, 0, sizeof(wq_vad_mode_config_t));
    vad_msg.record_vad.cfg.open.mode_cfg->src_chn = 0;
    vad_msg.record_vad.cfg.open.mode_cfg->frq = WQ_VAD_FREQUENCE_16K;
    vad_msg.record_vad.cfg.open.mode_cfg->size = WQ_VAD_MEM_SIZE_4K;
#if defined(CONFIG_METER_ADC_REPAIR_CALL_DATA)
    vad_msg.record_vad.cfg.open.mode_cfg->mode = WQ_VAD_SRC_MODE_DOUBLE;
    vad_msg.record_vad.cfg.open.mode_cfg->src[0] = WQ_VAD_SRC_MADC;
    vad_msg.record_vad.cfg.open.mode_cfg->src[1] = WQ_VAD_SRC_MADC;
#else
    vad_msg.record_vad.cfg.open.mode_cfg->mode = WQ_VAD_SRC_MODE_SINGLE;
    vad_msg.record_vad.cfg.open.mode_cfg->src[0] = WQ_VAD_SRC_AUD_INTF;
    /* HAL reads both source selectors; keep the unused selector deterministic. */
    vad_msg.record_vad.cfg.open.mode_cfg->src[1] = WQ_VAD_SRC_AUD_INTF;
#endif
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.low_ratio = 1024;
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.high_ratio = 1024;
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.method = true;
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.low_len = 0x7;
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.high_len = 0x7;
    vad_msg.record_vad.cfg.open.mode_cfg->dec_cfg.thr_min = 100;
    vad_msg.record_vad.cfg.open.trans_buf_cfg.tvad_trans_buffer = vad_env.tvad_buffer;
    vad_msg.record_vad.cfg.open.trans_buf_cfg.tvad_trans_length = AUD_SV_TVAD_BUF_LEN;
#if defined(CONFIG_BUILD_SERIES_BBB)
    vad_msg.record_vad.cfg.open.trans_buf_cfg.gvad_buffer = vad_env.gvad_buffer;
    vad_msg.record_vad.cfg.open.trans_buf_cfg.gvad_length = AUD_SV_GVAD_BUF_LEN;
#endif

    vad_msg.record_vad.cfg.open.module_process_cb = aud_sv_vad_tvad_process_cb;
    vad_msg.record_vad.cfg.open.done_cb = aud_sv_vad_tvad_done_cb;
    wq_aud_res_vad_config(&vad_msg);

#if defined(CONFIG_BUILD_SERIES_HORNET)
    memset(&vad_msg, 0, sizeof(vad_msg));
    vad_msg.record_vad.cmd = RECORD_VAD_CMD_SET_STATUS;
    vad_msg.record_vad.cfg.set.set_type = WQ_VAD_SET_TYPE_BUF_INT_MSK;
    wq_aud_res_vad_config(&vad_msg);
    vad_msg.record_vad.cfg.set.set_type = WQ_VAD_SET_TYPE_DBG_INT_MSK;
    wq_aud_res_vad_config(&vad_msg);
    DBGLOG_AUD_DEV_INFO("[VAD] tvad quiet irq mask buffer/debug\n");
#endif

    DBGLOG_AUD_DEV_INFO("[VAD] tvad open\n");
}

static void aud_sv_vad_tvad_start(void)
{
    audsys_record_msg_u vad_msg = {0};

    vad_msg.record_vad.cmd = RECORD_VAD_CMD_START;
#if defined(CONFIG_BUILD_SERIES_BBB)
    vad_msg.record_vad.cfg.start.start_type = WQ_VAD_START_TYPE_TVAD;
#else
    vad_msg.record_vad.cfg.start.start_type = WQ_VAD_START_TYPE_ALL;
#endif
    vad_msg.record_vad.cfg.start.done_cb = aud_sv_vad_tvad_done_cb;
    wq_aud_res_vad_config(&vad_msg);

    DBGLOG_AUD_DEV_INFO("[VAD] tvad start\n");
}

static void aud_sv_vad_tvad_mask(void)
{
    audsys_record_msg_u vad_msg = {0};

    vad_msg.record_vad.cmd = RECORD_VAD_CMD_SET_STATUS;
    vad_msg.record_vad.cfg.set.set_type = WQ_VAD_SET_TYPE_TVAD_INT_MSK;
    wq_aud_res_vad_config(&vad_msg);

    DBGLOG_AUD_DEV_INFO("[VAD] tvad mask\n");
}

static void aud_sv_vad_tvad_unmask(void)
{
    audsys_record_msg_u vad_msg = {0};

    vad_msg.record_vad.cmd = RECORD_VAD_CMD_SET_STATUS;
    vad_msg.record_vad.cfg.set.set_type = WQ_VAD_SET_TYPE_TVAD_INT_UNMSK;
    wq_aud_res_vad_config(&vad_msg);

    // reset tvad status, trigger tvad interrupt in next detect
    vad_msg.record_vad.cmd = RECORD_VAD_CMD_SET_STATUS;
    vad_msg.record_vad.cfg.set.set_type = WQ_VAD_SET_TYPF_FALSE_ALAM_SET;
    wq_aud_res_vad_config(&vad_msg);

    DBGLOG_AUD_DEV_INFO("[VAD] tvad unmask\n");
}

/* Re-arm TVAD only after the ACORE/DCORE consumer has been fully released.
 * Candidate cooldown deliberately keeps the hardware masked; the cooldown
 * completion path calls this helper once the listening state is eligible. */
static void aud_sv_vad_tvad_rearm(const char *reason)
{
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
    if (vad_env.cooldown_active) {
        DBGLOG_AUD_DEV_INFO("[VAD] tvad rearm deferred (%s), cooldown active\n", reason);
        return;
    }
#endif
    if (vad_env.state != VAD_ST_OPEN || !vad_env.public_open_desired) {
        DBGLOG_AUD_DEV_INFO("[VAD] tvad rearm deferred (%s), state:%d open:%d\n",
                            reason, vad_env.state, vad_env.public_open_desired);
        return;
    }
    aud_sv_vad_tvad_unmask();
    DBGLOG_AUD_DEV_INFO("[VAD] tvad rearm (%s)\n", reason);
}

static void aud_sv_vad_tvad_close(void)
{
    audsys_record_msg_u vad_msg = {0};

    vad_msg.record_vad.cmd = RECORD_VAD_CMD_CLOSE;
    vad_msg.record_vad.cfg.close.close_type = WQ_VAD_CLOSE_TYPE_ALL;
    wq_aud_res_vad_config(&vad_msg);

    DBGLOG_AUD_DEV_INFO("[VAD] tvad close\n");
}
#endif /* defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD) */

static void aud_sv_vad_gvad_enable(bool enable)
{
    uint8_t old_enable;

    cpu_critical_enter();
    old_enable = vad_env.gvad_enable;
    if (old_enable != (uint8_t)enable) {
        vad_env.gvad_ret = 0;
        vad_env.gvad_detect_pending = 0;
        /* No logging while interrupts are masked. Both enable edges start a
         * new confirmation window, without changing source ownership. */
        vad_env.gvad_front_samples = 0;
        vad_env.gvad_front_last_ms = 0;
        vad_env.gvad_enable = enable;
    }
    cpu_critical_exit();

    DBGLOG_AUD_DEV_INFO("[VAD] gvad enable %d->%d\n", old_enable, enable);
}

static void aud_sv_vad_mic_start_callback(uint32_t id, void *param)
{
    UNUSED(id);
    UNUSED(param);
    DBGLOG_LIB_AUD_SV_INFO("[VAD] mic start, id:%d\n", id);

    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    p_dmsg->evt = VAD_EVT_OPEN_DONE;
    (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL);
}

static void aud_sv_vad_mic_open_done(const wq_aud_res_mic_cb_param_t *param)
{
    uint64_t tgt_hus = 0;

    if (param->initiator == WQ_AUD_RES_MIC_USER_VAD) {
        if (param->is_triggered) {
            aud_sv_vad_mic_start_callback(param->timer_id, NULL);
        } else {
            tgt_hus = wq_audio_intf_get_timer_freerun_half_us((WQ_AUDIO_TIMER_ID)param->timer_id);
            tgt_hus = AUD_TIME_ADD_US(tgt_hus, AUD_SV_VAD_TRIGGER_DLY_MS * 1000u);
            aud_timer_cfg_t timer_start = {
                .timer_id = param->timer_id,
                .tgt_hus = tgt_hus,
            };
            vad_env.cur_save_pkt_num = 0;
            wq_aud_res_mic_aud_timer_start(&timer_start);
        }
    } else {
        if (param->is_triggered) {
            aud_sv_vad_mic_start_callback(param->timer_id, NULL);
        }
    }
}

static void aud_sv_vad_mic_close_done(const wq_aud_res_mic_cb_param_t *param)
{
    assert(param);

    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    p_dmsg->evt = VAD_EVT_CLOSE_DONE;
    (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL);
}

static void aud_sv_vad_mic_cb(const wq_aud_res_mic_cb_param_t *param)
{
    switch (param->event) {
    case WQ_AUD_RES_MIC_CB_EVT_CLOSED:
#if defined(CONFIG_AUDIO_VAD_WAKEUP_GVAD)
        wq_feat_res_stop(WQ_FEAT_RES_USR_ACORE_GVAD);
#endif
        aud_sv_vad_mic_close_done(param);
        break;
    case WQ_AUD_RES_MIC_CB_EVT_OPENED:
#if defined(CONFIG_AUDIO_VAD_WAKEUP_GVAD)
        wq_feat_res_start(WQ_FEAT_RES_USR_ACORE_GVAD, NULL);
#endif
        aud_sv_vad_mic_open_done(param);
        break;
    case WQ_AUD_RES_MIC_CB_EVT_TRIGGER:
        aud_sv_vad_mic_start_callback(param->timer_id, NULL);
        break;
    default:
        break;
    }

    DBGLOG_LIB_AUD_SV_INFO("[VAD] mic cb evt:%d tid:%d init:%d triggered:%d\n", param->event,
                           param->timer_id, param->initiator, param->is_triggered);
}

/**
 * @brief aud_sv_eq_report_ipc_hdl
 * @return uint8_t WQ_RET_OK for success else error
 */
/*lint -e{740} Unusual pointer cast (incompatible indirect types) */
/*lint -e{826} Suspicious pointer-to-pointer conversion (area too small) */
static uint32_t aud_sv_vad_ipc_hdl(WQ_CORES src_core, as_ipc_tr_msg_t *p_msg, uint16_t len)
{
    assert(src_core == WQ_CORES_DCORE);
    assert(p_msg != NULL && len != 0);

    aud_sv_ipc_vad_param_msg_t *vad_msg = (aud_sv_ipc_vad_param_msg_t *)p_msg;
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    assert(vad_msg->cmd < VAD_CMD_MAX);

    switch (vad_msg->cmd) {
    case VAD_CMD_KWS_ENABLE_ACK:
        DBGLOG_AUD_SV_INFO("[VAD] dcore kws enable ack seq:%d enable:%d ret:%d\n",
                           vad_msg->seq, vad_msg->param, vad_msg->result);
        p_dmsg->evt = VAD_EVT_RECORD_MONITOR_ACK;
        p_dmsg->ipc_seq = vad_msg->seq;
        p_dmsg->result = vad_msg->result;
        p_dmsg->enable = vad_msg->param;
        if (!aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL)) {
            return WQ_RET_FAIL;
        }
        break;
    case VAD_CMD_VOICE_HIT:
        DBGLOG_AUD_SV_INFO("[VAD] dcore voice hit status %d\n", vad_msg->param);
        p_dmsg->evt = VAD_EVT_VOICE_HIT;
        p_dmsg->hit_status = vad_msg->param;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        if (!aud_sv_vad_candidate_voice_receive(p_dmsg)) {
            return WQ_RET_OK;
        }
#endif
        if (!aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL)) {
            return WQ_RET_FAIL;
        }
        break;
    case VAD_CMD_KWS_HIT:
        DBGLOG_AUD_SV_INFO("[VAD] dcore kws hit status %d\n", vad_msg->param);
        p_dmsg->evt = VAD_EVT_KWS_HIT;
        p_dmsg->hit_status = vad_msg->param;
        if (!aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL)) {
            return WQ_RET_FAIL;
        }
        break;
    default:
        DBGLOG_AUD_SV_INFO("[VAD] dcore msg %d param %d not handler\n", vad_msg->cmd,
                           vad_msg->param);
        break;
    }

    return WQ_RET_OK;
}

static void aud_sv_vad_timer_handler(timer_id_t timer_id, void *arg)
{
    UNUSED(arg);

    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    if (timer_id == vad_env.retry_timer) {
        /* This runs in the FreeRTOS timer daemon. The auto-reload timer keeps
         * retrying without issuing a blocking timer command from its callback. */
        (void)aud_sv_vad_retry_post_one(false);
        aud_sv_vad_retry_timer_stop_no_wait(false);
        return;
    } else if (timer_id == vad_env.kws_timer) {
        uint32_t seq;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        vad_evt_t evt;
#endif
        TickType_t now = xTaskGetTickCount();

        cpu_critical_enter();
        if (!vad_env.kws_timer_armed
            || (int32_t)(now - vad_env.kws_timeout_deadline_tick) < 0) {
            cpu_critical_exit();
            return;
        }
        seq = vad_env.kws_timeout_seq;
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        evt = vad_env.kws_timer_event;
#endif
        vad_env.kws_timer_armed = 0;
        cpu_critical_exit();
#if defined(CONFIG_APP_VAD_CANDIDATE_POWEROPT_ENABLE)
        p_dmsg->evt = evt;
#else
        p_dmsg->evt = VAD_EVT_KWS_TIMEOUT;
#endif
        (void)aud_sv_vad_control_post_seq(p_dmsg, false, AUD_SV_VAD_RETRY_NO_WAIT, seq);
        return;
    }
#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    else if (timer_id == vad_env.gvad_timer) {
        TickType_t now = xTaskGetTickCount();

        cpu_critical_enter();
        if (!vad_env.gvad_timer_armed
            || (int32_t)(now - vad_env.gvad_timeout_deadline_tick) < 0) {
            cpu_critical_exit();
            return;
        }
        p_dmsg->gvad_window_seq = vad_env.gvad_window_seq;
        vad_env.gvad_timer_armed = 0;
        cpu_critical_exit();
        p_dmsg->evt = VAD_EVT_GVAD_TIMEOUT;
    }
#endif
    else {
        assert(0);
    }
    /* Queue-full control events are retained and retried by retry_timer. */
    (void)aud_sv_vad_control_post(p_dmsg, false, AUD_SV_VAD_RETRY_NO_WAIT);
}

static void aud_sv_vad_data_mic_hook(const uint8_t *p_data, uint32_t length)
    IRAM_TEXT(aud_sv_vad_data_mic_hook);
static void aud_sv_vad_data_mic_hook(const uint8_t *p_data, uint32_t length)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    bool gvad_enable;

    cpu_critical_enter();
    gvad_enable = vad_env.gvad_enable != 0;
    cpu_critical_exit();
    if (!gvad_enable) {
        if (!vad_env.p_mic->mic_stream_start) {
            /* Keep the pre-roll ring bounded without waking the VAD task or DCORE. */
            (void)aud_sv_vad_read_ring_pkt(length, false);
        }
        return;
    }
#endif

    p_dmsg->evt = VAD_EVT_MIC_PACKET;
    p_dmsg->data = p_data;
    p_dmsg->data_len = length;
    aud_sv_task_send_msg(AS_MSG_VAD, &msg, true, sizeof(*p_dmsg));
}

static void aud_sv_vad_evt_callback(AUD_SV_EVT_E evt, const aud_sv_evt_param_t *p_param)
{
    UNUSED(evt);
    UNUSED(p_param);

    switch (evt) {
    case AUD_SV_EVT_VOICE: {
        dp_msg_t msg = {0};
        vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

        DBGLOG_AUD_SV_INFO("[VAD] evt bts voice, sub_evt:%d\n", p_param->voice.sub_evt);
        switch (p_param->voice.sub_evt) {
#if defined(CONFIG_AUDIO_VAD_INBAND_RING_COEXIST)
        case AUD_SV_EVT_VOICE_CALL_ACTIVE:
#else
        case AUD_SV_EVT_VOICE_CALL_START:
#endif /* CONFIG_AUDIO_VAD_INBAND_RING_COEXIST */
            p_dmsg->evt = VAD_EVT_PEND;
            (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL);
            break;
        case AUD_SV_EVT_VOICE_CALL_END:
            p_dmsg->evt = VAD_EVT_RESUME;
            (void)aud_sv_vad_control_post(p_dmsg, in_irq(), AUD_SV_VAD_RETRY_NORMAL);
            break;
        default:
            break;
        }
    } break;
    default:
        break;
    }
}

WQ_RET aud_sv_vad_init(void)
{
    uint32_t event_bitmap = AUD_SV_ANC_MGR_EVENT_MUSIC_START | AUD_SV_ANC_MGR_EVENT_MUSIC_STOP
        | AUD_SV_ANC_MGR_EVENT_VOICE_START | AUD_SV_ANC_MGR_EVENT_VOICE_STOP;

    wq_aud_sv_anc_mgr_anc_feature_register_event_cb(WQ_AUD_SV_VAD_SWITCH_ANC, event_bitmap,
                                                    aud_sv_vad_feat_event_cb);

    aud_sv_task_msg_handler_register(AS_MSG_VAD, aud_sv_vad_msg_hdl);

    wq_aud_res_mic_register_cb(WQ_AUD_RES_MIC_USER_VAD, aud_sv_vad_mic_cb);

    // register ipc
    as_ipc_tr_regst(AS_IPC_TR_MSG_ID_VAD_STATUS, aud_sv_vad_ipc_hdl);

    vad_env.p_anc = aud_sv_stream_get_sh_info(SM_ANC);
    assert(vad_env.p_anc);
    vad_env.p_mic = aud_sv_stream_get_sh_info(SM_MIC);
    assert(vad_env.p_mic);
    vad_env.p_anc->bit_width = CONFIG_AUD_RES_MIC_BIT_WIDTH;

#if defined(CONFIG_AUDIO_VAD_WAKEUP_TVAD)
    vad_env.gvad_timer = os_create_timer(OS_TIMER_MID, false, aud_sv_vad_timer_handler, NULL);
    assert(vad_env.gvad_timer);
#endif
    vad_env.kws_timer = os_create_timer(OS_TIMER_MID, false, aud_sv_vad_timer_handler, NULL);
    assert(vad_env.kws_timer);
    vad_env.retry_timer = os_create_timer(OS_TIMER_MID, true, aud_sv_vad_timer_handler, NULL);
    assert(vad_env.retry_timer);

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_detect_restart(uint32_t reserved)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    DBGLOG_LIB_AUD_SV_INFO("[VAD] restart reson:%d\n", reserved);
    p_dmsg->evt = VAD_EVT_RESTART;
    if (!aud_sv_vad_control_post(p_dmsg, false, AUD_SV_VAD_RETRY_NORMAL)) {
        return WQ_RET_BUSY;
    }

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_record_monitor_set(bool enable)
{
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    p_dmsg->evt = VAD_EVT_RECORD_MONITOR;
    p_dmsg->enable = enable;
    DBGLOG_LIB_AUD_SV_INFO("[VAD] record monitor request:%d\n", enable);
    if (!aud_sv_vad_control_post_durable(p_dmsg)) {
        return WQ_RET_BUSY;
    }

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_start(void)
{
    uint32_t seq = aud_sv_vad_event_begin(VAD_EVT_OPEN);
    dp_msg_t msg = {0};
    vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);

    DBGLOG_LIB_AUD_SV_INFO("[VAD] start state:%s seq:%d\n", st_str[vad_env.state], seq);
    p_dmsg->evt = VAD_EVT_OPEN;
    if (!aud_sv_vad_control_post_seq(p_dmsg, false, AUD_SV_VAD_RETRY_NORMAL, seq)) {
        return WQ_RET_BUSY;
    }

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_stop(void)
{
    uint32_t seq = aud_sv_vad_event_begin(VAD_EVT_CLOSE);

    if (vad_env.state > VAD_ST_IDLE) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] stop seq:%d\n", seq);

        dp_msg_t msg = {0};
        vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);
        p_dmsg->evt = VAD_EVT_CLOSE;
        if (!aud_sv_vad_control_post_seq(p_dmsg, false, AUD_SV_VAD_RETRY_NORMAL, seq)) {
            return WQ_RET_BUSY;
        }
    } else {
        aud_sv_vad_retry_timer_stop_if_idle(false);
        DBGLOG_LIB_AUD_SV_INFO("[VAD] stop already idle seq:%d\n", seq);
        return WQ_RET_AGAIN;
    }

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_quit(wq_aud_vad_close_done_cb cb)
{
    uint32_t seq = aud_sv_vad_event_begin(VAD_EVT_CLOSE);

    vad_env.close_done_cb = cb;

    if (vad_env.state > VAD_ST_IDLE) {
        DBGLOG_LIB_AUD_SV_INFO("[VAD] close seq:%d\n", seq);

        dp_msg_t msg = {0};
        vad_msg_t *p_dmsg = (vad_msg_t *)(&msg.data[0]);
        p_dmsg->evt = VAD_EVT_CLOSE;
        if (!aud_sv_vad_control_post_seq(p_dmsg, false, AUD_SV_VAD_RETRY_NORMAL, seq)) {
            return WQ_RET_BUSY;
        }
    } else {
        aud_sv_vad_retry_timer_stop_if_idle(false);
        if (vad_env.close_done_cb) {
            vad_env.close_done_cb();
        }
    }

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_voice_hit_cb_register(wq_aud_vad_voice_hit_cb cb)
{
    vad_env.voice_cb = cb;

    return WQ_RET_OK;
}

WQ_RET aud_sv_vad_kws_hit_cb_register(wq_aud_vad_kws_hit_cb cb)
{
    vad_env.kws_cb = cb;

    return WQ_RET_OK;
}
#endif
