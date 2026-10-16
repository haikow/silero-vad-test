/****************************************************************************

Copyright(c) 2023 by WuQi Technologies. ALL RIGHTS RESERVED.

This Information is proprietary to WuQi Technologies and MAY NOT
be copied by any method or incorporated into another program without
the express written consent of WuQi. This Information or any portion
thereof remains the property of WuQi. The Information contained herein
is believed to be accurate and WuQi assumes no responsibility or
liability for its use in any way and conveys no license or title under
any patent or copyright and makes no representation or warranty that this
Information is free from patent or copyright infringement.

****************************************************************************/
/*lint -e766 header file not used */
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "cpu.h"
#include "os_lock.h"
#include "wq_heap.h"
#include "os_timer.h"
#include "audio_alg_dbglog.h"
#include "sw_vad.h"
#include "preset_mem.h"
#include "mic_stream.h"
#include "wq_pool.h"
#include "aud_sv_vad.h"
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
#include "vad_record_preroll.h"
#endif
#if defined(CONFIG_AUDIO_VAD_ENABLE)

// #define SW_VAD_DEBUG

struct sw_vad_handler {
    wq_gvad_hit_cb cb;
    char *hd;
    char *scratch;
    char *pool;
    uint8_t output;
    uint8_t cur_hit_cnt;
    uint8_t cur_stop_cnt;
    uint8_t continue_hit_cnt;
    uint8_t continue_stop_cnt;
    bool is_preset_mem;
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
    bool record_history_enable;
#endif
#if defined(CONFIG_APP_VAD_RECORD_OUTPUT_PRUNE_ENABLE)
    const bool *pcm_output_enabled;
    uint32_t pruned_frames;
    uint32_t pruned_bytes;
#endif
};

static char sw_vad_name[] = "sw_vad";

//lint -e843 Pointer variable could be declared as pointing to const
static WQ_RET sw_vad_query(void *param, wq_pipeline_processor_info_t *processor_info)
{
    assert(param);
    assert(processor_info);

    sw_vad_config_t *config = (sw_vad_config_t *)param;

    processor_info->scratch_size = (uint32_t)wq_get_sw_vad_scratch_size();
    if (preset_mem_allow_query(config->use_preset_mem)) {
        processor_info->persistent_size += sizeof(struct sw_vad_handler) + 4;
        processor_info->persistent_size += (uint32_t)wq_get_sw_vad_hd_size() + 16;
#if (CONFIG_VAD_SAVE_KWS_DATA != AUDIO_VAD_SAVE_DATA_BY_MALLOC)
        processor_info->persistent_size += (uint32_t)WQ_SW_VAD_POOL_LEN + 16;
#endif
    }

    return WQ_RET_OK;
}

static void *sw_vad_probe(void *param, wq_pipeline_processor_preset_t *preset)
{
    UNUSED(preset);
    assert(param);

    sw_vad_config_t *config = (sw_vad_config_t *)param;
    struct sw_vad_handler *handler;
    handler = pipeline_mem_aligned_malloc(preset->stream_id, sizeof(struct sw_vad_handler), 4,
                                          config->use_preset_mem);
    assert(handler);

    int hd_size = wq_get_sw_vad_hd_size();
    char *hd = pipeline_mem_aligned_malloc(preset->stream_id, hd_size, 16, config->use_preset_mem);
    assert(hd);
    memset(hd, 0, (size_t)hd_size);
    handler->hd = hd;
    handler->is_preset_mem = config->use_preset_mem;
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
    handler->record_history_enable = config->record_history_enable;
#endif
#if defined(CONFIG_APP_VAD_RECORD_OUTPUT_PRUNE_ENABLE)
    handler->pcm_output_enabled = config->pcm_output_enabled;
    handler->pruned_frames = 0;
    handler->pruned_bytes = 0;
    if (handler->pcm_output_enabled) {
        DBGLOG_AUDIO_ALG_INFO("[VAD] VG6 output prune armed, history preserved\n");
    }
#endif
    int scratch_size = wq_get_sw_vad_scratch_size();

#if defined(CONFIG_SCRATCH_MEM_ENABLE)
    handler->scratch = preset->scratch;
#else
    char *scratch = wq_heap_caps_malloc((size_t)scratch_size, MALLOC_CAP_ALIGN_16BYTE);
    assert(scratch);
    handler->scratch = scratch;
#endif
    int32_t pool_size;

    if (config->pool_addr && config->pool_len) {
        handler->pool = (char *)config->pool_addr;
        pool_size = (int32_t)config->pool_len;
    } else {
#if (CONFIG_VAD_SAVE_KWS_DATA != AUDIO_VAD_SAVE_DATA_BY_MALLOC)
        pool_size = WQ_SW_VAD_POOL_LEN;

        handler->pool =
            pipeline_mem_aligned_malloc(preset->stream_id, pool_size, 16, config->use_preset_mem);
#else
        pool_size = 0;
#endif
    }

    assert(handler->pool);

    wq_pool_init((wq_pool_t *)handler->pool, (uint32_t)(pool_size - sizeof(wq_pool_t)));

    handler->output = 0;
    handler->cb = config->cb;
    handler->continue_hit_cnt = config->continue_hit_cnt;
    if (handler->continue_hit_cnt == 0) {
        handler->continue_hit_cnt = 1;
    }
    handler->cur_hit_cnt = 0;
    handler->continue_stop_cnt = config->continue_stop_cnt;
    if (handler->continue_stop_cnt == 0) {
        handler->continue_stop_cnt = 6;
    }
    handler->cur_stop_cnt = handler->continue_stop_cnt + 1;
    wq_sw_vad_init(hd, handler->scratch);
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
    if (handler->record_history_enable) {
        vad_record_preroll_source_open(handler, (const wq_pool_t *)handler->pool);
    }
#endif
    DBGLOG_AUDIO_ALG_INFO("wq_sw_vad_init hd:%d scratch:%d hit_cnt:%d stop_cnt:%d pool:%d %x\n",
                          hd_size, scratch_size, handler->continue_hit_cnt,
                          handler->continue_stop_cnt, pool_size, handler->pool);

    return handler;
} /*lint !e818 !e844 function do not declared as pointing to const, due to pipeline interface */

/* Silero 会话级功耗: 流销毁时释放 192M DFS 申请(v7)。原厂库无此调用,
   Silero 实现内部为幂等空操作(未持有时不动), 对其它实现零影响 */
extern void wq_sw_vad_exit(void);

static void sw_vad_remove(void *h)
{
    struct sw_vad_handler *handler = (struct sw_vad_handler *)h;

    wq_sw_vad_exit();

#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
    if (handler->record_history_enable) {
        vad_record_preroll_source_close(handler);
    }
#endif
#if defined(CONFIG_APP_VAD_RECORD_OUTPUT_PRUNE_ENABLE)
    if (handler->pcm_output_enabled) {
        DBGLOG_AUDIO_ALG_INFO("[VAD] VG6 output prune frames:%u bytes:%u\n",
                              handler->pruned_frames, handler->pruned_bytes);
    }
#endif

    pipeline_mem_aligned_free(handler->hd, handler->is_preset_mem);
#if !defined(CONFIG_SCRATCH_MEM_ENABLE)
    wq_heap_caps_free(handler->scratch);
#endif
#if (CONFIG_VAD_SAVE_KWS_DATA != AUDIO_VAD_SAVE_DATA_BY_MALLOC)
    pipeline_mem_aligned_free(handler->pool, handler->is_preset_mem);
#endif
    pipeline_mem_aligned_free(handler, handler->is_preset_mem);

    DBGLOG_AUDIO_ALG_INFO("wq_sw_vad_exit...\n");
}

static WQ_RET sw_vad_update(void *h, void *param)
{
    struct sw_vad_handler *handler = (struct sw_vad_handler *)h;
    sw_vad_update_param_t *vad_param = (sw_vad_update_param_t *)param;

    DBGLOG_AUDIO_ALG_INFO("wq_sw_vad_update cmd:%d\n", vad_param->cmd);
    switch (vad_param->cmd) {
    case SW_VAD_CMD_RESET:
        handler->output = 0;
        handler->cur_hit_cnt = 0;
        handler->cur_stop_cnt = handler->continue_stop_cnt + 1;
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
        if (handler->record_history_enable) {
            vad_record_preroll_source_reset(handler);
        }
#endif
        break;
    default:
        break;
    }

    return WQ_RET_OK;
}

/*lint !e818 !e844 function do not declared as pointing to const, due to pipeline interface */
static WQ_RET sw_vad_run(void *h, wq_plb_t *in, wq_plb_t **out)
{
    struct sw_vad_handler *handler = (struct sw_vad_handler *)h;
    wq_pool_t *pool = (wq_pool_t *)handler->pool;
    int16_t *pcm_data = (int16_t *)in->data;

    if (wq_plb_get_channels(in) > 1) {
#if defined(CONFIG_RECORD_BASE_DEBUG_BF_DATA)
        pcm_data = (int16_t *)mic_stream_data_ptr_get(MIC_TYPE_VOICE_2_L, in);
#else
        pcm_data = (int16_t *)mic_stream_data_ptr_get(MIC_TYPE_VOICE_0_L, in);
#endif

        in->channel = WQ_PIPELINE_CHANNEL_MIX;
        in->channel_gap = 0;
    }

    int vad_result = wq_sw_vad_process(handler->hd, (char *)pcm_data, in->len);
    if (vad_result == 1) {
        if (handler->cur_hit_cnt == (handler->continue_hit_cnt - 1)) {
            if (handler->cb) {
                handler->cb(1);
                handler->output = 1;
                DBGLOG_AUDIO_ALG_INFO("wq_sw_vad_process hit, read %d write %d size %d used %d\n",
                                      pool->read, pool->write, pool->size, pool->remain);
            }
            handler->cur_hit_cnt++;
            handler->cur_stop_cnt = 0;
        } else if (handler->cur_hit_cnt < (handler->continue_hit_cnt - 1)) {
            handler->cur_hit_cnt++;
        } else {   // handler->cur_hit_cnt > handler->continue_hit_cnt
            handler->cur_stop_cnt = 0;
        }
    } else {
        if (handler->cur_stop_cnt == (handler->continue_stop_cnt - 1)) {
            if (handler->cb) {
                handler->cb(0);
            }
            handler->cur_stop_cnt++;
            handler->cur_hit_cnt = 0;
        } else if (handler->cur_stop_cnt < (handler->continue_stop_cnt - 1)) {
            handler->cur_stop_cnt++;
        } else {   // handler->cur_stop_cnt > handler->continue_stop_cnt
            handler->cur_hit_cnt = 0;
        }
    }
#ifdef SW_VAD_DEBUG
    DBGLOG_AUDIO_ALG_INFO("wq_sw_vad_proc len %d vad_result %d hit_cnt %d stop_cnt %d\n", in->len,
                          vad_result, handler->cur_hit_cnt, handler->cur_stop_cnt);
#endif

#if defined(CONFIG_APP_VAD_RECORD_OUTPUT_PRUNE_ENABLE)
    if (handler->output && handler->pcm_output_enabled && !*handler->pcm_output_enabled) {
        uint32_t read_len = in->len;
        /* Match wq_read_pool's state changes without copying unused PCM.
         * Keep the following write and the classifier state unchanged so
         * re-enabling KWS observes the same history as the legacy path. */
        if (pool->remain >= read_len) {
            pool->remain -= read_len;
            pool->read += read_len;
            if (pool->read >= pool->size) {
                pool->read %= pool->size;
            }
            if (handler->pruned_bytes <= UINT32_MAX - read_len) {
                handler->pruned_bytes += read_len;
            } else {
                handler->pruned_bytes = UINT32_MAX;
            }
        }
        if (handler->pruned_frames < UINT32_MAX) {
            handler->pruned_frames++;
        }
        *out = NULL;
    } else
#endif
    if (handler->output) {
        uint32_t read_len = in->len;
        wq_plb_t *new = wq_plb_alloc(read_len, WQ_PIPELINE_BUFF_DEFAULT_PAD_SIZE);
        assert(new);
        new->ts = in->ts;
        new->ts_sec = in->ts_sec;
        new->format = WQ_PIPELINE_DATA_FORMAT_PCM;
        new->channel = WQ_PIPELINE_CHANNEL_MIX;
        new->bit_width = WQ_PIPELINE_SAMPLE_WIDTH_16BIT;

        new->flags = in->flags;
        new->flags &= ~WQ_PIPELINE_BUFF_FLAG_READ_ONLY;

        wq_read_pool(pool, (char *)new->data, read_len);
#ifdef SW_VAD_DEBUG
        DBGLOG_AUDIO_ALG_INFO("wq_sw_vad read pool, read %d write %d used %d\n", pool->read,
                              pool->write, pool->remain);
#endif

        new->sample_rate = in->sample_rate;
        new->sp = in->sp;
        new->len = read_len;
        *out = new;
    } else {
        *out = NULL;
    }

    wq_write_pool(pool, NULL, (char *)pcm_data, in->len);
#if defined(CONFIG_APP_VAD_RECORD_PREROLL_ENABLE)
    if (handler->record_history_enable) {
        vad_record_preroll_source_written(handler, in, handler->output != 0);
    }
#endif
#ifdef SW_VAD_DEBUG
    DBGLOG_AUDIO_ALG_INFO("wq_sw_vad write pool, read %d write %d used %d\n", pool->read,
                          pool->write, pool->remain);
#endif

    wq_plb_free(in);

    return WQ_RET_OK;
}

//lint -e714 not referenced
const wq_pipeline_processor_t sw_vad_processor DRAM_RODATA(sw_vad_processor) = {
    .name = sw_vad_name,
    .query = sw_vad_query,
    .probe = sw_vad_probe,
    .remove = sw_vad_remove,
    .reset = NULL,
    .update = sw_vad_update,
    .run = sw_vad_run,
};

#endif
