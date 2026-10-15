/****************************************************************************

Copyright(c) 2022 by WuQi Technologies. ALL RIGHTS RESERVED.

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
#include "types.h"
#include "stdio.h"

#include "os_utils.h"
#include "os_task.h"
#include "os_hook.h"
#include "cpu_usage.h"
#include "generic_transmission.h"
#include "dbglog.h"
#include "wq_share_task.h"
#include "wq_wdt.h"
#include "wq_uart.h"
#include "wq_rtc.h"
#include "rpc_service.h"
#include "ipc_service.h"
#include "irq.h"
#include "tws_app.h"
#include "m_datapath.h"
#include "wq_debug.h"
#include "wq_timer.h"
#include "feat_res_mgnt.h"
#include "perf_monitor.h"
#include "power_mgnt.h"
#include "boot_reason.h"
#if defined(CONFIG_EXE_LOGGING_ENABLE)
#include "exe_logging.h"
#include "wq_debug.h"
#endif
#if defined(CONFIG_PROFILE_ENABLE)
#include "profiling.h"
#endif

#if defined(CONFIG_EXE_LOGGING_ENABLE)
#define WQ_EXELOGGING_TYPE_TASK          0
#define WQ_EXELOGGING_TYPE_ISR_TIMER     1
#define WQ_EXELOGGING_TYPE_ISR_SOFTWARE  2
#define WQ_EXELOGGING_TYPE_ISR_EXTERNAL  3
#define WQ_EXELOGGING_TYPE_ISR_EXCEPTION 4

static bool app_should_dump_exe_logging(void)
{
    BOOT_REASON_TYPE boot_reason = BOOT_REASON_UNKNOWN;
    uint8_t boot_source = 0;
    uint8_t boot_flag = 0;

    if (wq_boot_reason_get_reason(&boot_reason, &boot_source, &boot_flag) != WQ_RET_OK) {
        return false;
    }

    UNUSED(boot_source);
    UNUSED(boot_flag);

    return (boot_reason == BOOT_REASON_SOFT || boot_reason == BOOT_REASON_WDT
            || boot_reason == BOOT_REASON_CPU || boot_reason == BOOT_REASON_DEBOUNCE);
}

static void app_dump_exe_logging_print(uint16_t index)
{
    char *name = NULL;
    uint32_t tick = 0;
    uint16_t index_count = 0;
    DBGLOG_STREAM_INFO(0, "[exelogging] It's dcore preserved stream log\n");

    for (uint16_t i = index; i < (index + CONFIG_EXE_LOGGING_MAX_ITEMS); i++) {
        uint8_t res =
            wq_debug_exe_logging_record_context((i % CONFIG_EXE_LOGGING_MAX_ITEMS), &tick, &name);
        os_delay(20);
        if (res == WQ_EXELOGGING_TYPE_TASK) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu task-name=%s\n", index_count, tick, name);
        } else if (res == WQ_EXELOGGING_TYPE_ISR_TIMER) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu isr timer\n", index_count, tick);
        } else if (res == WQ_EXELOGGING_TYPE_ISR_SOFTWARE) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu isr sw\n", index_count, tick);
        } else if (res == WQ_EXELOGGING_TYPE_ISR_EXTERNAL) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu isr external\n", index_count, tick);
        } else if (res == WQ_EXELOGGING_TYPE_ISR_EXCEPTION) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu isr exc\n", index_count, tick);
        } else if (tick != 0) {
            DBGLOG_INFO(0, "[exelogging] idx=%d tick=%lu unknown\n", index_count, tick);
        }
        index_count++;
    }
}
#endif

void app_entry(void *arg);

static const task_priority_t task_list[] = {
    {"share_task_fast", 8},
    {"rpc_task",        8},
    {"share_task_slow", 3},
};

static void wdt_idle_hook_cb(void)
{
    if (wq_wdt_need_feed()) {
        /* can't use mutex as dbglog in idle */
        wq_wdt_do_feed();
    }
}

/*lint -esym(714, app_entry) not referenced */
void app_entry(void *arg)
{
    UNUSED(arg);
    os_task_init(task_list, ARRAY_SIZE(task_list));
    wq_share_task_init();
#if CONFIG_ENABLE_CORE_NUM > 1
    wq_generic_transmission_init();
    wq_dbglog_init();
#endif

#if !(defined(CONFIG_BUILD_SERIES_BBB) || defined(CONFIG_BUILD_SERIES_SCARAB))
    wq_timer_init();
#endif
    wq_wdt_enable();
    os_hook_idle_register_callback(wdt_idle_hook_cb);
    // need by cpu usage
    wq_rtc_init(true);
    os_hook_time_stamp_register_callback(wq_rtc_get_time);
    rpc_srv_init();
    ipc_srv_init(IPC_SRV_CH_MAX);
    wq_dma_init();
    wq_debug_init();
    wq_feat_res_init();
#if defined(CONFIG_AUDIO_VAD_ENABLE)
    /* Silero HIL 自检: golden 音频逐窗概率 [SHIL] 输出(测试钩子, 生产可移除)
       注意: 必须在 app_main_entry 之前——它启动调度后不返回
       CCCv3 起停用: HIL 独占 dcore 数秒, 与 factory_mode 持久化的开机自启 VAD
       (flash 里 VAD gate=1 时上电即开麦起流)竞态——acore record_start 的
       m_ring 握手在 dcore 初始化读指针前读到垃圾值 → m_ring.c:985 断言崩溃循环。
       数值验证已由 CCC/CCCv2 完成(0/40 翻转); 需复验时临时打开此开关,
       并确保板上 VAD gate 持久化为 0 */
#if 0
    {
        extern void wq_silero_hil_selftest(void);
        wq_silero_hil_selftest();
    }
#endif
#endif
    app_main_entry();
    wq_perf_monitor_enable(PERF_MALLOC_FAIL_ASSERT | PERF_CACHE_MISS);

#if defined(CONFIG_PROFILE_ENABLE)
    wq_power_mgnt_keep_active();
    profile_set_frequency(16000);
    profile_enable();
#endif
#if defined(CONFIG_EXE_LOGGING_ENABLE)
    if (app_should_dump_exe_logging()) {
        uint16_t index = wq_debug_exe_logging_record_stop();
        app_dump_exe_logging_print(index);
    }
    wq_debug_exe_logging_record_start();
    DBGLOG_INFO(0, "[exelogging] dcore exe logging started, dump on next reset boot\n");
#endif
}
