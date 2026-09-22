/*
 * wq_gcc_shim.c —— GCC 构建垫片: 补齐 XCC 预编译库引用的运行时符号
 *
 * 背景: audio_algorithm/lib 下的预编译库(wq_kws/wq_common/wq_enc_* 等)由 Xtensa XCC
 * 编译, 引用 XCC 运行时符号(_Assert/__recipsf2); 公开 GCC 工具链的 newlib 也缺
 * __ieee754_sqrtf。用 GCC 编译固件时需提供以下垫片(xt-clang/xcc 构建不需要本文件)。
 */
#include <stdint.h>

/* XCC assert() 的运行时入口(仅满足链接, 触发时挂起便于调试器定位) */
void _Assert(const char *expr, ...)
{
    (void)expr;
    volatile int hang = 1;
    while (hang) {
    }
}

/* XCC 浮点除法展开用的倒数近似种子; 用精确倒数替代(调用方牛顿迭代仍收敛) */
float __recipsf2(float x)
{
    return 1.0f / x;
}

/* 该 newlib 构建缺失的 sqrtf 后端: 借 double 版实现(舍入到 float 等价精确) */
extern double __ieee754_sqrt(double x);
float __ieee754_sqrtf(float x)
{
    return (float)__ieee754_sqrt((double)x);
}
