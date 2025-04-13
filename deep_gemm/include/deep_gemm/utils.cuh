#pragma once

#include <exception>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#ifdef __CLION_IDE__
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) { asm volatile("trap;"); }
#define printf host_device_printf
#endif

class AssertionException : public std::exception {
private:
    std::string message{};

public:
    explicit AssertionException(const std::string& message) : message(message) {}

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond)                                        \
do {                                                                \
    if (not (cond)) {                                               \
        printf("Assertion failed: %s:%d, condition: %s\n",          \
               __FILE__, __LINE__, #cond);                          \
        throw AssertionException("Assertion failed: " #cond);       \
    }                                                               \
} while (0)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                          \
do {                                                                                    \
    if (not (cond)) {                                                                   \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);  \
        asm("trap;");                                                                   \
    }                                                                                   \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

__host__ __device__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_e4m3 fp8_value) {
    // FP8 格式的常量
    const int exponent_bits = 4;  // E4M3 的指数位数
    const int mantissa_bits = 3; // E4M3 的尾数位数
    const int bias = (1 << (exponent_bits - 1)) - 1; // E4M3 格式的偏置值为 7

    // 将 FP8 值解释为 uint8_t
    uint8_t fp8_bits = *reinterpret_cast<uint8_t*>(&fp8_value);

    // 分解 FP8 值
    int sign = (fp8_bits >> 7) & 0x1;                                   // 符号位
    int exponent = ((fp8_bits >> mantissa_bits) & ((1 << exponent_bits) - 1)) - bias; // 指数位
    int mantissa = fp8_bits & ((1 << mantissa_bits) - 1);              // 尾数位

    // 特殊情况：处理 FP8 的 0 和非规格化数
    if (exponent < -bias) {
        return __nv_bfloat16(0.0f); // 非规格化数和 0 转为 0
    }

    // 特殊情况：处理 FP8 的无穷大和 NaN
    if (exponent == (1 << exponent_bits) - 1) {
        if (mantissa == 0) {
            // 无穷大
            return __nv_bfloat16(sign ? -INFINITY : INFINITY);
        } else {
            // NaN
            return __nv_bfloat16(NAN);
        }
    }

    // 正常情况：将 FP8 转换为 BF16
    exponent = exponent + 127; // 将 FP8 指数调整为 BF16 的偏置（127）

    // 构造 BF16 的二进制表示
    uint16_t bf16_value = (sign << 15) | (exponent << 7) | (mantissa << (7 - mantissa_bits));
    return *reinterpret_cast<__nv_bfloat16*>(&bf16_value);
}

__host__ __device__ float fp8_to_float(uint8_t fp8_value, bool e4m3 = true) {
    // 确定 FP8 格式的指数和尾数位数
    const int exponent_bits = e4m3 ? 4 : 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int bias = (1 << (exponent_bits - 1)) - 1; // E4M3: 7, E5M2: 15

    // 分解 FP8 值
    int sign = (fp8_value >> 7) & 0x1;
    int exponent = ((fp8_value >> mantissa_bits) & ((1 << exponent_bits) - 1)) - bias;
    int mantissa = fp8_value & ((1 << mantissa_bits) - 1);

    // 还原为 FP32 格式
    uint32_t fp32_value = (sign << 31) | ((exponent + 127) << 23) | (mantissa << (23 - mantissa_bits));
    return *reinterpret_cast<float*>(&fp32_value);
}