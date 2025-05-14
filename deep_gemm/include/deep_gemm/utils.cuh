#pragma once

#include <exception>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

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

__device__ static void e4m3x2_to_bf16x2(__nv_fp8_e4m3 x1, __nv_fp8_e4m3 x2, __nv_bfloat16& y1, __nv_bfloat16& y2) {
    uint16_t src_packed = (static_cast<uint16_t>(x2) << 8) | static_cast<uint16_t>(x1);

    uint32_t res_half;
    asm volatile(
        "{\n"
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n"
        "}\n"
        : "=r"(res_half)  
        : "h"(src_packed) 
    );

    float2 res_float = __half22float2(reinterpret_cast<__half2&>(res_half));

    uint16_t bf16_1, bf16_2;
    asm volatile(
        "{\n"
        "cvt.rn.bf16.f32 %0, %1;\n"
        "cvt.rn.bf16.f32 %2, %3;\n"
        "}\n"
        : "=h"(bf16_1), "=h"(bf16_2) 
        : "f"(res_float.x), "f"(res_float.y) 
    );

    y1 = reinterpret_cast<__nv_bfloat16&>(bf16_1);
    y2 = reinterpret_cast<__nv_bfloat16&>(bf16_2);
}