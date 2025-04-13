#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdint.h>

// 假设 kNumStages 是一个已知的常量
#define kNumStages 4

// FP8 的转换辅助函数
__host__ __device__ uint8_t float_to_fp8(float value, bool e4m3 = true) {
    // 确定 FP8 格式的指数和尾数位数
    const int exponent_bits = e4m3 ? 4 : 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int bias = (1 << (exponent_bits - 1)) - 1; // E4M3: 7, E5M2: 15

    // 将浮点数分解为符号、尾数和指数
    uint32_t int_value = *reinterpret_cast<uint32_t*>(&value);
    int sign = (int_value >> 31) & 0x1;                    // 符号位
    int exponent = ((int_value >> 23) & 0xFF) - 127;       // 去掉 FP32 的偏移量
    int mantissa = (int_value >> (23 - mantissa_bits)) & ((1 << mantissa_bits) - 1); // 截取尾数

    // 处理特殊情况（0 或非规格化数）
    if (exponent < -bias) {
        return 0; // FP8 下溢为 0
    }

    // 处理溢出（超过 FP8 的表示范围）
    if (exponent > bias) {
        return (sign << 7) | (0xF << mantissa_bits); // FP8 上溢为无穷大
    }

    // 重新计算 FP8 的指数
    exponent = exponent + bias;

    // 构造 FP8 值
    uint8_t fp8_value = (sign << 7) | (exponent << mantissa_bits) | mantissa;
    return fp8_value;
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

// __host__ __device__ 函数：将 FP8 转换为 BF16
__host__ __device__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_e4m3 fp8_value) {
    const int exponent_bits = 4;  // E4M3 的指数位数
    const int mantissa_bits = 3; // E4M3 的尾数位数
    const int bias = (1 << (exponent_bits - 1)) - 1; // 偏置值为 7

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
            return __nv_bfloat16(sign ? -INFINITY : INFINITY); // 无穷大
        } else {
            return __nv_bfloat16(NAN); // NaN
        }
    }

    // 正常情况：将 FP8 转换为 BF16
    exponent = exponent + 127; // 将 FP8 指数调整为 BF16 的偏置（127）

    // 构造 BF16 的二进制表示
    uint16_t bf16_value = (sign << 15) | (exponent << 7) | (mantissa << (7 - mantissa_bits));
    return *reinterpret_cast<__nv_bfloat16*>(&bf16_value);
}

__global__ void convert_fp8_to_bf16(__nv_fp8_e4m3* smem_a, __nv_bfloat16* smem_b, int num_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 检查线程索引是否越界
    if (idx < num_elements) {
        // 从 smem_a 读取 FP8 数据并转换为 BF16
        __nv_fp8_e4m3 fp8_value = smem_a[idx];
        // __nv_bfloat16 bf16_value = fp8_to_bf16(fp8_value);
        __nv_bfloat16 bf16_value = (__nv_bfloat16)(fp8_value);

        // 将转换后的 BF16 数据存储到 smem_b
        smem_b[idx] = bf16_value;
    }
}

int main() {
    const int num_elements = 256; // 数据长度

    // 主机内存分配
    __nv_fp8_e4m3* h_smem_a = new __nv_fp8_e4m3[num_elements];
    __nv_bfloat16* h_smem_b = new __nv_bfloat16[num_elements];

    // 初始化 FP8 数据
    for (int i = 0; i < num_elements; ++i) {
        float value = static_cast<float>(i % 256) / 256.0f; // 范围 [0, 1)
        uint8_t fp8_bits = float_to_fp8(value); // 浮点数转换为 FP8
        h_smem_a[i] = *reinterpret_cast<__nv_fp8_e4m3*>(&fp8_bits);
    }

    // 设备内存分配
    __nv_fp8_e4m3* d_smem_a;
    __nv_bfloat16* d_smem_b;
    cudaMalloc(&d_smem_a, num_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_smem_b, num_elements * sizeof(__nv_bfloat16));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_smem_a, h_smem_a, num_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // 配置 CUDA 内核
    int threads_per_block = 128;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // 调用 CUDA 核函数
    convert_fp8_to_bf16<<<blocks_per_grid, threads_per_block>>>(d_smem_a, d_smem_b, num_elements);

    // 同步设备
    cudaDeviceSynchronize();

    // 将结果从设备拷贝回主机
    cudaMemcpy(h_smem_b, d_smem_b, num_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // 输出前 10 个结果
    for (int i = 0; i < 10; ++i) {
        uint8_t fp8_bits = *reinterpret_cast<uint8_t*>(&h_smem_a[i]);  // 获取 FP8 的位表示
        // float original_value = fp8_to_float(fp8_bits);   
        float original_value = (float)h_smem_a[i];              // FP8 转浮点数
        float converted_value = __bfloat162float(h_smem_b[i]);         // BF16 转浮点数
        std::cout << "FP8: " << original_value << " -> BF16: " << converted_value << std::endl;
    }

    // 清理资源
    delete[] h_smem_a;
    delete[] h_smem_b;
    cudaFree(d_smem_a);
    cudaFree(d_smem_b);

    return 0;
}