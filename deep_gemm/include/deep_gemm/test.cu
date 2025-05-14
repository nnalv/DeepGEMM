#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

#define ELEMENTS_PER_THREAD 1
#define NUM_THREADS 4 
#define TOTAL_ELEMENTS (ELEMENTS_PER_THREAD * NUM_THREADS)  

__global__ void fp8_to_bf16_kernel() {
    __shared__ __nv_fp8_e4m3 smem_b[TOTAL_ELEMENTS];        
    __shared__ __nv_bfloat16 smem_b_tmp[TOTAL_ELEMENTS];    
    const uint8_t fp8_hex_values[TOTAL_ELEMENTS] = {0x01, 0x02, 0x03, 0x04};
    int idx = threadIdx.x * ELEMENTS_PER_THREAD;

    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        smem_b[idx + i] = __nv_fp8_e4m3(fp8_hex_values[idx + i]);
    }

    __syncthreads();

    int start_idx = threadIdx.x * ELEMENTS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = start_idx + i;
        auto fp_value = (float)smem_b[idx];
        smem_b_tmp[idx] = __float2bfloat16(fp_value);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Converted BF16 values:\n");
        for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
            printf("%d: FP8(hex: 0x%02x) -> Float: %f -> BF16: %f\n", 
                   i, static_cast<uint8_t>(smem_b[i]), 
                   (float)smem_b[i], 
                   __bfloat162float(smem_b_tmp[i]));
        }
    }
}

int main() {
    fp8_to_bf16_kernel<<<1, NUM_THREADS>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}