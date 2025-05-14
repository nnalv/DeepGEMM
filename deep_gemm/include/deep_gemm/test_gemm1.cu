#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

template <int N>
__device__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// GMMA Descriptor
union GmmaDescriptor {
    __host__ __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept : desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept : desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        uint16_t start_address_ : 14, : 2;
        uint16_t leading_byte_offset_ : 14, : 2;
        uint16_t stride_byte_offset_ : 14, : 2;
        uint8_t : 1, base_offset_ : 3, : 4;
        uint8_t : 6, layout_type_ : 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

// Helper function to create a shared memory descriptor
template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(PointerType smem_ptr, int layout_type,
                                         int leading_byte_offset = 0,
                                         int stride_byte_offset = 1024) {
    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = (desc.bitfield.start_address_ >> 0x7) & 0x7;
    return desc;
}

// GMMA kernel structure
struct SM90_64x16x16_F32BF16BF16_SS {
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 float& d00, float& d01, float& d02, float& d03,
                                 float& d04, float& d05, float& d06, float& d07,
                                 bool scale_d) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %10, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"
                     " {%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},"
                     " %8,"
                     " %9,"
                     " p   , 1,    1,  0, 0;\n"
                     "}\n"
                     : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
                       "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07)
                     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        wgmma(desc_a, desc_b,
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

// Kernel to perform the matrix multiplication
__global__ void test_matrix_mult() {
    constexpr int M = SM90_64x16x16_F32BF16BF16_SS::M;
    constexpr int N = SM90_64x16x16_F32BF16BF16_SS::N;
    constexpr int K = SM90_64x16x16_F32BF16BF16_SS::K;

    // Allocate shared memory for matrices
    __shared__ __nv_bfloat16 A[M][K];
    __shared__ __nv_bfloat16 B[K][N];
    __shared__ float D[M][N];

    // Initialize matrix A (64x16, all 1s)
    // printf("Initialize matrix A\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < M * K) {
        int row = tid / K;
        int col = tid % K;
        A[row][col] = __nv_bfloat16(1.0f);
    }

    // Initialize matrix B (16x16)
    // printf("Initialize matrix B\n");
    if (tid < K * N) {
        int row = tid / N;
        int col = tid % N;
        B[row][col] = __nv_bfloat16(1.0f * (col + 1)); // Column values increment
    }

    // Zero initialize matrix D
    // printf("Initialize matrix D\n");
    if (tid < M * N) {
        int row = tid / N;
        int col = tid % N;
        D[row][col] = 0.0f;
    }

    __syncthreads();

    // printf("make_smem_desc\n");
    // Create descriptors for A and B
    GmmaDescriptor desc_a = make_smem_desc(&A[0][0], /*layout_type=*/0, /*leading_byte_offset=*/16 * sizeof(__nv_bfloat16));
    GmmaDescriptor desc_b = make_smem_desc(&B[0][0], /*layout_type=*/0, /*leading_byte_offset=*/16 * sizeof(__nv_bfloat16));

    // Perform the matrix multiplication
    float accum[8] = {0}; // Accumulators for the 64x16x16 tile

    // printf("wgmma\n");
    SM90_64x16x16_F32BF16BF16_SS::wgmma(desc_a, desc_b, accum, false);

    printf("syncwarp\n");
    __syncwarp();

    printf("Write results back\n");
    // Write results back to D
    if (tid < M * N) {
        int row = tid / N;
        int col = tid % N;
        D[row][col] = accum[col / 8]; // Each warp handles part of the result
    }

    __syncthreads();

    printf("Print results\n");
    // Print the results (only from one thread to avoid messy output)
    printf("Matrix D (64x16):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
             printf("%f ", D[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // printf("size of bf16:%ld",sizeof(__nv_bfloat16));
    // Launch kernel
    test_matrix_mult<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}