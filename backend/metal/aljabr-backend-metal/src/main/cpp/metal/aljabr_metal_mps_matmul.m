/**
 * aljabr_metal_mps_matmul.m — public MPS matrix multiplication kernels.
 */

#import "aljabr_metal_buffers.h"
#import "aljabr_metal_cpu_fallback.h"
#import "aljabr_metal_mps_cache.h"
#import "aljabr_metal_mps_matmul.h"
#import "aljabr_metal_support.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>

static const int DEFAULT_SMALL_MATMUL_CPU_THRESHOLD = 1024;
static int g_small_matmul_cpu_threshold = DEFAULT_SMALL_MATMUL_CPU_THRESHOLD;

void aljabr_metal_mps_matmul_configure_from_env(void) {
  const char *threshold_env =
      getenv("ALJABR_METAL_SMALL_MATMUL_CPU_THRESHOLD");
  if (threshold_env != NULL && threshold_env[0] != '\0') {
    int parsed = atoi(threshold_env);
    if (parsed >= 0) {
      g_small_matmul_cpu_threshold = parsed;
    }
  }
}

static BOOL resolve_half_weight_type(int is_bf16, MPSDataType *b_mps_type) {
  if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
    if (@available(macOS 14.0, iOS 17.0, *)) {
      *b_mps_type = MPSDataTypeBFloat16;
      return YES;
    }
    return NO;
#else
    return NO;
#endif
  }
  *b_mps_type = MPSDataTypeFloat16;
  return YES;
}

static int wait_for_mps_command(id<MTLCommandBuffer> cmd) {
  [cmd commit];
  [cmd waitUntilCompleted];
  return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
}

int aljabr_metal_matmul(void *C, const void *A, const void *B, int M,
                             int K, int N, float alpha, float beta) {
  if (!g_initialized)
    return -1;

  if (M * N < g_small_matmul_cpu_threshold) {
    return aljabr_metal_cpu_matmul_nn(C, A, B, M, K, N, alpha, beta);
  }

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)K * N * sizeof(float));

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB =
        cached_matrix_descriptor(K, N, N * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descC =
        cached_matrix_descriptor(M, N, N * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mmul =
        cached_mmul(NO, NO, M, N, K, MPSDataTypeFloat32, MPSDataTypeFloat32,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB
                   resultMatrix:matC];
    return wait_for_mps_command(cmd);
  }
}

int aljabr_metal_matmul_tb(void *C, const void *A, const void *B, int M,
                                int K, int N, float alpha, float beta) {
  if (!g_initialized)
    return -1;

  if (M * N < g_small_matmul_cpu_threshold) {
    return aljabr_metal_cpu_matmul_tb(C, A, B, M, K, N, alpha, beta);
  }

  // Guard: MPS hangs indefinitely on very large F32 weight matrices (e.g.
  // lm_head with vocab_size=152064 and hidden=7168 ~= 1.09B elements).
  // Fall back to CPU Accelerate for safety when weight exceeds ~1 GB.
  static const long MAX_MPS_F32_WEIGHT_ELEMENTS = 256L * 1024L * 1024L;
  if ((long)N * K > MAX_MPS_F32_WEIGHT_ELEMENTS) {
    return aljabr_metal_cpu_matmul_tb(C, A, B, M, K, N, alpha, beta);
  }

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * sizeof(float));

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB =
        cached_matrix_descriptor(N, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descC =
        cached_matrix_descriptor(M, N, N * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mmul =
        cached_mmul(NO, YES, M, N, K, MPSDataTypeFloat32, MPSDataTypeFloat32,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB
                   resultMatrix:matC];
    return wait_for_mps_command(cmd);
  }
}

int aljabr_metal_matmul_tb_half(void *C, const void *A, const void *B,
                                     int M, int K, int N, float alpha,
                                     float beta, int is_bf16) {
  if (!g_initialized)
    return -1;

  @autoreleasepool {
    MPSDataType b_mps_type;
    if (!resolve_half_weight_type(is_bf16, &b_mps_type)) {
      return -2;
    }
    int b_elem_size = 2;

    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * b_elem_size);

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB =
        cached_matrix_descriptor(N, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descC =
        cached_matrix_descriptor(M, N, N * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mmul =
        cached_mmul(NO, YES, M, N, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB
                   resultMatrix:matC];
    return wait_for_mps_command(cmd);
  }
}

int aljabr_metal_matmul_tb_half_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int M,
                                          int K, int N, float alpha, float beta,
                                          int is_bf16) {
  if (!g_initialized)
    return -1;

  @autoreleasepool {
    MPSDataType b_mps_type;
    if (!resolve_half_weight_type(is_bf16, &b_mps_type)) {
      return -2;
    }
    int b_elem_size = 2;

    id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N * sizeof(float));
    id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB0 = wrap_weight_ptr(B0, (size_t)N * K * b_elem_size);
    id<MTLBuffer> bufB1 = wrap_weight_ptr(B1, (size_t)N * K * b_elem_size);

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB =
        cached_matrix_descriptor(N, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descC =
        cached_matrix_descriptor(M, N, N * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0
                                              descriptor:descB];
    MPSMatrix *matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1
                                              descriptor:descB];
    MPSMatrix *matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0
                                              descriptor:descC];
    MPSMatrix *matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1
                                              descriptor:descC];

    MPSMatrixMultiplication *mmul =
        cached_mmul(NO, YES, M, N, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB0
                   resultMatrix:matC0];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB1
                   resultMatrix:matC1];
    return wait_for_mps_command(cmd);
  }
}

int aljabr_metal_matmul_tb_half_pair_mixed(void *C0, void *C1,
                                                const void *A, const void *B0,
                                                const void *B1, int M, int K,
                                                int N0, int N1, float alpha,
                                                float beta, int is_bf16) {
  if (!g_initialized)
    return -1;
  if (M <= 0 || K <= 0 || N0 <= 0 || N1 <= 0)
    return -2;

  @autoreleasepool {
    MPSDataType b_mps_type;
    if (!resolve_half_weight_type(is_bf16, &b_mps_type)) {
      return -2;
    }
    int b_elem_size = 2;

    id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N0 * sizeof(float));
    id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N1 * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB0 = wrap_weight_ptr(B0, (size_t)N0 * K * b_elem_size);
    id<MTLBuffer> bufB1 = wrap_weight_ptr(B1, (size_t)N1 * K * b_elem_size);
    if (bufC0 == nil || bufC1 == nil || bufA == nil || bufB0 == nil ||
        bufB1 == nil) {
      return -3;
    }

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB0 =
        cached_matrix_descriptor(N0, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descB1 =
        cached_matrix_descriptor(N1, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descC0 =
        cached_matrix_descriptor(M, N0, N0 * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descC1 =
        cached_matrix_descriptor(M, N1, N1 * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0
                                              descriptor:descB0];
    MPSMatrix *matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1
                                              descriptor:descB1];
    MPSMatrix *matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0
                                              descriptor:descC0];
    MPSMatrix *matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1
                                              descriptor:descC1];

    MPSMatrixMultiplication *mmul0 =
        cached_mmul(NO, YES, M, N0, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);
    MPSMatrixMultiplication *mmul1 =
        cached_mmul(NO, YES, M, N1, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul0 encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB0
                    resultMatrix:matC0];
    [mmul1 encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB1
                    resultMatrix:matC1];
    return wait_for_mps_command(cmd);
  }
}

int aljabr_metal_matmul_tb_half_triple_mixed(
    void *C0, void *C1, void *C2, const void *A, const void *B0, const void *B1,
    const void *B2, int M, int K, int N0, int N1, int N2, float alpha,
    float beta, int is_bf16) {
  if (!g_initialized)
    return -1;
  if (M <= 0 || K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0)
    return -2;

  @autoreleasepool {
    MPSDataType b_mps_type;
    if (!resolve_half_weight_type(is_bf16, &b_mps_type)) {
      return -2;
    }
    int b_elem_size = 2;

    id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N0 * sizeof(float));
    id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N1 * sizeof(float));
    id<MTLBuffer> bufC2 = wrap_ptr(C2, (size_t)M * N2 * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)M * K * sizeof(float));
    id<MTLBuffer> bufB0 = wrap_weight_ptr(B0, (size_t)N0 * K * b_elem_size);
    id<MTLBuffer> bufB1 = wrap_weight_ptr(B1, (size_t)N1 * K * b_elem_size);
    id<MTLBuffer> bufB2 = wrap_weight_ptr(B2, (size_t)N2 * K * b_elem_size);
    if (bufC0 == nil || bufC1 == nil || bufC2 == nil || bufA == nil ||
        bufB0 == nil || bufB1 == nil || bufB2 == nil) {
      return -3;
    }

    MPSMatrixDescriptor *descA =
        cached_matrix_descriptor(M, K, K * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descB0 =
        cached_matrix_descriptor(N0, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descB1 =
        cached_matrix_descriptor(N1, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descB2 =
        cached_matrix_descriptor(N2, K, K * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descC0 =
        cached_matrix_descriptor(M, N0, N0 * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descC1 =
        cached_matrix_descriptor(M, N1, N1 * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descC2 =
        cached_matrix_descriptor(M, N2, N2 * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0
                                              descriptor:descB0];
    MPSMatrix *matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1
                                              descriptor:descB1];
    MPSMatrix *matB2 = [[MPSMatrix alloc] initWithBuffer:bufB2
                                              descriptor:descB2];
    MPSMatrix *matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0
                                              descriptor:descC0];
    MPSMatrix *matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1
                                              descriptor:descC1];
    MPSMatrix *matC2 = [[MPSMatrix alloc] initWithBuffer:bufC2
                                              descriptor:descC2];

    MPSMatrixMultiplication *mmul0 =
        cached_mmul(NO, YES, M, N0, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);
    MPSMatrixMultiplication *mmul1 =
        cached_mmul(NO, YES, M, N1, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);
    MPSMatrixMultiplication *mmul2 =
        cached_mmul(NO, YES, M, N2, K, MPSDataTypeFloat32, b_mps_type,
                    MPSDataTypeFloat32, alpha, beta);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul0 encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB0
                    resultMatrix:matC0];
    [mmul1 encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB1
                    resultMatrix:matC1];
    [mmul2 encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB2
                    resultMatrix:matC2];
    return wait_for_mps_command(cmd);
  }
}
