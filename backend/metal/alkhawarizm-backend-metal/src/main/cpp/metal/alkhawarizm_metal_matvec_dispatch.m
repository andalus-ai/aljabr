/**
 * alkhawarizm_metal_matvec_dispatch.m — low-level custom matvec Metal dispatch.
 */

#import "alkhawarizm_metal_buffers.h"
#import "alkhawarizm_metal_matvec_dispatch.h"
#import "alkhawarizm_metal_pipelines.h"
#import "alkhawarizm_metal_support.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

static BOOL is_bf16_pair_x4_pipeline(id<MTLComputePipelineState> pipeline) {
  AlkhawarizmMetalPipelines *pipelines = alkhawarizm_metal_pipelines();
  return pipeline == pipelines->matvec_bf16_pair_x4 ||
         pipeline == pipelines->matvec_bf16_pair_x4_simd;
}

int alkhawarizm_metal_dispatch_matvec_tb_half(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = is_bf16_pair_x4_pipeline(pipeline)
                            ? (((NSUInteger)N + 3u) / 4u)
                            : (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_int4(
    void *C, const void *A, const void *B, const void *scales, int K, int N,
    int blockSize, id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)((N * K + 1) / 2));
    id<MTLBuffer> bufScales =
        wrap_weight_ptr(scales, (size_t)((N * K / blockSize) * sizeof(float)));
    if (bufC == nil || bufA == nil || bufB == nil || bufScales == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    uint32_t bs = (uint32_t)blockSize;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBuffer:bufScales offset:0 atIndex:3];
    [enc setBytes:&kk length:sizeof(kk) atIndex:4];
    [enc setBytes:&nn length:sizeof(nn) atIndex:5];
    [enc setBytes:&bs length:sizeof(bs) atIndex:6];

    NSUInteger groups = (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_q4_k(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    // Q4_K block is 144 bytes for 256 elements
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K / 256 * 144);
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_q6_k(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    // Q6_K block is 210 bytes for 256 elements
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K / 256 * 210);
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_q8_0(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    // Q8_0 block is 34 bytes for 32 elements
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K / 32 * 34);
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_nf4(
    void *C, const void *A, const void *B_packed, const void *absmax, int K,
    int N, int blockSize, id<MTLComputePipelineState> pipeline,
    NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B_packed, (size_t)((N * K + 1) / 2));
    id<MTLBuffer> bufAbsmax =
        wrap_weight_ptr(absmax, (size_t)((N * K / blockSize) * sizeof(float)));
    if (bufC == nil || bufA == nil || bufB == nil || bufAbsmax == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    uint32_t bs = (uint32_t)blockSize;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBuffer:bufAbsmax offset:0 atIndex:3];
    [enc setBytes:&kk length:sizeof(kk) atIndex:4];
    [enc setBytes:&nn length:sizeof(nn) atIndex:5];
    [enc setBytes:&bs length:sizeof(bs) atIndex:6];

    NSUInteger groups = (NSUInteger)N;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_half_x4(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = (((NSUInteger)N + 3u) / 4u);
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_half_x8(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    NSUInteger groups = (((NSUInteger)N + 7u) / 8u);
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_t_half(
    void *C, const void *A, const void *B, int K, int N,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB = wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufB == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&kk length:sizeof(kk) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_half_pair(
    void *C0, void *C1, const void *A, const void *B0, const void *B1, int K,
    int N, id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)N * sizeof(float));
    id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB0 = wrap_weight_ptr(B0, (size_t)N * K * sizeof(uint16_t));
    id<MTLBuffer> bufB1 = wrap_weight_ptr(B1, (size_t)N * K * sizeof(uint16_t));
    if (bufC0 == nil || bufC1 == nil || bufA == nil || bufB0 == nil ||
        bufB1 == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t nn = (uint32_t)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC0 offset:0 atIndex:0];
    [enc setBuffer:bufC1 offset:0 atIndex:1];
    [enc setBuffer:bufA offset:0 atIndex:2];
    [enc setBuffer:bufB0 offset:0 atIndex:3];
    [enc setBuffer:bufB1 offset:0 atIndex:4];
    [enc setBytes:&kk length:sizeof(kk) atIndex:5];
    [enc setBytes:&nn length:sizeof(nn) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int alkhawarizm_metal_dispatch_matvec_tb_half_triple_mixed(
    void *C0, void *C1, void *C2, const void *A, const void *B0, const void *B1,
    const void *B2, int K, int N0, int N1, int N2,
    id<MTLComputePipelineState> pipeline, NSUInteger threads) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0)
    return -2;
  if (pipeline == nil)
    return -3;
  if (pipeline.maxTotalThreadsPerThreadgroup < threads)
    return -3;

  @autoreleasepool {
    id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)N0 * sizeof(float));
    id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)N1 * sizeof(float));
    id<MTLBuffer> bufC2 = wrap_ptr(C2, (size_t)N2 * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)K * sizeof(float));
    id<MTLBuffer> bufB0 =
        wrap_weight_ptr(B0, (size_t)N0 * K * sizeof(uint16_t));
    id<MTLBuffer> bufB1 =
        wrap_weight_ptr(B1, (size_t)N1 * K * sizeof(uint16_t));
    id<MTLBuffer> bufB2 =
        wrap_weight_ptr(B2, (size_t)N2 * K * sizeof(uint16_t));
    if (bufC0 == nil || bufC1 == nil || bufC2 == nil || bufA == nil ||
        bufB0 == nil || bufB1 == nil || bufB2 == nil) {
      return -4;
    }

    uint32_t kk = (uint32_t)K;
    uint32_t n0 = (uint32_t)N0;
    uint32_t n1 = (uint32_t)N1;
    uint32_t n2 = (uint32_t)N2;
    NSUInteger total = (NSUInteger)N0 + (NSUInteger)N1 + (NSUInteger)N2;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufC0 offset:0 atIndex:0];
    [enc setBuffer:bufC1 offset:0 atIndex:1];
    [enc setBuffer:bufC2 offset:0 atIndex:2];
    [enc setBuffer:bufA offset:0 atIndex:3];
    [enc setBuffer:bufB0 offset:0 atIndex:4];
    [enc setBuffer:bufB1 offset:0 atIndex:5];
    [enc setBuffer:bufB2 offset:0 atIndex:6];
    [enc setBytes:&kk length:sizeof(kk) atIndex:7];
    [enc setBytes:&n0 length:sizeof(n0) atIndex:8];
    [enc setBytes:&n1 length:sizeof(n1) atIndex:9];
    [enc setBytes:&n2 length:sizeof(n2) atIndex:10];

    NSUInteger groups =
        pipeline == alkhawarizm_metal_pipelines()->matvec_bf16_triple_mixed_x4
            ? ((total + 3u) / 4u)
            : total;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}
