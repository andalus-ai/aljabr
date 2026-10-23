/**
 * aljabr_metal_matvec_api.m — exported Metal matvec C API for Aljabr.
 *
 * Optimized for reduced memory pressure and fixed KV cache layout.
 */

#import "aljabr_metal_matvec_api.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#import "aljabr_metal_attention.h"
#import "aljabr_metal_buffers.h"
#import "aljabr_metal_matvec_dispatch.h"
#import "aljabr_metal_matvec_tuning.h"
#import "aljabr_metal_mps_cache.h"
#import "aljabr_metal_mps_matvec_policy.h"
#import "aljabr_metal_mps_validation.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_scratch.h"
#import "aljabr_metal_support.h"

#define env_truthy aljabr_metal_env_truthy
#define env_int_or_default aljabr_metal_env_int_or_default
#define env_float_or_default aljabr_metal_env_float_or_default
#define monotonic_nanos aljabr_metal_monotonic_nanos

// ── Globals
// ───────────────────────────────────────────────────────────────────

#define g_pipelines (*aljabr_metal_pipelines())
#define g_matvec_half_pipeline g_pipelines.matvec_half
#define g_matvec_t_half_pipeline g_pipelines.matvec_t_half
#define g_matvec_tb_nf4_pipeline g_pipelines.matvec_tb_nf4
#define g_matvec_tb_nf4_128_pipeline g_pipelines.matvec_tb_nf4_128
#define g_matvec_half_pair_pipeline g_pipelines.matvec_half_pair
#define g_matvec_half_triple_mixed_pipeline g_pipelines.matvec_half_triple_mixed
#define g_matvec_bf16_pipeline g_pipelines.matvec_bf16
#define g_matvec_bf16_pair_pipeline g_pipelines.matvec_bf16_pair
#define g_matvec_bf16_pair_simd_pipeline g_pipelines.matvec_bf16_pair_simd
#define g_matvec_bf16_triple_mixed_pipeline g_pipelines.matvec_bf16_triple_mixed
#define g_matvec_bf16_triple_mixed_x4_pipeline                                 \
  g_pipelines.matvec_bf16_triple_mixed_x4
#define g_matvec_bf16_x4_pipeline g_pipelines.matvec_bf16_x4
#define g_matvec_bf16_x8_pipeline g_pipelines.matvec_bf16_x8
#define g_matvec_bf16_pair_x4_pipeline g_pipelines.matvec_bf16_pair_x4
#define g_matvec_bf16_x4_simd_pipeline g_pipelines.matvec_bf16_x4_simd
#define g_matvec_bf16_pair_x4_simd_pipeline g_pipelines.matvec_bf16_pair_x4_simd
#define g_matvec_half_gated_pair_pipeline g_pipelines.matvec_half_gated_pair
#define g_matvec_bf16_gated_pair_pipeline g_pipelines.matvec_bf16_gated_pair
#define g_matvec_half_128_pipeline g_pipelines.matvec_half_128
#define g_matvec_t_half_128_pipeline g_pipelines.matvec_t_half_128
#define g_matvec_half_pair_128_pipeline g_pipelines.matvec_half_pair_128
#define g_matvec_half_triple_mixed_128_pipeline                                \
  g_pipelines.matvec_half_triple_mixed_128
#define g_matvec_bf16_128_pipeline g_pipelines.matvec_bf16_128
#define g_matvec_bf16_pair_128_pipeline g_pipelines.matvec_bf16_pair_128
#define g_matvec_bf16_triple_mixed_128_pipeline                                \
  g_pipelines.matvec_bf16_triple_mixed_128
#define g_matvec_half_gated_pair_128_pipeline                                  \
  g_pipelines.matvec_half_gated_pair_128
#define g_matvec_bf16_gated_pair_128_pipeline                                  \
  g_pipelines.matvec_bf16_gated_pair_128
// ── aljabr_metal_softcap ──────────────────────────────────────────────────────
// GPU logit soft-capping: x[i] = cap * tanh(x[i] / cap)  (in-place)
// Uses aljabr_softcap_kernel with precise::tanh for correctness.
// On Gemma-4 12B, vocab=262144; 256 threads → 1024 threadgroups.
// ─────────────────────────────────────────────────────────────────────────────

int aljabr_metal_softcap(void *logits, int N, float cap) {
  if (!g_initialized) return -1;
  if (N <= 0 || cap <= 0.0f) return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->softcap == nil) return -3;

  size_t bytes = (size_t)N * sizeof(float);
  size_t page_size = (size_t)getpagesize();
  uintptr_t addr = (uintptr_t)logits;
  uintptr_t offset = addr & (page_size - 1);
  void *aligned_ptr = (void *)(addr - offset);
  size_t aligned_bytes = (bytes + offset + page_size - 1) & ~(page_size - 1);

  id<MTLBuffer> bufLogits = [g_device newBufferWithBytesNoCopy:aligned_ptr
                                                         length:aligned_bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
  if (bufLogits == nil) {
    fprintf(stderr, "[aljabr-metal] softcap failed to wrap ptr %p (len=%zu)\n", logits, bytes);
    return -4;
  }

  uint32_t uN = (uint32_t)N;

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipelines->softcap];
  [enc setBuffer:bufLogits   offset:offset atIndex:0];
  [enc setBytes:&cap   length:sizeof(cap) atIndex:1];
  [enc setBytes:&uN    length:sizeof(uN)  atIndex:2];
  // Dispatch with 256 threads per threadgroup → ceil(N/256) groups
  NSUInteger threads = MIN((NSUInteger)256, pipelines->softcap.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];

  if ([cmd status] != MTLCommandBufferStatusCompleted) return -5;
  return 0;
}

// ── Matvec C API ─────────────────────────────────────────────────────────────

static int aljabr_metal_matvec_tb_half_custom(void *C, const void *A,
                                                   const void *B, int K,
                                                   int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("tb", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_half_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_half_128_pipeline
                                                : g_matvec_half_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_half(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  if (aljabr_metal_mps_matvec_should_try(K, N)) {
    @autoreleasepool {
      NSString *shapeKey = mps_matvec_shape_key(K, N);
      BOOL validateEveryCall =
          aljabr_metal_mps_matvec_validate_every_call();
      BOOL autotune =
          aljabr_metal_mps_matvec_autotune_enabled_for_output(N);
      AljabrMetalMpsMatvecShapeState shapeState =
          aljabr_metal_mps_matvec_shape_state(shapeKey, validateEveryCall,
                                                   autotune);
      if (!shapeState.failed && !shapeState.custom_preferred) {
        uint64_t mpsStart = autotune ? monotonic_nanos() : 0;
        uint16_t *halfA =
            aljabr_metal_ensure_half_input_scratch((size_t)K);
        uint16_t *halfC =
            aljabr_metal_ensure_half_output_scratch((size_t)N);
        if (halfA != NULL && halfC != NULL) {
          const float *input = (const float *)A;
          for (int i = 0; i < K; i++) {
            halfA[i] = f32_to_f16_bits(input[i]);
          }
        }
        id<MTLBuffer> bufC =
            halfC != NULL ? wrap_ptr(halfC, (size_t)N * sizeof(uint16_t)) : nil;
        id<MTLBuffer> bufA =
            halfA != NULL ? wrap_ptr(halfA, (size_t)K * sizeof(uint16_t)) : nil;
        id<MTLBuffer> bufB =
            wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
        if (bufC != nil && bufA != nil && bufB != nil) {
          @try {
            MPSMatrixDescriptor *descB = cached_matrix_descriptor(
                N, K, K * sizeof(uint16_t), MPSDataTypeFloat16);
            MPSVectorDescriptor *descA =
                cached_vector_descriptor(K, MPSDataTypeFloat16);
            MPSVectorDescriptor *descC =
                cached_vector_descriptor(N, MPSDataTypeFloat16);
            MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB
                                                     descriptor:descB];
            MPSVector *vecA = [[MPSVector alloc] initWithBuffer:bufA
                                                     descriptor:descA];
            MPSVector *vecC = [[MPSVector alloc] initWithBuffer:bufC
                                                     descriptor:descC];
            MPSMatrixVectorMultiplication *mvec =
                cached_mvec(NO, N, K, 1.0, 0.0);
            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            [mvec encodeToCommandBuffer:cmd
                            inputMatrix:matB
                            inputVector:vecA
                           resultVector:vecC];
            [cmd commit];
            [cmd waitUntilCompleted];
            if ([cmd status] == MTLCommandBufferStatusCompleted) {
              float *output = (float *)C;
              for (int i = 0; i < N; i++) {
                output[i] = f16_to_f32(halfC[i]);
              }
              uint64_t mpsNanos = autotune ? (monotonic_nanos() - mpsStart) : 0;
              if (shapeState.validated ||
                  validate_mps_matvec_half_output((const float *)C,
                                                  (const float *)A,
                                                  (const uint16_t *)B, K, N)) {
                if (!shapeState.validated && !validateEveryCall &&
                    !env_truthy("ALJABR_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
                  aljabr_metal_mps_matvec_mark_validated(shapeKey);
                  if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                    fprintf(stderr,
                            "[aljabr-metal] MPS matvec validated shape "
                            "K=%d N=%d\n",
                            K, N);
                  }
                }
                if (autotune && !shapeState.mps_preferred) {
                  uint64_t customStart = monotonic_nanos();
                  int customRc =
                      aljabr_metal_matvec_tb_half_custom(C, A, B, K, N);
                  uint64_t customNanos = monotonic_nanos() - customStart;
                  if (customRc == 0) {
                    float margin = env_float_or_default(
                        "ALJABR_METAL_MPS_MATVEC_AUTOTUNE_MARGIN", 0.05f);
                    if (margin < 0.0f)
                      margin = 0.0f;
                    BOOL preferMps =
                        (double)mpsNanos <
                        ((double)customNanos * (1.0 - (double)margin));
                    aljabr_metal_mps_matvec_record_autotune_preference(
                        shapeKey, preferMps);
                    if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                      fprintf(
                          stderr,
                          "[aljabr-metal] MPS matvec autotune shape K=%d "
                          "N=%d mps=%.3fms custom=%.3fms selected=%s\n",
                          K, N, (double)mpsNanos / 1000000.0,
                          (double)customNanos / 1000000.0,
                          preferMps ? "mps" : "custom");
                    }
                    return 0;
                  }
                  aljabr_metal_mps_matvec_mark_mps_preferred(shapeKey);
                }
                return 0;
              }
              aljabr_metal_mps_matvec_mark_failed(shapeKey);
              if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                fprintf(stderr,
                        "[aljabr-metal] disabling MPS matvec for shape "
                        "K=%d N=%d; falling back to custom Metal reduction\n",
                        K, N);
              }
            } else {
              aljabr_metal_mps_matvec_mark_disable_after_failure();
              if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                NSString *message = cmd.error.localizedDescription;
                fprintf(stderr,
                        "[aljabr-metal] MPS matvec command failed (%s); "
                        "falling back to custom Metal reduction\n",
                        message != nil ? [message UTF8String]
                                       : "unknown error");
              }
            }
          } @catch (NSException *ex) {
            // Fall back to the custom reduction kernel; correctness is more
            // important than this fast path.
            aljabr_metal_mps_matvec_mark_disable_after_failure();
            if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
              NSString *reason = [ex reason];
              fprintf(stderr,
                      "[aljabr-metal] MPS matvec threw %s; falling back "
                      "to custom Metal reduction\n",
                      reason != nil ? [reason UTF8String]
                                    : "unknown exception");
            }
          }
        }
      }
    }
  }

  NSString *key = matvec_shape_key("tb", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_half_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_half_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 =
        dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_half_128_pipeline,
                                ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_half_pipeline,
                                        ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half(C, A, B, K, N,
                                       g_matvec_half_128_pipeline,
                                       ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_half_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_half_128_pipeline
                                                : g_matvec_half_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_half_mps(void *C, const void *A, const void *B,
                                         int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  @synchronized(aljabr_metal_mps_matvec_policy_lock()) {
    AljabrMetalMpsMatvecOverrideSnapshot snapshot =
        aljabr_metal_mps_matvec_force_shape(K, N);
    if (!env_truthy("ALJABR_METAL_VALIDATE_LOGITS_MPS_MATVEC") &&
        !env_truthy("ALJABR_METAL_MPS_MATVEC_VALIDATE_EVERY_CALL")) {
      aljabr_metal_mps_matvec_mark_validated(mps_matvec_shape_key(K, N));
    }
    int rc = aljabr_metal_matvec_tb_half(C, A, B, K, N);

    aljabr_metal_mps_matvec_restore_overrides(snapshot);
    return rc;
  }
}

#define g_matvec_tb_int4_pipeline g_pipelines.matvec_tb_int4
#define g_matvec_tb_int4_128_pipeline g_pipelines.matvec_tb_int4_128

int aljabr_metal_matvec_tb_int4(void *C, const void *A,
                                     const void *B_packed, const void *scales,
                                     int K, int N, int blockSize) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  NSString *key = matvec_shape_key("tb_int4", K, N, blockSize, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_tb_int4_128_pipeline != nil;

  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_tb_int4_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                        g_matvec_tb_int4_128_pipeline,
                                        ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;

    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                        g_matvec_tb_int4_pipeline,
                                        ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;

    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_int4", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                       g_matvec_tb_int4_128_pipeline,
                                       ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    return rc256; // both autotune paths failed; propagate error
  } else {
    if (threads == 0) {
      threads = default_matvec_threads(g_matvec_tb_int4_128_pipeline, K, N);
    }
    id<MTLComputePipelineState> pipeline =
        threads == ALJABR_MATVEC_THREADS_128
            ? g_matvec_tb_int4_128_pipeline
            : g_matvec_tb_int4_pipeline;
    return dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                   pipeline, threads);
  }
}

#define g_matvec_tb_q4_k_pipeline g_pipelines.matvec_tb_q4_k
#define g_matvec_tb_q6_k_pipeline g_pipelines.matvec_tb_q6_k
#define g_matvec_tb_q8_0_pipeline g_pipelines.matvec_tb_q8_0

int aljabr_metal_matvec_tb_q4_k(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_pipelines.matvec_tb_q4_k_mr4 != nil) {
    return aljabr_metal_dispatch_matvec_tb_q4_k_mr4(
        C, A, B, K, N, g_pipelines.matvec_tb_q4_k_mr4);
  }
  if (g_matvec_tb_q4_k_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q4_k(C, A, B, K, N, g_matvec_tb_q4_k_pipeline,
                                 ALJABR_MATVEC_THREADS_32);
}

int aljabr_metal_matvec_tb_q4_k_softcap(void *C, const void *A, const void *B,
                                             int K, int N, float cap) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (cap > 0.0f && g_pipelines.matvec_tb_q4_k_mr4_softcap != nil) {
    return aljabr_metal_dispatch_matvec_tb_q4_k_mr4_softcap(
        C, A, B, K, N, cap, g_pipelines.matvec_tb_q4_k_mr4_softcap);
  }
  int rc = aljabr_metal_matvec_tb_q4_k(C, A, B, K, N);
  if (rc != 0) return rc;
  if (cap > 0.0f) {
    return aljabr_metal_softcap(C, N, cap);
  }
  return 0;
}

int aljabr_metal_matvec_tb_q6_k(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_pipelines.matvec_tb_q6_k_mr4 != nil) {
    return aljabr_metal_dispatch_matvec_tb_q6_k_mr4(
        C, A, B, K, N, g_pipelines.matvec_tb_q6_k_mr4);
  }
  if (g_matvec_tb_q6_k_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q6_k(C, A, B, K, N, g_matvec_tb_q6_k_pipeline,
                                 ALJABR_MATVEC_THREADS_32);
}

int aljabr_metal_matvec_tb_q6_k_softcap(void *C, const void *A, const void *B,
                                             int K, int N, float cap) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (cap > 0.0f && g_pipelines.matvec_tb_q6_k_mr4_softcap != nil) {
    return aljabr_metal_dispatch_matvec_tb_q6_k_mr4_softcap(
        C, A, B, K, N, cap, g_pipelines.matvec_tb_q6_k_mr4_softcap);
  }
  int rc = aljabr_metal_matvec_tb_q6_k(C, A, B, K, N);
  if (rc != 0) return rc;
  if (cap > 0.0f) {
    return aljabr_metal_softcap(C, N, cap);
  }
  return 0;
}

int aljabr_metal_matvec_tb_q8_0(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_matvec_tb_q8_0_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q8_0(C, A, B, K, N, g_matvec_tb_q8_0_pipeline,
                                 ALJABR_MATVEC_THREADS_32);
}

int aljabr_metal_matvec_tb_nf4(void *C, const void *A,
                                    const void *B_packed, const void *absmax,
                                    int K, int N, int blockSize) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  NSString *key = matvec_shape_key("tb_nf4", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_tb_nf4_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_tb_nf4_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                       g_matvec_tb_nf4_128_pipeline,
                                       ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                       g_matvec_tb_nf4_pipeline,
                                       ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_nf4", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                      g_matvec_tb_nf4_128_pipeline,
                                      ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_tb_nf4_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_tb_nf4_128_pipeline
                                                : g_matvec_tb_nf4_pipeline;
  return dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                pipeline, threads);
}

int aljabr_metal_matvec_t_half(void *C, const void *A, const void *B,
                                    int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("t", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_t_half_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_t_half_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 =
        dispatch_matvec_t_half(C, A, B, K, N, g_matvec_t_half_128_pipeline,
                               ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_t_half(C, A, B, K, N, g_matvec_t_half_pipeline,
                                       ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("t", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_t_half(C, A, B, K, N,
                                      g_matvec_t_half_128_pipeline,
                                      ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_t_half_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_t_half_128_pipeline
                                                : g_matvec_t_half_pipeline;
  return dispatch_matvec_t_half(C, A, B, K, N, pipeline, threads);
}

static int aljabr_metal_matvec_tb_bf16_custom(void *C, const void *A,
                                                   const void *B, int K,
                                                   int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("tb_bf16", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_bf16_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_bf16_128_pipeline
                                                : g_matvec_bf16_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_bf16(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  if (should_use_bf16_matvec_x8(K, N) && g_matvec_bf16_x8_pipeline != nil) {
    return dispatch_matvec_tb_half_x8(C, A, B, K, N, g_matvec_bf16_x8_pipeline,
                                      ALJABR_MATVEC_THREADS_256);
  }
  if (should_use_simdgroup_reduction() && should_use_bf16_matvec_x4(K, N) &&
      g_matvec_bf16_x4_simd_pipeline != nil) {
    return dispatch_matvec_tb_half_x4(C, A, B, K, N,
                                      g_matvec_bf16_x4_simd_pipeline,
                                      ALJABR_MATVEC_THREADS_256);
  }
  if (should_use_bf16_matvec_x4(K, N) && g_matvec_bf16_x4_pipeline != nil) {
    return dispatch_matvec_tb_half_x4(C, A, B, K, N, g_matvec_bf16_x4_pipeline,
                                      ALJABR_MATVEC_THREADS_256);
  }

  if (aljabr_metal_mps_bf16_matvec_should_try(K, N)) {
    @autoreleasepool {
      NSString *shapeKey = [NSString stringWithFormat:@"bf16:%d:%d", K, N];
      BOOL validateEveryCall =
          aljabr_metal_mps_matvec_validate_every_call();
      BOOL autotune =
          aljabr_metal_mps_matvec_autotune_enabled_for_output(N);
      AljabrMetalMpsMatvecShapeState shapeState =
          aljabr_metal_mps_matvec_shape_state(shapeKey, validateEveryCall,
                                                   autotune);
      if (!shapeState.failed && !shapeState.custom_preferred) {
        MPSDataType bf16Type = MPSDataTypeFloat16;
        BOOL bf16MpsAvailable = NO;
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
        if (@available(macOS 14.0, iOS 17.0, *)) {
          bf16Type = MPSDataTypeBFloat16;
          bf16MpsAvailable = YES;
        }
#endif
        if (bf16MpsAvailable) {
          uint64_t mpsStart = autotune ? monotonic_nanos() : 0;
          uint16_t *bf16A =
              aljabr_metal_ensure_half_input_scratch((size_t)K);
          uint16_t *bf16C =
              aljabr_metal_ensure_half_output_scratch((size_t)N);
          if (bf16A != NULL && bf16C != NULL) {
            const float *input = (const float *)A;
            for (int i = 0; i < K; i++) {
              bf16A[i] = f32_to_bf16_bits_bridge(input[i]);
            }
          }
          id<MTLBuffer> bufC =
              bf16C != NULL ? wrap_ptr(bf16C, (size_t)N * sizeof(uint16_t))
                            : nil;
          id<MTLBuffer> bufA =
              bf16A != NULL ? wrap_ptr(bf16A, (size_t)K * sizeof(uint16_t))
                            : nil;
          id<MTLBuffer> bufB =
              wrap_weight_ptr(B, (size_t)N * K * sizeof(uint16_t));
          if (bufC != nil && bufA != nil && bufB != nil) {
            @try {
              MPSMatrixDescriptor *descB = cached_matrix_descriptor(
                  N, K, K * sizeof(uint16_t), bf16Type);
              MPSVectorDescriptor *descA =
                  cached_vector_descriptor(K, bf16Type);
              MPSVectorDescriptor *descC =
                  cached_vector_descriptor(N, bf16Type);
              MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB
                                                       descriptor:descB];
              MPSVector *vecA = [[MPSVector alloc] initWithBuffer:bufA
                                                       descriptor:descA];
              MPSVector *vecC = [[MPSVector alloc] initWithBuffer:bufC
                                                       descriptor:descC];
              MPSMatrixVectorMultiplication *mvec =
                  cached_mvec(NO, N, K, 1.0, 0.0);
              id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
              [mvec encodeToCommandBuffer:cmd
                              inputMatrix:matB
                              inputVector:vecA
                             resultVector:vecC];
              [cmd commit];
              [cmd waitUntilCompleted];
              if ([cmd status] == MTLCommandBufferStatusCompleted) {
                float *output = (float *)C;
                for (int i = 0; i < N; i++) {
                  output[i] = bf16_to_f32_bridge(bf16C[i]);
                }
                uint64_t mpsNanos =
                    autotune ? (monotonic_nanos() - mpsStart) : 0;
                if (shapeState.validated ||
                    validate_mps_matvec_bf16_output(
                        (const float *)C, (const float *)A, (const uint16_t *)B,
                        K, N)) {
                  if (!shapeState.validated && !validateEveryCall &&
                      !env_truthy(
                          "ALJABR_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
                    aljabr_metal_mps_matvec_mark_validated(shapeKey);
                    if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                      fprintf(stderr,
                              "[aljabr-metal] MPS BF16 matvec validated "
                              "shape K=%d N=%d\n",
                              K, N);
                    }
                  }
                  if (autotune && !shapeState.mps_preferred) {
                    uint64_t customStart = monotonic_nanos();
                    int customRc =
                        aljabr_metal_matvec_tb_bf16_custom(C, A, B, K, N);
                    uint64_t customNanos = monotonic_nanos() - customStart;
                    if (customRc == 0) {
                      float margin = env_float_or_default(
                          "ALJABR_METAL_MPS_MATVEC_AUTOTUNE_MARGIN",
                          0.05f);
                      if (margin < 0.0f)
                        margin = 0.0f;
                      BOOL preferMps =
                          (double)mpsNanos <
                          ((double)customNanos * (1.0 - (double)margin));
                      aljabr_metal_mps_matvec_record_autotune_preference(
                          shapeKey, preferMps);
                      if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                        fprintf(stderr,
                                "[aljabr-metal] MPS BF16 matvec autotune "
                                "shape K=%d N=%d mps=%.3fms custom=%.3fms "
                                "selected=%s\n",
                                K, N, (double)mpsNanos / 1000000.0,
                                (double)customNanos / 1000000.0,
                                preferMps ? "mps" : "custom");
                      }
                      return 0;
                    }
                    aljabr_metal_mps_matvec_mark_mps_preferred(shapeKey);
                  }
                  return 0;
                }
                aljabr_metal_mps_matvec_mark_failed(shapeKey);
                if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                  fprintf(
                      stderr,
                      "[aljabr-metal] disabling MPS BF16 matvec for shape "
                      "K=%d N=%d; falling back to custom Metal reduction\n",
                      K, N);
                }
              } else {
                aljabr_metal_mps_matvec_mark_disable_after_failure();
                if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                  NSString *message = cmd.error.localizedDescription;
                  fprintf(stderr,
                          "[aljabr-metal] MPS BF16 matvec command failed "
                          "(%s); falling back to custom Metal reduction\n",
                          message != nil ? [message UTF8String]
                                         : "unknown error");
                }
              }
            } @catch (NSException *ex) {
              aljabr_metal_mps_matvec_mark_disable_after_failure();
              if (env_truthy("ALJABR_METAL_MPS_MATVEC_DEBUG")) {
                NSString *reason = [ex reason];
                fprintf(stderr,
                        "[aljabr-metal] MPS BF16 matvec threw %s; falling "
                        "back to custom Metal reduction\n",
                        reason != nil ? [reason UTF8String]
                                      : "unknown exception");
              }
            }
          }
        }
      }
    }
  }

  NSString *key = matvec_shape_key("tb_bf16", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_bf16_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_bf16_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 =
        dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_bf16_128_pipeline,
                                ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_bf16_pipeline,
                                        ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_bf16", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half(C, A, B, K, N,
                                       g_matvec_bf16_128_pipeline,
                                       ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_bf16_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128 ? g_matvec_bf16_128_pipeline
                                                : g_matvec_bf16_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_half_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("pair", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_half_pair_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_half_pair_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_half_pair_128_pipeline,
                                             ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_half_pair_pipeline,
                                             ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("pair", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                            g_matvec_half_pair_128_pipeline,
                                            ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_half_pair_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128
          ? g_matvec_half_pair_128_pipeline
          : g_matvec_half_pair_pipeline;
  return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                      threads);
}

int aljabr_metal_matvec_tb_bf16_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  BOOL usePairX4 = !env_truthy("ALJABR_METAL_DISABLE_BF16_PAIR_X4") &&
                   g_matvec_bf16_pair_x4_pipeline != nil &&
                   (env_truthy("ALJABR_METAL_ENABLE_BF16_PAIR_X4") ||
                    should_use_bf16_matvec_x4(K, N));
  if (usePairX4) {
    id<MTLComputePipelineState> pipeline =
        should_use_simdgroup_reduction() &&
                g_matvec_bf16_pair_x4_simd_pipeline != nil
            ? g_matvec_bf16_pair_x4_simd_pipeline
            : g_matvec_bf16_pair_x4_pipeline;
    return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                        ALJABR_MATVEC_THREADS_256);
  }
  if (should_use_bf16_pair_simd_reduction(K, N) &&
      g_matvec_bf16_pair_simd_pipeline != nil) {
    return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                        g_matvec_bf16_pair_simd_pipeline,
                                        ALJABR_MATVEC_THREADS_256);
  }
  NSString *key = matvec_shape_key("pair_bf16", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_bf16_pair_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_bf16_pair_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_bf16_pair_128_pipeline,
                                             ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_bf16_pair_pipeline,
                                             ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("pair_bf16", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                            g_matvec_bf16_pair_128_pipeline,
                                            ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_bf16_pair_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128
          ? g_matvec_bf16_pair_128_pipeline
          : g_matvec_bf16_pair_pipeline;
  return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                      threads);
}

int aljabr_metal_matvec_tb_half_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0)
    return -2;
  int total = N0 + N1 + N2;
  NSString *key = matvec_shape_key("triple", K, N0, N1, N2);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_half_triple_mixed_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, total, can128) &&
      g_matvec_half_triple_mixed_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_half_triple_mixed_128_pipeline,
        ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_half_triple_mixed_pipeline, ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("triple", K, N0, N1, N2, nanos128, nanos256, threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_triple_mixed(
            C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
            g_matvec_half_triple_mixed_128_pipeline,
            ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_half_triple_mixed_128_pipeline, K,
                                     total);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128
          ? g_matvec_half_triple_mixed_128_pipeline
          : g_matvec_half_triple_mixed_pipeline;
  return dispatch_matvec_tb_half_triple_mixed(C0, C1, C2, A, B0, B1, B2, K, N0,
                                              N1, N2, pipeline, threads);
}

int aljabr_metal_matvec_tb_bf16_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0)
    return -2;
  int total = N0 + N1 + N2;
  if (should_use_bf16_matvec_x4(K, total) &&
      g_matvec_bf16_triple_mixed_x4_pipeline != nil) {
    return dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_bf16_triple_mixed_x4_pipeline, ALJABR_MATVEC_THREADS_256);
  }
  NSString *key = matvec_shape_key("triple_bf16", K, N0, N1, N2);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_bf16_triple_mixed_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, total, can128) &&
      g_matvec_bf16_triple_mixed_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_bf16_triple_mixed_128_pipeline,
        ALJABR_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_bf16_triple_mixed_pipeline, ALJABR_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALJABR_MATVEC_THREADS_128
                    : ALJABR_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("triple_bf16", K, N0, N1, N2, nanos128, nanos256,
                          threads);
      if (threads == ALJABR_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_triple_mixed(
            C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
            g_matvec_bf16_triple_mixed_128_pipeline,
            ALJABR_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
    if (rc128 == 0)
      return 0;
    return rc256;
  }
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_bf16_triple_mixed_128_pipeline, K,
                                     total);
  }
  id<MTLComputePipelineState> pipeline =
      threads == ALJABR_MATVEC_THREADS_128
          ? g_matvec_bf16_triple_mixed_128_pipeline
          : g_matvec_bf16_triple_mixed_pipeline;
  return dispatch_matvec_tb_half_triple_mixed(C0, C1, C2, A, B0, B1, B2, K, N0,
                                              N1, N2, pipeline, threads);
}

static BOOL encode_matvec_sub(
    id<MTLComputeCommandEncoder> enc,
    AljabrMetalPipelines *pipelines,
    id<MTLBuffer> bufC, id<MTLBuffer> bufA, const void *B,
    uint32_t kk, uint32_t nn, int type, uint32_t batchSize) {
  id<MTLComputePipelineState> p = nil;
  size_t bBytes = 0;
  BOOL is_mr4 = NO;
  BOOL is_b4 = NO;
  if (type == 12) { // Q4_K
    if (batchSize > 1 && pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
      p = pipelines->matvec_tb_q4_k_b4_mr4;
      is_b4 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      p = pipelines->matvec_tb_q4_k_mr4;
      is_mr4 = YES;
    } else {
      p = pipelines->matvec_tb_q4_k;
    }
    bBytes = (size_t)nn * kk / 256 * 144;
  } else if (type == 14) { // Q6_K
    if (batchSize > 1 && pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
      p = pipelines->matvec_tb_q6_k_b4_mr4;
      is_b4 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      p = pipelines->matvec_tb_q6_k_mr4;
      is_mr4 = YES;
    } else {
      p = pipelines->matvec_tb_q6_k;
    }
    bBytes = (size_t)nn * kk / 256 * 210;
  } else if (type == 8) { // Q8_0
    p = pipelines->matvec_tb_q8_0;
    bBytes = (size_t)nn * kk / 32 * 34;
  } else if (type == 1) { // F16
    p = pipelines->matvec_half;
    bBytes = (size_t)nn * kk * sizeof(uint16_t);
  } else if (type == 30 || type == 16) { // BF16
    p = pipelines->matvec_bf16;
    bBytes = (size_t)nn * kk * sizeof(uint16_t);
  }
  if (p == nil || bBytes == 0) return NO;

  size_t bOff = 0;
  id<MTLBuffer> bufB = wrap_weight_ptr_offset(B, bBytes, &bOff);
  if (bufC == nil || bufB == nil) return NO;

  [enc setComputePipelineState:p];
  [enc setBuffer:bufC offset:0 atIndex:0];
  [enc setBuffer:bufA offset:0 atIndex:1];
  [enc setBuffer:bufB offset:bOff atIndex:2];
  [enc setBytes:&kk length:sizeof(kk) atIndex:3];
  [enc setBytes:&nn length:sizeof(nn) atIndex:4];
  NSUInteger bSz = (batchSize > 0) ? (NSUInteger)batchSize : 1;
  if (is_b4) {
    uint32_t bSz32 = (uint32_t)bSz;
    [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nn + 3) / 4, (bSz + 1) / 2, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  } else if (is_mr4) {
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nn + 3) / 4, bSz, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  } else {
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nn, bSz, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }
  return YES;
}

static id<MTLBuffer> g_scratch_single_A = nil;
static id<MTLBuffer> g_scratch_single_C = nil;
static size_t g_scratch_single_A_bytes = 0;
static size_t g_scratch_single_C_bytes = 0;

static id<MTLBuffer> g_scratch_triple_A = nil;
static id<MTLBuffer> g_scratch_triple_C0 = nil;
static id<MTLBuffer> g_scratch_triple_C1 = nil;
static id<MTLBuffer> g_scratch_triple_C2 = nil;
static size_t g_scratch_triple_A_bytes = 0;
static size_t g_scratch_triple_C0_bytes = 0;
static size_t g_scratch_triple_C1_bytes = 0;
static size_t g_scratch_triple_C2_bytes = 0;

static inline id<MTLBuffer> ensure_shared_buf(id<MTLBuffer> __strong *slot, size_t *cap, size_t needed) {
  if (*slot == nil || *cap < needed) {
    size_t sz = (needed + 16383) & ~16383;
    *slot = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    *cap = sz;
  }
  return *slot;
}

int aljabr_metal_matvec_tb_batch(void *C, const void *A, const void *B,
                                 int K, int N, int type, int batchSize) {
  if (!g_initialized) return -1;
  if (K <= 0 || N <= 0 || batchSize <= 0) return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t aBytes = (size_t)batchSize * (size_t)K * sizeof(float);
    size_t cBytes = (size_t)batchSize * (size_t)N * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_single_A, &g_scratch_single_A_bytes, aBytes);
    id<MTLBuffer> bufC = ensure_shared_buf(&g_scratch_single_C, &g_scratch_single_C_bytes, cBytes);
    if (bufA == nil || bufC == nil) return -4;

    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;
    uint32_t bSz = (uint32_t)batchSize;

    if (!encode_matvec_sub(enc, pipelines, bufC, bufA, B, kk, (uint32_t)N, type, bSz)) {
      [enc endEncoding];
      return -3;
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] == MTLCommandBufferStatusCompleted) {
      memcpy(C, [bufC contents], cBytes);
      return 0;
    }
    return -1;
  }
}

int aljabr_metal_matvec_tb_pair(void *C0, void *C1, const void *A,
                                const void *B0, const void *B1, int K,
                                int N0, int N1, int type0, int type1) {
  if (!g_initialized) return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0) return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t aBytes = (size_t)K * sizeof(float);
    size_t c0Bytes = (size_t)N0 * sizeof(float);
    size_t c1Bytes = (size_t)N1 * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_triple_A, &g_scratch_triple_A_bytes, aBytes);
    id<MTLBuffer> bufC0 = ensure_shared_buf(&g_scratch_triple_C0, &g_scratch_triple_C0_bytes, c0Bytes);
    id<MTLBuffer> bufC1 = ensure_shared_buf(&g_scratch_triple_C1, &g_scratch_triple_C1_bytes, c1Bytes);
    if (bufA == nil || bufC0 == nil || bufC1 == nil) return -4;

    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;

    if (!encode_matvec_sub(enc, pipelines, bufC0, bufA, B0, kk, (uint32_t)N0, type0, 1) ||
        !encode_matvec_sub(enc, pipelines, bufC1, bufA, B1, kk, (uint32_t)N1, type1, 1)) {
      [enc endEncoding];
      return -3;
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] == MTLCommandBufferStatusCompleted) {
      memcpy(C0, [bufC0 contents], c0Bytes);
      memcpy(C1, [bufC1 contents], c1Bytes);
      return 0;
    }
    return -1;
  }
}

int aljabr_metal_matvec_tb_triple_batch(void *C0, void *C1, void *C2,
                                        const void *A, const void *B0,
                                        const void *B1, const void *B2,
                                        int K, int N0, int N1, int N2,
                                        int type0, int type1, int type2,
                                        int batchSize) {
  if (!g_initialized) return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0 || batchSize <= 0) return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t bSz = (size_t)batchSize;
    size_t aBytes = bSz * (size_t)K * sizeof(float);
    size_t c0Bytes = bSz * (size_t)N0 * sizeof(float);
    size_t c1Bytes = bSz * (size_t)N1 * sizeof(float);
    size_t c2Bytes = bSz * (size_t)N2 * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_triple_A, &g_scratch_triple_A_bytes, aBytes);
    id<MTLBuffer> bufC0 = ensure_shared_buf(&g_scratch_triple_C0, &g_scratch_triple_C0_bytes, c0Bytes);
    id<MTLBuffer> bufC1 = ensure_shared_buf(&g_scratch_triple_C1, &g_scratch_triple_C1_bytes, c1Bytes);
    id<MTLBuffer> bufC2 = ensure_shared_buf(&g_scratch_triple_C2, &g_scratch_triple_C2_bytes, c2Bytes);
    if (bufA == nil || bufC0 == nil || bufC1 == nil || bufC2 == nil) return -4;

    uint64_t t_start = monotonic_nanos();
    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;
    uint32_t batch32 = (uint32_t)batchSize;

    if (!encode_matvec_sub(enc, pipelines, bufC0, bufA, B0, kk, (uint32_t)N0, type0, batch32) ||
        !encode_matvec_sub(enc, pipelines, bufC1, bufA, B1, kk, (uint32_t)N1, type1, batch32) ||
        !encode_matvec_sub(enc, pipelines, bufC2, bufA, B2, kk, (uint32_t)N2, type2, batch32)) {
      [enc endEncoding];
      return -3;
    }

    [enc endEncoding];
    uint64_t t_commit = monotonic_nanos();
    [cmd commit];
    [cmd waitUntilCompleted];
    uint64_t t_done = monotonic_nanos();
    if ([cmd status] == MTLCommandBufferStatusCompleted) {
      memcpy(C0, [bufC0 contents], c0Bytes);
      memcpy(C1, [bufC1 contents], c1Bytes);
      memcpy(C2, [bufC2 contents], c2Bytes);
      uint64_t t_end = monotonic_nanos();
      static int triple_batch_call_count = 0;
      triple_batch_call_count++;
      if (triple_batch_call_count <= 2) {
        printf("[NATIVE TRIPLE BATCH #%d] K=%d N0=%d N1=%d N2=%d bSz=%d types=(%d,%d,%d) prep: %.2f ms, gpu: %.2f ms, copy_out: %.2f ms, total: %.2f ms\n",
               triple_batch_call_count, K, N0, N1, N2, batchSize, type0, type1, type2,
               (t_commit - t_start) / 1e6, (t_done - t_commit) / 1e6, (t_end - t_done) / 1e6, (t_end - t_start) / 1e6);
      }
      return 0;
    }
    return -1;
  }
}

int aljabr_metal_matvec_tb_triple(void *C0, void *C1, void *C2,
                                  const void *A, const void *B0,
                                  const void *B1, const void *B2,
                                  int K, int N0, int N1, int N2,
                                  int type0, int type1, int type2) {
  return aljabr_metal_matvec_tb_triple_batch(C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2, type0, type1, type2, 1);
}

static id<MTLBuffer> g_scratch_gate = nil;
static id<MTLBuffer> g_scratch_up = nil;
static id<MTLBuffer> g_scratch_hb = nil;
static id<MTLBuffer> g_scratch_inA = nil;
static id<MTLBuffer> g_scratch_outDown = nil;
static size_t g_scratch_ffn_bytes = 0;
static size_t g_scratch_dim_bytes = 0;

int aljabr_metal_ffn_gated_fused_batch(void *outDown, const void *inA,
                                      const void *gateW, const void *upW, const void *downW,
                                      int dim, int ffnDim,
                                      int typeGate, int typeUp, int typeDown,
                                      int activationKind,
                                      int batchSize) {
  if (!g_initialized) return -1;
  if (dim <= 0 || ffnDim <= 0 || batchSize <= 0) return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    uint64_t t0 = monotonic_nanos();
    size_t bSz = (size_t)batchSize;
    size_t dimBytes = bSz * (size_t)dim * sizeof(float);
    if (g_scratch_inA == nil || g_scratch_dim_bytes < dimBytes) {
      size_t sz = (dimBytes + 16383) & ~16383;
      g_scratch_inA = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
      g_scratch_outDown = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
      g_scratch_dim_bytes = sz;
    }
    if (g_scratch_inA == nil || g_scratch_outDown == nil) return -4;

    memcpy([g_scratch_inA contents], inA, dimBytes);

    size_t needed = bSz * (size_t)ffnDim * sizeof(float);
    if (g_scratch_gate == nil || g_scratch_ffn_bytes < needed) {
      size_t sz = (needed + 16383) & ~16383;
      g_scratch_gate = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
      g_scratch_up = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
      g_scratch_hb = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
      g_scratch_ffn_bytes = sz;
    }
    if (g_scratch_gate == nil || g_scratch_up == nil || g_scratch_hb == nil) return -4;

    id<MTLComputePipelineState> actPipeline = (activationKind == 2)
        ? pipelines->gelu_ffn
        : pipelines->silu_ffn;

    // Resolve weight buffers
    size_t gateBytes = 0, upBytes = 0, downBytes = 0;
    id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
    BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

    // Gate
    if (typeGate == 12) {
      if (pipelines->matvec_tb_q4_k_mr4 != nil) { pGate = pipelines->matvec_tb_q4_k_mr4; gate_mr4 = YES; }
      else { pGate = pipelines->matvec_tb_q4_k; }
      gateBytes = (size_t)ffnDim * dim / 256 * 144;
    }
    else if (typeGate == 14) {
      if (pipelines->matvec_tb_q6_k_mr4 != nil) { pGate = pipelines->matvec_tb_q6_k_mr4; gate_mr4 = YES; }
      else { pGate = pipelines->matvec_tb_q6_k; }
      gateBytes = (size_t)ffnDim * dim / 256 * 210;
    }
    else if (typeGate == 8)  { pGate = pipelines->matvec_tb_q8_0; gateBytes = (size_t)ffnDim * dim / 32 * 34; }
    else if (typeGate == 1)  { pGate = pipelines->matvec_half; gateBytes = (size_t)ffnDim * dim * sizeof(uint16_t); }
    else if (typeGate == 30 || typeGate == 16) { pGate = pipelines->matvec_bf16; gateBytes = (size_t)ffnDim * dim * sizeof(uint16_t); }

    // Up
    if (typeUp == 12) {
      if (pipelines->matvec_tb_q4_k_mr4 != nil) { pUp = pipelines->matvec_tb_q4_k_mr4; up_mr4 = YES; }
      else { pUp = pipelines->matvec_tb_q4_k; }
      upBytes = (size_t)ffnDim * dim / 256 * 144;
    }
    else if (typeUp == 14) {
      if (pipelines->matvec_tb_q6_k_mr4 != nil) { pUp = pipelines->matvec_tb_q6_k_mr4; up_mr4 = YES; }
      else { pUp = pipelines->matvec_tb_q6_k; }
      upBytes = (size_t)ffnDim * dim / 256 * 210;
    }
    else if (typeUp == 8)  { pUp = pipelines->matvec_tb_q8_0; upBytes = (size_t)ffnDim * dim / 32 * 34; }
    else if (typeUp == 1)  { pUp = pipelines->matvec_half; upBytes = (size_t)ffnDim * dim * sizeof(uint16_t); }
    else if (typeUp == 30 || typeUp == 16) { pUp = pipelines->matvec_bf16; upBytes = (size_t)ffnDim * dim * sizeof(uint16_t); }

    // Down: matrix is (dim rows) x (ffnDim cols)
    if (typeDown == 12) {
      if (pipelines->matvec_tb_q4_k_mr4 != nil) { pDown = pipelines->matvec_tb_q4_k_mr4; down_mr4 = YES; }
      else { pDown = pipelines->matvec_tb_q4_k; }
      downBytes = (size_t)dim * ffnDim / 256 * 144;
    }
    else if (typeDown == 14) {
      if (pipelines->matvec_tb_q6_k_mr4 != nil) { pDown = pipelines->matvec_tb_q6_k_mr4; down_mr4 = YES; }
      else { pDown = pipelines->matvec_tb_q6_k; }
      downBytes = (size_t)dim * ffnDim / 256 * 210;
    }
    else if (typeDown == 8)  { pDown = pipelines->matvec_tb_q8_0; downBytes = (size_t)dim * ffnDim / 32 * 34; }
    else if (typeDown == 1)  { pDown = pipelines->matvec_half; downBytes = (size_t)dim * ffnDim * sizeof(uint16_t); }
    else if (typeDown == 30 || typeDown == 16) { pDown = pipelines->matvec_bf16; downBytes = (size_t)dim * ffnDim * sizeof(uint16_t); }

    BOOL can_fuse_gate_up = ((typeGate == 12 && typeUp == 12 && pipelines->ffn_gate_up_act_q4_k_mr4 != nil)
                              || (typeGate == 14 && typeUp == 14 && pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
    if (!can_fuse_gate_up && actPipeline == nil) {
      fprintf(stderr, "[aljabr-metal] actPipeline is nil for activationKind=%d\n", activationKind);
      return -3;
    }
    if (pGate == nil || pUp == nil || pDown == nil) {
      fprintf(stderr, "[aljabr-metal] pipeline nil: pGate=%p pUp=%p pDown=%p (types: %d, %d, %d)\n",
              pGate, pUp, pDown, typeGate, typeUp, typeDown);
      return -3;
    }
    size_t gateOff = 0, upOff = 0, downOff = 0;
    id<MTLBuffer> bufGateW = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUpW = wrap_weight_ptr_offset(upW, upBytes, &upOff);
    id<MTLBuffer> bufDownW = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    if (bufGateW == nil || bufUpW == nil || bufDownW == nil) return -4;

    uint32_t kDim = (uint32_t)dim;
    uint32_t nFfn = (uint32_t)ffnDim;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

    if ((typeGate == 12 && typeUp == 12 && pipelines->ffn_gate_up_act_q4_k_mr4 != nil)
        || (typeGate == 14 && typeUp == 14 && pipelines->ffn_gate_up_act_q6_k_mr4 != nil)) {
      // 1. Fused Gate + Up + Activation (GeGLU / SwiGLU) directly into g_scratch_hb
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      BOOL use_b4 = (bSz > 1 && ((typeGate == 12 && typeUp == 12 && pipelines->ffn_gate_up_act_q4_k_b4_mr4 != nil)
                                 || (typeGate == 14 && typeUp == 14 && pipelines->ffn_gate_up_act_q6_k_b4_mr4 != nil)));
      id<MTLComputePipelineState> fuseP;
      if (use_b4) {
        fuseP = (typeGate == 14) ? pipelines->ffn_gate_up_act_q6_k_b4_mr4 : pipelines->ffn_gate_up_act_q4_k_b4_mr4;
      } else if (typeGate == 14) {
        fuseP = pipelines->ffn_gate_up_act_q6_k_mr4;
      } else {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr4;
      }
      [enc setComputePipelineState:fuseP];
      [enc setBuffer:g_scratch_hb offset:0 atIndex:0];
      [enc setBuffer:g_scratch_inA offset:0 atIndex:1];
      [enc setBuffer:bufGateW offset:gateOff atIndex:2];
      [enc setBuffer:bufUpW offset:upOff atIndex:3];
      [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
      uint32_t actKind = (uint32_t)activationKind;
      [enc setBytes:&actKind length:sizeof(actKind) atIndex:6];
      if (use_b4) {
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 2. MatVec Down directly from g_scratch_hb to g_scratch_outDown
      if (bSz > 1 && typeDown == 12 && pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q4_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else if (bSz > 1 && typeDown == 14 && pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q6_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc setComputePipelineState:pDown];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        if (down_mr4) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, bSz, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
          [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)dim, bSz, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
      }
      [enc endEncoding];
    } else {
      // High-occupancy 1-encoder FFN pipeline: Gate MR4 + Up MR4 + Act + Down MR4
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      // 1. Gate
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_scratch_gate offset:0 atIndex:0];
      [enc setBuffer:g_scratch_inA offset:0 atIndex:1];
      [enc setBuffer:bufGateW offset:gateOff atIndex:2];
      [enc setBytes:&kDim length:sizeof(kDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, bSz, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      // 2. Up
      [enc setComputePipelineState:pUp];
      [enc setBuffer:g_scratch_up offset:0 atIndex:0];
      [enc setBuffer:g_scratch_inA offset:0 atIndex:1];
      [enc setBuffer:bufUpW offset:upOff atIndex:2];
      [enc setBytes:&kDim length:sizeof(kDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, bSz, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 3. Activation (GeGLU or SwiGLU)
      [enc setComputePipelineState:actPipeline];
      [enc setBuffer:g_scratch_hb offset:0 atIndex:0];
      [enc setBuffer:g_scratch_gate offset:0 atIndex:1];
      [enc setBuffer:g_scratch_up offset:0 atIndex:2];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
      NSUInteger threads = MIN((NSUInteger)256, actPipeline.maxTotalThreadsPerThreadgroup);
      NSUInteger totalElem = bSz * (NSUInteger)ffnDim;
      NSUInteger groups = (totalElem + threads - 1) / threads;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 4. MatVec Down: (dim rows) x (ffnDim cols)
      if (bSz > 1 && typeDown == 12 && pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q4_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else if (bSz > 1 && typeDown == 14 && pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q6_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc setComputePipelineState:pDown];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        if (down_mr4) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, bSz, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
          [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)dim, bSz, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
      }
      [enc endEncoding];
    }

    static int ffn_call_count = 0;
    uint64_t t_encoded = monotonic_nanos();
    [cmd commit];
    [cmd waitUntilCompleted];
    uint64_t t_done = monotonic_nanos();
    if (getenv("ALJABR_METAL_DEBUG_FFN") != NULL && (ffn_call_count < 10 || (ffn_call_count % 48 == 0))) {
      printf("[NATIVE FFN %d] bSz=%d, wrap+encode: %.2f ms, gpu_wait: %.2f ms, total: %.2f ms (mr4: gate=%d up=%d down=%d)\n",
             ffn_call_count, (int)bSz,
             (t_encoded - t0) / 1e6,
             (t_done - t_encoded) / 1e6,
             (t_done - t0) / 1e6,
             (int)gate_mr4, (int)up_mr4, (int)down_mr4);
    }
    ffn_call_count++;

    if ([cmd status] == MTLCommandBufferStatusCompleted) {
      memcpy(outDown, [g_scratch_outDown contents], dimBytes);
      return 0;
    }
    return -1;
  }
}

int aljabr_metal_ffn_gated_fused(void *outDown, const void *inA,
                                const void *gateW, const void *upW, const void *downW,
                                int dim, int ffnDim,
                                int typeGate, int typeUp, int typeDown,
                                int activationKind) {
  return aljabr_metal_ffn_gated_fused_batch(outDown, inA, gateW, upW, downW, dim, ffnDim,
                                           typeGate, typeUp, typeDown, activationKind, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// aljabr_metal_wo_ffn_fused — fused wO + norms + residuals + FFN
// Eliminates 2 waitUntilCompleted per layer → saves ~28 ms per decode token
// ─────────────────────────────────────────────────────────────────────────────

static id<MTLBuffer> g_woff_attnFull = nil;
static id<MTLBuffer> g_woff_xb       = nil;
static id<MTLBuffer> g_woff_x        = nil;
static id<MTLBuffer> g_woff_normW1   = nil;
static id<MTLBuffer> g_woff_normW2   = nil;
static id<MTLBuffer> g_woff_normW3   = nil;
static id<MTLBuffer> g_woff_ffn_gate = nil;
static id<MTLBuffer> g_woff_ffn_up   = nil;
static id<MTLBuffer> g_woff_ffn_hb   = nil;
static size_t g_woff_attnFull_cap = 0;
static size_t g_woff_dim_cap      = 0;
static size_t g_woff_ffn_cap      = 0;
static BOOL g_woff_x_valid        = NO;

static id<MTLBuffer> g_woff_nextQ = nil;
static id<MTLBuffer> g_woff_nextK = nil;
static id<MTLBuffer> g_woff_nextV = nil;
static id<MTLBuffer> g_woff_nextAttnNormW = nil;
static size_t g_woff_nextQ_cap = 0;
static size_t g_woff_nextK_cap = 0;
static size_t g_woff_nextV_cap = 0;
static size_t g_woff_nextAttnNormW_cap = 0;

static BOOL ensure_woff_next_qkv(size_t qBytes, size_t kBytes, size_t vBytes, size_t normBytes) {
  if (qBytes > g_woff_nextQ_cap) {
    size_t sz = (qBytes + 16383) & ~16383;
    g_woff_nextQ = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_nextQ_cap = sz;
    if (!g_woff_nextQ) return NO;
  }
  if (kBytes > g_woff_nextK_cap) {
    size_t sz = (kBytes + 16383) & ~16383;
    g_woff_nextK = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_nextK_cap = sz;
    if (!g_woff_nextK) return NO;
  }
  if (vBytes > g_woff_nextV_cap) {
    size_t sz = (vBytes + 16383) & ~16383;
    g_woff_nextV = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_nextV_cap = sz;
    if (!g_woff_nextV) return NO;
  }
  if (normBytes > g_woff_nextAttnNormW_cap) {
    size_t sz = (normBytes + 16383) & ~16383;
    g_woff_nextAttnNormW = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_nextAttnNormW_cap = sz;
    if (!g_woff_nextAttnNormW) return NO;
  }
  return YES;
}

static BOOL ensure_woff_dim_bufs(size_t attnKBytes, size_t dimBytes) {
  if (attnKBytes > g_woff_attnFull_cap) {
    size_t sz = (attnKBytes + 16383) & ~16383;
    g_woff_attnFull = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_attnFull_cap = sz;
    if (!g_woff_attnFull) return NO;
  }
  if (dimBytes > g_woff_dim_cap) {
    size_t sz = (dimBytes + 16383) & ~16383;
    g_woff_xb     = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_x      = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_normW1 = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_normW2 = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_normW3 = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_dim_cap = sz;
    if (!g_woff_xb || !g_woff_x || !g_woff_normW1 || !g_woff_normW2 || !g_woff_normW3) return NO;
  }
  return YES;
}

static BOOL ensure_woff_ffn_bufs(size_t ffnBytes) {
  if (ffnBytes > g_woff_ffn_cap) {
    size_t sz = (ffnBytes + 16383) & ~16383;
    g_woff_ffn_gate = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_up   = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_hb   = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_cap = sz;
    if (!g_woff_ffn_gate || !g_woff_ffn_up || !g_woff_ffn_hb) return NO;
  }
  return YES;
}

static void encode_woff_rmsnorm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> bufOut, id<MTLBuffer> bufIn, id<MTLBuffer> bufW, size_t wOffset,
    uint32_t N, float eps, uint32_t addOne,
    AljabrMetalPipelines *pipelines) {
  id<MTLComputePipelineState> p =
      (N % 4 == 0 && pipelines->rmsnorm_float4 != nil)
          ? pipelines->rmsnorm_float4
          : pipelines->rmsnorm;
  if (p == nil) return;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufOut  offset:0 atIndex:0];
  [enc setBuffer:bufIn   offset:0 atIndex:1];
  [enc setBuffer:bufW    offset:wOffset atIndex:2];
  [enc setBytes:&N       length:sizeof(N)      atIndex:3];
  [enc setBytes:&eps     length:sizeof(eps)    atIndex:4];
  [enc setBytes:&addOne  length:sizeof(addOne) atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_woff_add(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> bufC, id<MTLBuffer> bufA, id<MTLBuffer> bufB,
    uint32_t N, AljabrMetalPipelines *pipelines) {
  id<MTLComputePipelineState> p = pipelines->add;
  if (p == nil) return;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufC offset:0 atIndex:0];
  [enc setBuffer:bufA offset:0 atIndex:1];
  [enc setBuffer:bufB offset:0 atIndex:2];
  [enc setBytes:&N length:sizeof(N) atIndex:3];
  NSUInteger threads = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

int aljabr_metal_wo_ffn_next_qkv_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW,
    const float *postFfnW, int hasPostFfnNorm,
    float eps,
    float outScale,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind,
    int hasNext,
    const float *nextAttnNormW,
    float *nextQ, float *nextK, float *nextV,
    const void *nextWQ, const void *nextWK, const void *nextWV,
    int nextTypeQ, int nextTypeK, int nextTypeV,
    int nextN0, int nextN1, int nextN2
) {
  if (!g_initialized) return -1;
  if (wO_K <= 0 || wO_N <= 0 || ffnDim <= 0) return -2;

  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->rmsnorm == nil && pipelines->rmsnorm_float4 == nil) return -3;
  if (pipelines->add == nil) return -3;

  // ── Resolve wO pipeline ────────────────────────────────────────────────────
  id<MTLComputePipelineState> pWO = nil;
  size_t wOBytes = 0;
  BOOL wO_mr4 = NO;
  if (typeWO == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pWO = pipelines->matvec_tb_q4_k_mr4; wO_mr4 = YES; }
    else pWO = pipelines->matvec_tb_q4_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 144;
  } else if (typeWO == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pWO = pipelines->matvec_tb_q6_k_mr4; wO_mr4 = YES; }
    else pWO = pipelines->matvec_tb_q6_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 210;
  } else if (typeWO == 8) {
    pWO = pipelines->matvec_tb_q8_0;
    wOBytes = (size_t)wO_N * wO_K / 32 * 34;
  } else if (typeWO == 1) { pWO = pipelines->matvec_half;  wOBytes = (size_t)wO_N * wO_K * 2; }
  else if (typeWO == 30 || typeWO == 16) { pWO = pipelines->matvec_bf16; wOBytes = (size_t)wO_N * wO_K * 2; }
  if (pWO == nil) return -3;

  // ── Resolve FFN pipelines ──────────────────────────────────────────────────
  BOOL can_fuse_gate_up = ((typeGate == 12 && typeUp == 12
                            && pipelines->ffn_gate_up_act_q4_k_mr4 != nil)
                           || (typeGate == 14 && typeUp == 14
                            && pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
  id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
  size_t gateBytes = 0, upBytes = 0, downBytes = 0;
  BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

  if (typeGate == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pGate = pipelines->matvec_tb_q4_k_mr4; gate_mr4 = YES; }
    else pGate = pipelines->matvec_tb_q4_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeGate == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pGate = pipelines->matvec_tb_q6_k_mr4; gate_mr4 = YES; }
    else pGate = pipelines->matvec_tb_q6_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 210;
  } else if (typeGate == 8) { pGate = pipelines->matvec_tb_q8_0; gateBytes = (size_t)ffnDim * wO_N / 32 * 34; }
  else if (typeGate == 1) { pGate = pipelines->matvec_half; gateBytes = (size_t)ffnDim * wO_N * 2; }
  else if (typeGate == 30 || typeGate == 16) { pGate = pipelines->matvec_bf16; gateBytes = (size_t)ffnDim * wO_N * 2; }

  if (typeUp == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pUp = pipelines->matvec_tb_q4_k_mr4; up_mr4 = YES; }
    else pUp = pipelines->matvec_tb_q4_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeUp == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pUp = pipelines->matvec_tb_q6_k_mr4; up_mr4 = YES; }
    else pUp = pipelines->matvec_tb_q6_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 210;
  } else if (typeUp == 8) { pUp = pipelines->matvec_tb_q8_0; upBytes = (size_t)ffnDim * wO_N / 32 * 34; }
  else if (typeUp == 1) { pUp = pipelines->matvec_half; upBytes = (size_t)ffnDim * wO_N * 2; }
  else if (typeUp == 30 || typeUp == 16) { pUp = pipelines->matvec_bf16; upBytes = (size_t)ffnDim * wO_N * 2; }

  if (typeDown == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pDown = pipelines->matvec_tb_q6_k_mr4; down_mr4 = YES; }
    else pDown = pipelines->matvec_tb_q6_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 210;
  } else if (typeDown == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pDown = pipelines->matvec_tb_q4_k_mr4; down_mr4 = YES; }
    else pDown = pipelines->matvec_tb_q4_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 144;
  } else if (typeDown == 8) { pDown = pipelines->matvec_tb_q8_0; downBytes = (size_t)wO_N * ffnDim / 32 * 34; }
  else if (typeDown == 1) { pDown = pipelines->matvec_half; downBytes = (size_t)wO_N * ffnDim * 2; }
  else if (typeDown == 30 || typeDown == 16) { pDown = pipelines->matvec_bf16; downBytes = (size_t)wO_N * ffnDim * 2; }

  if (!can_fuse_gate_up && (pGate == nil || pUp == nil)) return -3;
  if (pDown == nil) return -3;

  // ── Ensure scratch buffers ─────────────────────────────────────────────────
  size_t attnKBytes = (size_t)wO_K * sizeof(float);
  size_t dimBytes   = (size_t)wO_N * sizeof(float);
  size_t ffnBytes   = (size_t)ffnDim * sizeof(float);
  if (!ensure_woff_dim_bufs(attnKBytes, dimBytes)) return -4;
  if (!ensure_woff_ffn_bufs(ffnBytes)) return -4;

  if (hasNext && nextWQ != NULL) {
    size_t qBytes = (size_t)nextN0 * sizeof(float);
    size_t kBytes = (size_t)nextN1 * sizeof(float);
    size_t vBytes = (size_t)nextN2 * sizeof(float);
    if (!ensure_woff_next_qkv(qBytes, kBytes, vBytes, dimBytes)) return -4;
  }

  @autoreleasepool {
    // ── Copy inputs to GPU-accessible Shared buffers ───────────────────────
    memcpy([g_woff_attnFull contents], attn_full, attnKBytes);
    if (!g_woff_x_valid) {
      memcpy([g_woff_x contents], x, dimBytes);
      g_woff_x_valid = YES;
    }
    // Copy norm weight data into pre-allocated shared scratch buffers.
    // (Norms are dequantized Java-heap float[] arrays — not in the master Metal model
    //  buffer — so zero-copy wrapping is not possible. Direct memcpy into the small
    //  pre-allocated g_woff_normW* buffers is cheaper than MTLBuffer allocation.)
    memcpy([g_woff_normW2 contents], ffnNormW, dimBytes);
    id<MTLBuffer> bufFfnNorm = g_woff_normW2;
    const size_t ffnNormOff = 0;

    id<MTLBuffer> bufPostAttn = nil;
    const size_t postAttnOff = 0;
    if (hasPostAttnNorm && postAttnW) {
      memcpy([g_woff_normW1 contents], postAttnW, dimBytes);
      bufPostAttn = g_woff_normW1;
    }

    id<MTLBuffer> bufPostFfn = nil;
    const size_t postFfnOff = 0;
    if (hasPostFfnNorm && postFfnW) {
      memcpy([g_woff_normW3 contents], postFfnW, dimBytes);
      bufPostFfn = g_woff_normW3;
    }

    id<MTLBuffer> bufNextAttn = nil;
    const size_t nextAttnOff = 0;
    if (hasNext && nextAttnNormW) {
      memcpy([g_woff_nextAttnNormW contents], nextAttnNormW, dimBytes);
      bufNextAttn = g_woff_nextAttnNormW;
    }

    // Wrap weight buffers (zero-copy via registered master model buffer)
    size_t wOOff = 0, gateOff = 0, upOff = 0, downOff = 0;
    id<MTLBuffer> bufWO   = wrap_weight_ptr_offset(wO,    wOBytes,   &wOOff);
    id<MTLBuffer> bufGate = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUp   = wrap_weight_ptr_offset(upW,   upBytes,   &upOff);
    id<MTLBuffer> bufDown = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    if (!bufWO || !bufGate || !bufUp || !bufDown) return -4;

    uint32_t kDim = (uint32_t)wO_K;
    uint32_t nDim = (uint32_t)wO_N;
    uint32_t nFfn = (uint32_t)ffnDim;
    uint32_t actK = (uint32_t)activationKind;

    // ── Single command buffer, single encoder ──────────────────────────────
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Step 1: xb = wO(attnFull)
    [enc setComputePipelineState:pWO];
    [enc setBuffer:g_woff_xb       offset:0     atIndex:0];
    [enc setBuffer:g_woff_attnFull offset:0    atIndex:1];
    [enc setBuffer:bufWO           offset:wOOff atIndex:2];
    [enc setBytes:&kDim  length:sizeof(kDim) atIndex:3];
    [enc setBytes:&nDim  length:sizeof(nDim) atIndex:4];
    if (wO_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 2 (optional): xb = postAttnNorm(xb)
    if (hasPostAttnNorm && bufPostAttn) {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostAttn, postAttnOff, nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 3: x += xb  (attn residual)
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 4: xb = ffnNorm(x)
    {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufFfnNorm, ffnNormOff, nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 5+6: FFN gate+up+act → hb
    if (can_fuse_gate_up) {
      id<MTLComputePipelineState> fuseP = (typeGate == 14)
          ? pipelines->ffn_gate_up_act_q6_k_mr4
          : pipelines->ffn_gate_up_act_q4_k_mr4;
      [enc setComputePipelineState:fuseP];
      [enc setBuffer:g_woff_ffn_hb  offset:0        atIndex:0];
      [enc setBuffer:g_woff_xb      offset:0        atIndex:1];
      [enc setBuffer:bufGate        offset:gateOff   atIndex:2];
      [enc setBuffer:bufUp          offset:upOff     atIndex:3];
      [enc setBytes:&nDim  length:sizeof(nDim) atIndex:4];
      [enc setBytes:&nFfn  length:sizeof(nFfn) atIndex:5];
      [enc setBytes:&actK  length:sizeof(actK) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_woff_ffn_gate offset:0       atIndex:0];
      [enc setBuffer:g_woff_xb       offset:0       atIndex:1];
      [enc setBuffer:bufGate         offset:gateOff atIndex:2];
      [enc setBytes:&nDim  length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn  length:sizeof(nFfn) atIndex:4];
      if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc setComputePipelineState:pUp];
      [enc setBuffer:g_woff_ffn_up   offset:0      atIndex:0];
      [enc setBuffer:g_woff_xb       offset:0      atIndex:1];
      [enc setBuffer:bufUp           offset:upOff  atIndex:2];
      [enc setBytes:&nDim  length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn  length:sizeof(nFfn) atIndex:4];
      if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      id<MTLComputePipelineState> actP =
          (activationKind == 2) ? pipelines->gelu_ffn : pipelines->silu_ffn;
      if (actP == nil) { [enc endEncoding]; return -3; }
      [enc setComputePipelineState:actP];
      [enc setBuffer:g_woff_ffn_hb   offset:0      atIndex:0];
      [enc setBuffer:g_woff_ffn_gate offset:0      atIndex:1];
      [enc setBuffer:g_woff_ffn_up   offset:0      atIndex:2];
      [enc setBytes:&nFfn  length:sizeof(nFfn) atIndex:3];
      NSUInteger actTh = MIN((NSUInteger)256, actP.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(actTh, 1, 1)];
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 7: xb = down(hb)   [FFN down projection]
    [enc setComputePipelineState:pDown];
    [enc setBuffer:g_woff_xb       offset:0        atIndex:0];
    [enc setBuffer:g_woff_ffn_hb   offset:0        atIndex:1];
    [enc setBuffer:bufDown         offset:downOff  atIndex:2];
    [enc setBytes:&nFfn  length:sizeof(nFfn) atIndex:3];
    [enc setBytes:&nDim  length:sizeof(nDim) atIndex:4];
    if (down_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Step 8 (optional): xb = postFfnNorm(xb)
    if (hasPostFfnNorm && bufPostFfn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostFfn, postFfnOff, nDim, eps, 0, pipelines);
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 9: x += xb   (FFN residual — x is now fully updated)
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);

    // Optional layer output scale: x *= outScale on GPU before next layer QKV
    if (outScale > 0.0f && outScale != 1.0f && pipelines->scale != nil) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [enc setComputePipelineState:pipelines->scale];
      [enc setBuffer:g_woff_x offset:0 atIndex:0];
      [enc setBytes:&outScale length:sizeof(outScale) atIndex:1];
      [enc setBytes:&nDim     length:sizeof(nDim)     atIndex:2];
      NSUInteger threads = MIN((NSUInteger)256, pipelines->scale.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)nDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // ── Step 10+: Chain Next Layer QKV directly into the same command buffer ──
    if (hasNext && nextWQ != NULL && bufNextAttn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // Step 10: xb = rmsNorm(x, nextAttnNormW)
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufNextAttn, nextAttnOff, nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // Step 11, 12, 13: next Q, K, V matvecs
      uint32_t nextKK = nDim;
      encode_matvec_sub(enc, pipelines, g_woff_nextQ, g_woff_xb, nextWQ, nextKK, (uint32_t)nextN0, nextTypeQ, 1);
      if (nextWK != NULL) {
        encode_matvec_sub(enc, pipelines, g_woff_nextK, g_woff_xb, nextWK, nextKK, (uint32_t)nextN1, nextTypeK, 1);
      }
      if (nextWV != NULL) {
        encode_matvec_sub(enc, pipelines, g_woff_nextV, g_woff_xb, nextWV, nextKK, (uint32_t)nextN2, nextTypeV, 1);
      }
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      g_woff_x_valid = NO;
      return -1;
    }

    // Return fully-updated residual x to caller only when finishing the layer chain
    if (!hasNext) {
      memcpy(x, [g_woff_x contents], dimBytes);
      g_woff_x_valid = NO;
    }

    if (hasNext && nextWQ != NULL) {
      memcpy(nextQ, [g_woff_nextQ contents], (size_t)nextN0 * sizeof(float));
      if (nextWK != NULL && nextK != NULL) {
        memcpy(nextK, [g_woff_nextK contents], (size_t)nextN1 * sizeof(float));
      }
      if (nextWV != NULL && nextV != NULL) {
        memcpy(nextV, [g_woff_nextV contents], (size_t)nextN2 * sizeof(float));
      }
    }
    return 0;
  }
}

int aljabr_metal_wo_ffn_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm, int addOnePostAttn,
    const float *ffnNormW, int addOneFfnNorm,
    const float *postFfnW, int hasPostFfnNorm, int addOnePostFfn,
    float eps,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind
) {
  return aljabr_metal_wo_ffn_next_qkv_fused(
      x, attn_full, wO, typeWO, wO_K, wO_N,
      postAttnW, hasPostAttnNorm,
      ffnNormW,
      postFfnW, hasPostFfnNorm,
      eps,
      1.0f,
      gateW, upW, downW,
      typeGate, typeUp, typeDown,
      ffnDim, activationKind,
      0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0);
}

void aljabr_metal_wo_ffn_reset_x(void) {
  g_woff_x_valid = NO;
}

static id<MTLBuffer> g_woff_logits = nil;
static size_t g_woff_logits_cap = 0;
static id<MTLBuffer> g_woff_outputNormW = nil;
static size_t g_woff_outputNormW_cap = 0;

static BOOL ensure_woff_logits(size_t logitsBytes, size_t normBytes) {
  if (logitsBytes > g_woff_logits_cap) {
    size_t sz = (logitsBytes + 16383) & ~16383;
    g_woff_logits = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_logits_cap = sz;
    if (!g_woff_logits) return NO;
  }
  if (normBytes > g_woff_outputNormW_cap) {
    size_t sz = (normBytes + 16383) & ~16383;
    g_woff_outputNormW = [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_outputNormW_cap = sz;
    if (!g_woff_outputNormW) return NO;
  }
  return YES;
}

int aljabr_metal_wo_ffn_logits_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW,
    const float *postFfnW, int hasPostFfnNorm,
    float eps,
    float outScale,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind,
    const float *outputNormW,
    const void *outputW, int typeOutputW, int outputK, int outputN,
    float *logits, float cap
) {
  if (!g_initialized) return -1;
  if (wO_K <= 0 || wO_N <= 0 || ffnDim <= 0 || outputK <= 0 || outputN <= 0) return -2;

  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->rmsnorm == nil && pipelines->rmsnorm_float4 == nil) return -3;
  if (pipelines->add == nil) return -3;

  // Resolve wO pipeline
  id<MTLComputePipelineState> pWO = nil;
  size_t wOBytes = 0;
  BOOL wO_mr4 = NO;
  if (typeWO == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pWO = pipelines->matvec_tb_q4_k_mr4; wO_mr4 = YES; }
    else pWO = pipelines->matvec_tb_q4_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 144;
  } else if (typeWO == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pWO = pipelines->matvec_tb_q6_k_mr4; wO_mr4 = YES; }
    else pWO = pipelines->matvec_tb_q6_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 210;
  }
  if (pWO == nil) return -3;

  // Resolve FFN pipelines
  BOOL can_fuse_gate_up = ((typeGate == 12 && typeUp == 12 && pipelines->ffn_gate_up_act_q4_k_mr4 != nil)
                           || (typeGate == 14 && typeUp == 14 && pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
  id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
  size_t gateBytes = 0, upBytes = 0, downBytes = 0;
  BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

  if (typeGate == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pGate = pipelines->matvec_tb_q4_k_mr4; gate_mr4 = YES; }
    else pGate = pipelines->matvec_tb_q4_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeGate == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pGate = pipelines->matvec_tb_q6_k_mr4; gate_mr4 = YES; }
    else pGate = pipelines->matvec_tb_q6_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 210;
  }

  if (typeUp == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pUp = pipelines->matvec_tb_q4_k_mr4; up_mr4 = YES; }
    else pUp = pipelines->matvec_tb_q4_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeUp == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pUp = pipelines->matvec_tb_q6_k_mr4; up_mr4 = YES; }
    else pUp = pipelines->matvec_tb_q6_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 210;
  }

  if (typeDown == 12) {
    if (pipelines->matvec_tb_q4_k_mr4 != nil) { pDown = pipelines->matvec_tb_q4_k_mr4; down_mr4 = YES; }
    else pDown = pipelines->matvec_tb_q4_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 144;
  } else if (typeDown == 14) {
    if (pipelines->matvec_tb_q6_k_mr4 != nil) { pDown = pipelines->matvec_tb_q6_k_mr4; down_mr4 = YES; }
    else pDown = pipelines->matvec_tb_q6_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 210;
  }

  if (!can_fuse_gate_up && (pGate == nil || pUp == nil)) return -3;
  if (pDown == nil) return -3;

  // Resolve Logits pipeline
  id<MTLComputePipelineState> pLogits = nil;
  BOOL logits_mr4 = NO;
  size_t outWBytes = 0;
  if (typeOutputW == 12) { // Q4_K
    if (cap > 0.0f && pipelines->matvec_tb_q4_k_mr4_softcap != nil) {
      pLogits = pipelines->matvec_tb_q4_k_mr4_softcap;
      logits_mr4 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pLogits = pipelines->matvec_tb_q4_k_mr4;
      logits_mr4 = YES;
    } else {
      pLogits = pipelines->matvec_tb_q4_k;
    }
    outWBytes = (size_t)outputN * outputK / 256 * 144;
  } else if (typeOutputW == 14) { // Q6_K
    if (cap > 0.0f && pipelines->matvec_tb_q6_k_mr4_softcap != nil) {
      pLogits = pipelines->matvec_tb_q6_k_mr4_softcap;
      logits_mr4 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pLogits = pipelines->matvec_tb_q6_k_mr4;
      logits_mr4 = YES;
    } else {
      pLogits = pipelines->matvec_tb_q6_k;
    }
    outWBytes = (size_t)outputN * outputK / 256 * 210;
  }
  if (pLogits == nil || outWBytes == 0) return -3;

  size_t attnKBytes = (size_t)wO_K * sizeof(float);
  size_t dimBytes   = (size_t)wO_N * sizeof(float);
  size_t ffnBytes   = (size_t)ffnDim * sizeof(float);
  size_t logitsBytes = (size_t)outputN * sizeof(float);

  if (!ensure_woff_dim_bufs(attnKBytes, dimBytes)) return -4;
  if (!ensure_woff_ffn_bufs(ffnBytes)) return -4;
  if (!ensure_woff_logits(logitsBytes, dimBytes)) return -4;

  @autoreleasepool {
    memcpy([g_woff_attnFull contents], attn_full, attnKBytes);
    if (!g_woff_x_valid) {
      memcpy([g_woff_x contents], x, dimBytes);
    }
    // Copy norm weight data into pre-allocated shared scratch buffers.
    memcpy([g_woff_normW2 contents], ffnNormW, dimBytes);
    id<MTLBuffer> bufFfnNorm = g_woff_normW2;
    const size_t ffnNormOff = 0;

    id<MTLBuffer> bufPostAttn = nil;
    const size_t postAttnOff = 0;
    if (hasPostAttnNorm && postAttnW) {
      memcpy([g_woff_normW1 contents], postAttnW, dimBytes);
      bufPostAttn = g_woff_normW1;
    }

    id<MTLBuffer> bufPostFfn = nil;
    const size_t postFfnOff = 0;
    if (hasPostFfnNorm && postFfnW) {
      memcpy([g_woff_normW3 contents], postFfnW, dimBytes);
      bufPostFfn = g_woff_normW3;
    }

    memcpy([g_woff_outputNormW contents], outputNormW, dimBytes);
    id<MTLBuffer> bufOutNorm = g_woff_outputNormW;
    const size_t outNormOff = 0;


    size_t wOOff = 0, gateOff = 0, upOff = 0, downOff = 0, outWOff = 0;
    id<MTLBuffer> bufWO   = wrap_weight_ptr_offset(wO, wOBytes, &wOOff);
    id<MTLBuffer> bufGate = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUp   = wrap_weight_ptr_offset(upW, upBytes, &upOff);
    id<MTLBuffer> bufDown = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    id<MTLBuffer> bufOutW = wrap_weight_ptr_offset(outputW, outWBytes, &outWOff);
    if (!bufWO || !bufGate || !bufUp || !bufDown || !bufOutW) return -4;

    uint32_t kDim = (uint32_t)wO_K;
    uint32_t nDim = (uint32_t)wO_N;
    uint32_t nFfn = (uint32_t)ffnDim;
    uint32_t actK = (uint32_t)activationKind;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Step 1: xb = wO(attnFull)
    [enc setComputePipelineState:pWO];
    [enc setBuffer:g_woff_xb offset:0 atIndex:0];
    [enc setBuffer:g_woff_attnFull offset:0 atIndex:1];
    [enc setBuffer:bufWO offset:wOOff atIndex:2];
    [enc setBytes:&kDim length:sizeof(kDim) atIndex:3];
    [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
    if (wO_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 2: xb = postAttnNorm(xb)
    if (hasPostAttnNorm && bufPostAttn) {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostAttn, postAttnOff, nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 3: x += xb
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 4: xb = ffnNorm(x)
    encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufFfnNorm, ffnNormOff, nDim, eps, 0, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 5+6: gate+up+act
    if (can_fuse_gate_up) {
      id<MTLComputePipelineState> fuseP = (typeGate == 14)
          ? pipelines->ffn_gate_up_act_q6_k_mr4
          : pipelines->ffn_gate_up_act_q4_k_mr4;
      [enc setComputePipelineState:fuseP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBuffer:bufUp offset:upOff atIndex:3];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
      [enc setBytes:&actK length:sizeof(actK) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc setComputePipelineState:pUp];
      [enc setBuffer:g_woff_ffn_up offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufUp offset:upOff atIndex:2];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      id<MTLComputePipelineState> actP = (activationKind == 2) ? pipelines->gelu_ffn : pipelines->silu_ffn;
      if (actP == nil) { [enc endEncoding]; return -3; }
      [enc setComputePipelineState:actP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:1];
      [enc setBuffer:g_woff_ffn_up offset:0 atIndex:2];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
      NSUInteger actTh = MIN((NSUInteger)256, actP.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(actTh, 1, 1)];
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 7: xb = down(hb)
    [enc setComputePipelineState:pDown];
    [enc setBuffer:g_woff_xb offset:0 atIndex:0];
    [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:1];
    [enc setBuffer:bufDown offset:downOff atIndex:2];
    [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
    [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
    if (down_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Step 8: postFfnNorm
    if (hasPostFfnNorm && bufPostFfn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostFfn, postFfnOff, nDim, eps, 0, pipelines);
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 9: x += xb
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);

    // Step 9b: outScale
    if (outScale > 0.0f && outScale != 1.0f && pipelines->scale != nil) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [enc setComputePipelineState:pipelines->scale];
      [enc setBuffer:g_woff_x offset:0 atIndex:0];
      [enc setBytes:&outScale length:sizeof(outScale) atIndex:1];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:2];
      NSUInteger threads = MIN((NSUInteger)256, pipelines->scale.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)nDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // ── Step 10: Output RMSNorm directly into g_woff_xb ──
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufOutNorm, outNormOff, nDim, eps, 0, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ── Step 11: Output Logits Matvec + Softcap ──
    [enc setComputePipelineState:pLogits];
    [enc setBuffer:g_woff_logits offset:0 atIndex:0];
    [enc setBuffer:g_woff_xb offset:0 atIndex:1];
    [enc setBuffer:bufOutW offset:outWOff atIndex:2];
    uint32_t outK = (uint32_t)outputK;
    uint32_t outN = (uint32_t)outputN;
    [enc setBytes:&outK length:sizeof(outK) atIndex:3];
    [enc setBytes:&outN length:sizeof(outN) atIndex:4];
    if (cap > 0.0f) {
      [enc setBytes:&cap length:sizeof(cap) atIndex:5];
    }
    if (logits_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)outputN + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)outputN, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      g_woff_x_valid = NO;
      return -1;
    }

    memcpy(logits, [g_woff_logits contents], (size_t)outputN * sizeof(float));
    if (x != NULL) {
      memcpy(x, [g_woff_x contents], dimBytes);
    }
    g_woff_x_valid = NO;
    return 0;
  }
}

