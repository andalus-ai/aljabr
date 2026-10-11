/**
 * alkhawarizm_metal_matvec_api.m — exported Metal matvec C API for Alkhawarizm.
 *
 * Optimized for reduced memory pressure and fixed KV cache layout.
 */

#import "alkhawarizm_metal_matvec_api.h"

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

#import "alkhawarizm_metal_attention.h"
#import "alkhawarizm_metal_buffers.h"
#import "alkhawarizm_metal_matvec_dispatch.h"
#import "alkhawarizm_metal_matvec_tuning.h"
#import "alkhawarizm_metal_mps_cache.h"
#import "alkhawarizm_metal_mps_matvec_policy.h"
#import "alkhawarizm_metal_mps_validation.h"
#import "alkhawarizm_metal_pipelines.h"
#import "alkhawarizm_metal_scratch.h"
#import "alkhawarizm_metal_support.h"

#define env_truthy alkhawarizm_metal_env_truthy
#define env_int_or_default alkhawarizm_metal_env_int_or_default
#define env_float_or_default alkhawarizm_metal_env_float_or_default
#define monotonic_nanos alkhawarizm_metal_monotonic_nanos

// ── Globals
// ───────────────────────────────────────────────────────────────────

#define g_pipelines (*alkhawarizm_metal_pipelines())
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
// ── Matvec C API ─────────────────────────────────────────────────────────────

static int alkhawarizm_metal_matvec_tb_half_custom(void *C, const void *A,
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_half_128_pipeline
                                                : g_matvec_half_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int alkhawarizm_metal_matvec_tb_half(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  if (alkhawarizm_metal_mps_matvec_should_try(K, N)) {
    @autoreleasepool {
      NSString *shapeKey = mps_matvec_shape_key(K, N);
      BOOL validateEveryCall =
          alkhawarizm_metal_mps_matvec_validate_every_call();
      BOOL autotune =
          alkhawarizm_metal_mps_matvec_autotune_enabled_for_output(N);
      AlkhawarizmMetalMpsMatvecShapeState shapeState =
          alkhawarizm_metal_mps_matvec_shape_state(shapeKey, validateEveryCall,
                                                   autotune);
      if (!shapeState.failed && !shapeState.custom_preferred) {
        uint64_t mpsStart = autotune ? monotonic_nanos() : 0;
        uint16_t *halfA =
            alkhawarizm_metal_ensure_half_input_scratch((size_t)K);
        uint16_t *halfC =
            alkhawarizm_metal_ensure_half_output_scratch((size_t)N);
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
                    !env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
                  alkhawarizm_metal_mps_matvec_mark_validated(shapeKey);
                  if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                    fprintf(stderr,
                            "[alkhawarizm-metal] MPS matvec validated shape "
                            "K=%d N=%d\n",
                            K, N);
                  }
                }
                if (autotune && !shapeState.mps_preferred) {
                  uint64_t customStart = monotonic_nanos();
                  int customRc =
                      alkhawarizm_metal_matvec_tb_half_custom(C, A, B, K, N);
                  uint64_t customNanos = monotonic_nanos() - customStart;
                  if (customRc == 0) {
                    float margin = env_float_or_default(
                        "ALKHAWARIZM_METAL_MPS_MATVEC_AUTOTUNE_MARGIN", 0.05f);
                    if (margin < 0.0f)
                      margin = 0.0f;
                    BOOL preferMps =
                        (double)mpsNanos <
                        ((double)customNanos * (1.0 - (double)margin));
                    alkhawarizm_metal_mps_matvec_record_autotune_preference(
                        shapeKey, preferMps);
                    if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                      fprintf(
                          stderr,
                          "[alkhawarizm-metal] MPS matvec autotune shape K=%d "
                          "N=%d mps=%.3fms custom=%.3fms selected=%s\n",
                          K, N, (double)mpsNanos / 1000000.0,
                          (double)customNanos / 1000000.0,
                          preferMps ? "mps" : "custom");
                    }
                    return 0;
                  }
                  alkhawarizm_metal_mps_matvec_mark_mps_preferred(shapeKey);
                }
                return 0;
              }
              alkhawarizm_metal_mps_matvec_mark_failed(shapeKey);
              if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                fprintf(stderr,
                        "[alkhawarizm-metal] disabling MPS matvec for shape "
                        "K=%d N=%d; falling back to custom Metal reduction\n",
                        K, N);
              }
            } else {
              alkhawarizm_metal_mps_matvec_mark_disable_after_failure();
              if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                NSString *message = cmd.error.localizedDescription;
                fprintf(stderr,
                        "[alkhawarizm-metal] MPS matvec command failed (%s); "
                        "falling back to custom Metal reduction\n",
                        message != nil ? [message UTF8String]
                                       : "unknown error");
              }
            }
          } @catch (NSException *ex) {
            // Fall back to the custom reduction kernel; correctness is more
            // important than this fast path.
            alkhawarizm_metal_mps_matvec_mark_disable_after_failure();
            if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
              NSString *reason = [ex reason];
              fprintf(stderr,
                      "[alkhawarizm-metal] MPS matvec threw %s; falling back "
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
                                ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_half_pipeline,
                                        ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half(C, A, B, K, N,
                                       g_matvec_half_128_pipeline,
                                       ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_half_128_pipeline
                                                : g_matvec_half_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int alkhawarizm_metal_matvec_tb_half_mps(void *C, const void *A, const void *B,
                                         int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  @synchronized(alkhawarizm_metal_mps_matvec_policy_lock()) {
    AlkhawarizmMetalMpsMatvecOverrideSnapshot snapshot =
        alkhawarizm_metal_mps_matvec_force_shape(K, N);
    if (!env_truthy("ALKHAWARIZM_METAL_VALIDATE_LOGITS_MPS_MATVEC") &&
        !env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_VALIDATE_EVERY_CALL")) {
      alkhawarizm_metal_mps_matvec_mark_validated(mps_matvec_shape_key(K, N));
    }
    int rc = alkhawarizm_metal_matvec_tb_half(C, A, B, K, N);

    alkhawarizm_metal_mps_matvec_restore_overrides(snapshot);
    return rc;
  }
}

#define g_matvec_tb_int4_pipeline g_pipelines.matvec_tb_int4
#define g_matvec_tb_int4_128_pipeline g_pipelines.matvec_tb_int4_128

int alkhawarizm_metal_matvec_tb_int4(void *C, const void *A,
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
                                        ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;

    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                        g_matvec_tb_int4_pipeline,
                                        ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;

    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_int4", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                       g_matvec_tb_int4_128_pipeline,
                                       ALKHAWARIZM_MATVEC_THREADS_128);
      }
      return 0;
    }
    if (rc256 == 0)
      return 0;
  } else {
    if (threads == 0) {
      threads = default_matvec_threads(g_matvec_tb_int4_128_pipeline, K, N);
    }
    id<MTLComputePipelineState> pipeline =
        threads == ALKHAWARIZM_MATVEC_THREADS_128
            ? g_matvec_tb_int4_128_pipeline
            : g_matvec_tb_int4_pipeline;
    return dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                   pipeline, threads);
  }
}

#define g_matvec_tb_q4_k_pipeline g_pipelines.matvec_tb_q4_k
#define g_matvec_tb_q6_k_pipeline g_pipelines.matvec_tb_q6_k
#define g_matvec_tb_q8_0_pipeline g_pipelines.matvec_tb_q8_0

int alkhawarizm_metal_matvec_tb_q4_k(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_matvec_tb_q4_k_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q4_k(C, A, B, K, N, g_matvec_tb_q4_k_pipeline,
                                 ALKHAWARIZM_MATVEC_THREADS_256);
}

int alkhawarizm_metal_matvec_tb_q6_k(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_matvec_tb_q6_k_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q6_k(C, A, B, K, N, g_matvec_tb_q6_k_pipeline,
                                 ALKHAWARIZM_MATVEC_THREADS_256);
}

int alkhawarizm_metal_matvec_tb_q8_0(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_matvec_tb_q8_0_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q8_0(C, A, B, K, N, g_matvec_tb_q8_0_pipeline,
                                 ALKHAWARIZM_MATVEC_THREADS_256);
}

int alkhawarizm_metal_matvec_tb_nf4(void *C, const void *A,
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
                                       ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                       g_matvec_tb_nf4_pipeline,
                                       ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_nf4", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                      g_matvec_tb_nf4_128_pipeline,
                                      ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_tb_nf4_128_pipeline
                                                : g_matvec_tb_nf4_pipeline;
  return dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                pipeline, threads);
}

int alkhawarizm_metal_matvec_t_half(void *C, const void *A, const void *B,
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
                               ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_t_half(C, A, B, K, N, g_matvec_t_half_pipeline,
                                       ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("t", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_t_half(C, A, B, K, N,
                                      g_matvec_t_half_128_pipeline,
                                      ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_t_half_128_pipeline
                                                : g_matvec_t_half_pipeline;
  return dispatch_matvec_t_half(C, A, B, K, N, pipeline, threads);
}

static int alkhawarizm_metal_matvec_tb_bf16_custom(void *C, const void *A,
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_bf16_128_pipeline
                                                : g_matvec_bf16_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int alkhawarizm_metal_matvec_tb_bf16(void *C, const void *A, const void *B,
                                     int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  if (should_use_bf16_matvec_x8(K, N) && g_matvec_bf16_x8_pipeline != nil) {
    return dispatch_matvec_tb_half_x8(C, A, B, K, N, g_matvec_bf16_x8_pipeline,
                                      ALKHAWARIZM_MATVEC_THREADS_256);
  }
  if (should_use_simdgroup_reduction() && should_use_bf16_matvec_x4(K, N) &&
      g_matvec_bf16_x4_simd_pipeline != nil) {
    return dispatch_matvec_tb_half_x4(C, A, B, K, N,
                                      g_matvec_bf16_x4_simd_pipeline,
                                      ALKHAWARIZM_MATVEC_THREADS_256);
  }
  if (should_use_bf16_matvec_x4(K, N) && g_matvec_bf16_x4_pipeline != nil) {
    return dispatch_matvec_tb_half_x4(C, A, B, K, N, g_matvec_bf16_x4_pipeline,
                                      ALKHAWARIZM_MATVEC_THREADS_256);
  }

  if (alkhawarizm_metal_mps_bf16_matvec_should_try(K, N)) {
    @autoreleasepool {
      NSString *shapeKey = [NSString stringWithFormat:@"bf16:%d:%d", K, N];
      BOOL validateEveryCall =
          alkhawarizm_metal_mps_matvec_validate_every_call();
      BOOL autotune =
          alkhawarizm_metal_mps_matvec_autotune_enabled_for_output(N);
      AlkhawarizmMetalMpsMatvecShapeState shapeState =
          alkhawarizm_metal_mps_matvec_shape_state(shapeKey, validateEveryCall,
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
              alkhawarizm_metal_ensure_half_input_scratch((size_t)K);
          uint16_t *bf16C =
              alkhawarizm_metal_ensure_half_output_scratch((size_t)N);
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
                          "ALKHAWARIZM_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
                    alkhawarizm_metal_mps_matvec_mark_validated(shapeKey);
                    if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                      fprintf(stderr,
                              "[alkhawarizm-metal] MPS BF16 matvec validated "
                              "shape K=%d N=%d\n",
                              K, N);
                    }
                  }
                  if (autotune && !shapeState.mps_preferred) {
                    uint64_t customStart = monotonic_nanos();
                    int customRc =
                        alkhawarizm_metal_matvec_tb_bf16_custom(C, A, B, K, N);
                    uint64_t customNanos = monotonic_nanos() - customStart;
                    if (customRc == 0) {
                      float margin = env_float_or_default(
                          "ALKHAWARIZM_METAL_MPS_MATVEC_AUTOTUNE_MARGIN",
                          0.05f);
                      if (margin < 0.0f)
                        margin = 0.0f;
                      BOOL preferMps =
                          (double)mpsNanos <
                          ((double)customNanos * (1.0 - (double)margin));
                      alkhawarizm_metal_mps_matvec_record_autotune_preference(
                          shapeKey, preferMps);
                      if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                        fprintf(stderr,
                                "[alkhawarizm-metal] MPS BF16 matvec autotune "
                                "shape K=%d N=%d mps=%.3fms custom=%.3fms "
                                "selected=%s\n",
                                K, N, (double)mpsNanos / 1000000.0,
                                (double)customNanos / 1000000.0,
                                preferMps ? "mps" : "custom");
                      }
                      return 0;
                    }
                    alkhawarizm_metal_mps_matvec_mark_mps_preferred(shapeKey);
                  }
                  return 0;
                }
                alkhawarizm_metal_mps_matvec_mark_failed(shapeKey);
                if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                  fprintf(
                      stderr,
                      "[alkhawarizm-metal] disabling MPS BF16 matvec for shape "
                      "K=%d N=%d; falling back to custom Metal reduction\n",
                      K, N);
                }
              } else {
                alkhawarizm_metal_mps_matvec_mark_disable_after_failure();
                if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                  NSString *message = cmd.error.localizedDescription;
                  fprintf(stderr,
                          "[alkhawarizm-metal] MPS BF16 matvec command failed "
                          "(%s); falling back to custom Metal reduction\n",
                          message != nil ? [message UTF8String]
                                         : "unknown error");
                }
              }
            } @catch (NSException *ex) {
              alkhawarizm_metal_mps_matvec_mark_disable_after_failure();
              if (env_truthy("ALKHAWARIZM_METAL_MPS_MATVEC_DEBUG")) {
                NSString *reason = [ex reason];
                fprintf(stderr,
                        "[alkhawarizm-metal] MPS BF16 matvec threw %s; falling "
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
                                ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half(C, A, B, K, N, g_matvec_bf16_pipeline,
                                        ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("tb_bf16", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half(C, A, B, K, N,
                                       g_matvec_bf16_128_pipeline,
                                       ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128 ? g_matvec_bf16_128_pipeline
                                                : g_matvec_bf16_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int alkhawarizm_metal_matvec_tb_half_pair(void *C0, void *C1, const void *A,
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
                                             ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_half_pair_pipeline,
                                             ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("pair", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                            g_matvec_half_pair_128_pipeline,
                                            ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128
          ? g_matvec_half_pair_128_pipeline
          : g_matvec_half_pair_pipeline;
  return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                      threads);
}

int alkhawarizm_metal_matvec_tb_bf16_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  BOOL usePairX4 = !env_truthy("ALKHAWARIZM_METAL_DISABLE_BF16_PAIR_X4") &&
                   g_matvec_bf16_pair_x4_pipeline != nil &&
                   (env_truthy("ALKHAWARIZM_METAL_ENABLE_BF16_PAIR_X4") ||
                    should_use_bf16_matvec_x4(K, N));
  if (usePairX4) {
    id<MTLComputePipelineState> pipeline =
        should_use_simdgroup_reduction() &&
                g_matvec_bf16_pair_x4_simd_pipeline != nil
            ? g_matvec_bf16_pair_x4_simd_pipeline
            : g_matvec_bf16_pair_x4_pipeline;
    return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                        ALKHAWARIZM_MATVEC_THREADS_256);
  }
  if (should_use_bf16_pair_simd_reduction(K, N) &&
      g_matvec_bf16_pair_simd_pipeline != nil) {
    return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                        g_matvec_bf16_pair_simd_pipeline,
                                        ALKHAWARIZM_MATVEC_THREADS_256);
  }
  NSString *key = matvec_shape_key("pair_bf16", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  BOOL can128 = g_matvec_bf16_pair_128_pipeline != nil;
  if (threads == 0 && matvec_autotune_enabled(K, N, can128) &&
      g_matvec_bf16_pair_pipeline != nil) {
    uint64_t start128 = monotonic_nanos();
    int rc128 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_bf16_pair_128_pipeline,
                                             ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                             g_matvec_bf16_pair_pipeline,
                                             ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("pair_bf16", K, N, 0, 0, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N,
                                            g_matvec_bf16_pair_128_pipeline,
                                            ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128
          ? g_matvec_bf16_pair_128_pipeline
          : g_matvec_bf16_pair_pipeline;
  return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                      threads);
}

int alkhawarizm_metal_matvec_tb_half_triple_mixed(void *C0, void *C1, void *C2,
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
        ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_half_triple_mixed_pipeline, ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("triple", K, N0, N1, N2, nanos128, nanos256, threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_triple_mixed(
            C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
            g_matvec_half_triple_mixed_128_pipeline,
            ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128
          ? g_matvec_half_triple_mixed_128_pipeline
          : g_matvec_half_triple_mixed_pipeline;
  return dispatch_matvec_tb_half_triple_mixed(C0, C1, C2, A, B0, B1, B2, K, N0,
                                              N1, N2, pipeline, threads);
}

int alkhawarizm_metal_matvec_tb_bf16_triple_mixed(void *C0, void *C1, void *C2,
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
        g_matvec_bf16_triple_mixed_x4_pipeline, ALKHAWARIZM_MATVEC_THREADS_256);
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
        ALKHAWARIZM_MATVEC_THREADS_128);
    uint64_t nanos128 = monotonic_nanos() - start128;
    uint64_t start256 = monotonic_nanos();
    int rc256 = dispatch_matvec_tb_half_triple_mixed(
        C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
        g_matvec_bf16_triple_mixed_pipeline, ALKHAWARIZM_MATVEC_THREADS_256);
    uint64_t nanos256 = monotonic_nanos() - start256;
    if (rc128 == 0 && rc256 == 0) {
      threads = matvec_autotune_prefers_128(nanos128, nanos256)
                    ? ALKHAWARIZM_MATVEC_THREADS_128
                    : ALKHAWARIZM_MATVEC_THREADS_256;
      cache_matvec_threads(key, threads);
      log_matvec_autotune("triple_bf16", K, N0, N1, N2, nanos128, nanos256,
                          threads);
      if (threads == ALKHAWARIZM_MATVEC_THREADS_128) {
        return dispatch_matvec_tb_half_triple_mixed(
            C0, C1, C2, A, B0, B1, B2, K, N0, N1, N2,
            g_matvec_bf16_triple_mixed_128_pipeline,
            ALKHAWARIZM_MATVEC_THREADS_128);
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
      threads == ALKHAWARIZM_MATVEC_THREADS_128
          ? g_matvec_bf16_triple_mixed_128_pipeline
          : g_matvec_bf16_triple_mixed_pipeline;
  return dispatch_matvec_tb_half_triple_mixed(C0, C1, C2, A, B0, B1, B2, K, N0,
                                              N1, N2, pipeline, threads);
}
