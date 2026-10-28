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
#import "aljabr_metal_runtime.h"

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

// ── aljabr_metal_softcap
// ────────────────────────────────────────────────────── GPU logit
// soft-capping: x[i] = cap * tanh(x[i] / cap)  (in-place) Uses
// aljabr_softcap_kernel with precise::tanh for correctness. On Gemma-4 12B,
// vocab=262144; 256 threads → 1024 threadgroups.
// ─────────────────────────────────────────────────────────────────────────────

int aljabr_metal_softcap(void *logits, int N, float cap) {
  if (!g_initialized)
    return -1;
  if (N <= 0 || cap <= 0.0f)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->softcap == nil)
    return -3;

  size_t bytes = (size_t)N * sizeof(float);
  size_t page_size = (size_t)getpagesize();
  uintptr_t addr = (uintptr_t)logits;
  uintptr_t offset = addr & (page_size - 1);
  void *aligned_ptr = (void *)(addr - offset);
  size_t aligned_bytes = (bytes + offset + page_size - 1) & ~(page_size - 1);

  id<MTLBuffer> bufLogits =
      [g_device newBufferWithBytesNoCopy:aligned_ptr
                                  length:aligned_bytes
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];
  if (bufLogits == nil) {
    fprintf(stderr, "[aljabr-metal] softcap failed to wrap ptr %p (len=%zu)\n",
            logits, bytes);
    return -4;
  }

  uint32_t uN = (uint32_t)N;

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipelines->softcap];
  [enc setBuffer:bufLogits offset:offset atIndex:0];
  [enc setBytes:&cap length:sizeof(cap) atIndex:1];
  [enc setBytes:&uN length:sizeof(uN) atIndex:2];
  // Dispatch with 256 threads per threadgroup → ceil(N/256) groups
  NSUInteger threads =
      MIN((NSUInteger)256, pipelines->softcap.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];

  if ([cmd status] != MTLCommandBufferStatusCompleted)
    return -5;
  return 0;
}

// ── Matvec C API ─────────────────────────────────────────────────────────────

static int aljabr_metal_matvec_tb_half_custom(void *C, const void *A,
                                              const void *B, int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("tb", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_half_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_half_128_pipeline
                                             : g_matvec_half_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_half(void *C, const void *A, const void *B, int K,
                                int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;

  if (aljabr_metal_mps_matvec_should_try(K, N)) {
    @autoreleasepool {
      NSString *shapeKey = mps_matvec_shape_key(K, N);
      BOOL validateEveryCall = aljabr_metal_mps_matvec_validate_every_call();
      BOOL autotune = aljabr_metal_mps_matvec_autotune_enabled_for_output(N);
      AljabrMetalMpsMatvecShapeState shapeState =
          aljabr_metal_mps_matvec_shape_state(shapeKey, validateEveryCall,
                                              autotune);
      if (!shapeState.failed && !shapeState.custom_preferred) {
        uint64_t mpsStart = autotune ? monotonic_nanos() : 0;
        uint16_t *halfA = aljabr_metal_ensure_half_input_scratch((size_t)K);
        uint16_t *halfC = aljabr_metal_ensure_half_output_scratch((size_t)N);
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
                      fprintf(stderr,
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
    int rc128 = dispatch_matvec_tb_half(
        C, A, B, K, N, g_matvec_half_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_half_128_pipeline
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

int aljabr_metal_matvec_tb_int4(void *C, const void *A, const void *B_packed,
                                const void *scales, int K, int N,
                                int blockSize) {
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
    id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                               ? g_matvec_tb_int4_128_pipeline
                                               : g_matvec_tb_int4_pipeline;
    return dispatch_matvec_tb_int4(C, A, B_packed, scales, K, N, blockSize,
                                   pipeline, threads);
  }
}

#define g_matvec_tb_q4_k_pipeline g_pipelines.matvec_tb_q4_k
#define g_matvec_tb_q6_k_pipeline g_pipelines.matvec_tb_q6_k
#define g_matvec_tb_q8_0_pipeline g_pipelines.matvec_tb_q8_0

static inline BOOL mr8_enabled(void) {
  static int enabled = -1;
  if (enabled == -1) {
    const char *e = getenv("ALJABR_METAL_DISABLE_MR8");
    if (e && (strcmp(e, "1") == 0 || strcmp(e, "true") == 0)) {
      enabled = 0;
    } else {
      enabled = 1;
    }
  }
  return enabled == 1;
}

int aljabr_metal_matvec_tb_q4_k(void *C, const void *A, const void *B, int K,
                                int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_pipelines.matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
    return aljabr_metal_dispatch_matvec_tb_q4_k_mr8(
        C, A, B, K, N, g_pipelines.matvec_tb_q4_k_mr8);
  }
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
  if (rc != 0)
    return rc;
  if (cap > 0.0f) {
    return aljabr_metal_softcap(C, N, cap);
  }
  return 0;
}

int aljabr_metal_matvec_tb_q6_k(void *C, const void *A, const void *B, int K,
                                int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_pipelines.matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
    return aljabr_metal_dispatch_matvec_tb_q6_k_mr8(
        C, A, B, K, N, g_pipelines.matvec_tb_q6_k_mr8);
  }
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
  if (rc != 0)
    return rc;
  if (cap > 0.0f) {
    return aljabr_metal_softcap(C, N, cap);
  }
  return 0;
}

int aljabr_metal_matvec_tb_q8_0(void *C, const void *A, const void *B, int K,
                                int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  if (g_matvec_tb_q8_0_pipeline == nil)
    return -3;
  return dispatch_matvec_tb_q8_0(C, A, B, K, N, g_matvec_tb_q8_0_pipeline,
                                 ALJABR_MATVEC_THREADS_32);
}

int aljabr_metal_matvec_tb_nf4(void *C, const void *A, const void *B_packed,
                               const void *absmax, int K, int N,
                               int blockSize) {
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_tb_nf4_128_pipeline
                                             : g_matvec_tb_nf4_pipeline;
  return dispatch_matvec_tb_nf4(C, A, B_packed, absmax, K, N, blockSize,
                                pipeline, threads);
}

int aljabr_metal_matvec_t_half(void *C, const void *A, const void *B, int K,
                               int N) {
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
    int rc128 = dispatch_matvec_t_half(
        C, A, B, K, N, g_matvec_t_half_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_t_half_128_pipeline
                                             : g_matvec_t_half_pipeline;
  return dispatch_matvec_t_half(C, A, B, K, N, pipeline, threads);
}

static int aljabr_metal_matvec_tb_bf16_custom(void *C, const void *A,
                                              const void *B, int K, int N) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0)
    return -2;
  NSString *key = matvec_shape_key("tb_bf16", K, N, 0, 0);
  NSUInteger threads = cached_matvec_threads(key);
  if (threads == 0) {
    threads = default_matvec_threads(g_matvec_bf16_128_pipeline, K, N);
  }
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_bf16_128_pipeline
                                             : g_matvec_bf16_pipeline;
  return dispatch_matvec_tb_half(C, A, B, K, N, pipeline, threads);
}

int aljabr_metal_matvec_tb_bf16(void *C, const void *A, const void *B, int K,
                                int N) {
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
      BOOL validateEveryCall = aljabr_metal_mps_matvec_validate_every_call();
      BOOL autotune = aljabr_metal_mps_matvec_autotune_enabled_for_output(N);
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
          uint16_t *bf16A = aljabr_metal_ensure_half_input_scratch((size_t)K);
          uint16_t *bf16C = aljabr_metal_ensure_half_output_scratch((size_t)N);
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
                      !env_truthy("ALJABR_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
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
                          "ALJABR_METAL_MPS_MATVEC_AUTOTUNE_MARGIN", 0.05f);
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
                  fprintf(stderr,
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
    int rc128 = dispatch_matvec_tb_half(
        C, A, B, K, N, g_matvec_bf16_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_bf16_128_pipeline
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
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
  id<MTLComputePipelineState> pipeline = threads == ALJABR_MATVEC_THREADS_128
                                             ? g_matvec_bf16_pair_128_pipeline
                                             : g_matvec_bf16_pair_pipeline;
  return dispatch_matvec_tb_half_pair(C0, C1, A, B0, B1, K, N, pipeline,
                                      threads);
}

int aljabr_metal_matvec_tb_half_triple_mixed(void *C0, void *C1, void *C2,
                                             const void *A, const void *B0,
                                             const void *B1, const void *B2,
                                             int K, int N0, int N1, int N2) {
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
        g_matvec_half_triple_mixed_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
            g_matvec_half_triple_mixed_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
                                             const void *B1, const void *B2,
                                             int K, int N0, int N1, int N2) {
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
        g_matvec_bf16_triple_mixed_128_pipeline, ALJABR_MATVEC_THREADS_128);
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
            g_matvec_bf16_triple_mixed_128_pipeline, ALJABR_MATVEC_THREADS_128);
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

/*
 * B4 Q4_K/Q6_K two-token batched kernels are validated up to batch size 32.
 *
 * Historical note: an earlier revision without output-write guards produced a
 * Q4_K mismatch at batch 27, row 15 (expected 0.0078125, got -0.0625) in CI.
 * That kernel had no `if (t_base + 1u < bSz)` guard. The current kernel has
 * correct bounds guards (`t_base >= bSz` early-return, `t_base+1 < bSz` write
 * guard, and `min(t_base+1, bSz-1)` read-clamping). Raising the cap to 32
 * enables B4 dispatch for all prefill batch sizes (16, 24, 27, 32), cutting
 * FFN memory-bandwidth by ~2× vs per-token MR8/MR4 dispatch.
 */
static const uint32_t kMaxValidatedB4BatchSize = 32;

static size_t calc_matvec_weight_bytes(int type, uint32_t N, uint32_t K) {
  if (type == 12)
    return (size_t)N * K / 256 * 144; // Q4_K
  if (type == 14)
    return (size_t)N * K / 256 * 210; // Q6_K
  if (type == 8)
    return (size_t)N * K / 32 * 34; // Q8_0
  if (type == 1 || type == 30 || type == 16)
    return (size_t)N * K * 2; // F16/BF16
  return (size_t)N * K * sizeof(float);
}

static BOOL encode_matvec_buf(id<MTLComputeCommandEncoder> enc,
                              AljabrMetalPipelines *pipelines,
                              id<MTLBuffer> bufC, id<MTLBuffer> bufA,
                              id<MTLBuffer> bufB, size_t bOff,
                              uint32_t kk, uint32_t nn, int type,
                              uint32_t batchSize) {
  id<MTLComputePipelineState> p = nil;
  BOOL is_mr8 = NO;
  BOOL is_mr4 = NO;
  BOOL is_b4 = NO;
  if (type == 12) { // Q4_K
    // The Q4_K B4 kernel currently disagrees with the single-row reference at
    // batch=16 (last row). Keep the optimized path only within the range
    // covered by the current correctness evidence; larger batches use MR4.
    // Do not raise this limit until MetalTripleBatchCorrectnessTest passes on
    // Apple Silicon for all tested batch sizes.
    static const uint32_t kMaxCorrectnessValidatedQ4KB4BatchSize = 15;
    if (batchSize > 1 && batchSize <= kMaxCorrectnessValidatedQ4KB4BatchSize &&
        pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
      p = pipelines->matvec_tb_q4_k_b4_mr4;
      is_b4 = YES;
    } else if (batchSize > kMaxCorrectnessValidatedQ4KB4BatchSize &&
               pipelines->matvec_tb_q4_k_mr4 != nil) {
      // For batches outside the B4-validated range, deliberately bypass MR8:
      // the batch-16 regression persisted after the B4 cap was lowered, so
      // keep this containment path deterministic and use the known single-row
      // MR4 implementation until the failing native path is identified.
      p = pipelines->matvec_tb_q4_k_mr4;
      is_mr4 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      p = pipelines->matvec_tb_q4_k_mr8;
      is_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      p = pipelines->matvec_tb_q4_k_mr4;
      is_mr4 = YES;
    } else {
      p = pipelines->matvec_tb_q4_k;
    }
  } else if (type == 14) { // Q6_K
    if (batchSize > 1 && batchSize <= kMaxValidatedB4BatchSize &&
        pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
      p = pipelines->matvec_tb_q6_k_b4_mr4;
      is_b4 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      p = pipelines->matvec_tb_q6_k_mr8;
      is_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      p = pipelines->matvec_tb_q6_k_mr4;
      is_mr4 = YES;
    } else {
      p = pipelines->matvec_tb_q6_k;
    }
  } else if (type == 8) { // Q8_0
    p = pipelines->matvec_tb_q8_0;
  } else if (type == 1) { // F16
    p = pipelines->matvec_half;
  } else if (type == 30 || type == 16) { // BF16
    p = pipelines->matvec_bf16;
  }
  if (p == nil || bufC == nil || bufB == nil)
    return NO;

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
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nn + 3) / 4,
                                          (bSz + 1) / 2, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  } else if (is_mr8) {
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nn + 7) / 8, bSz, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  } else if (is_mr4) {
    // MR4 maps two SIMD groups (2 × 32 lanes) to four output rows:
    // each SIMD group computes a pair of rows. A 128-thread group would
    // launch four SIMD groups and overlap the next tile's row indices.
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nn + 3) / 4, bSz, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  } else {
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nn, bSz, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }
  return YES;
}

static BOOL encode_matvec_sub(id<MTLComputeCommandEncoder> enc,
                              AljabrMetalPipelines *pipelines,
                              id<MTLBuffer> bufC, id<MTLBuffer> bufA,
                              const void *B, uint32_t kk, uint32_t nn, int type,
                              uint32_t batchSize) {
  size_t bBytes = calc_matvec_weight_bytes(type, nn, kk);
  if (bBytes == 0)
    return NO;
  size_t bOff = 0;
  id<MTLBuffer> bufB = wrap_weight_ptr_offset(B, bBytes, &bOff);
  return encode_matvec_buf(enc, pipelines, bufC, bufA, bufB, bOff, kk, nn, type,
                           batchSize);
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

static inline id<MTLBuffer> ensure_shared_buf(id<MTLBuffer> __strong *slot,
                                              size_t *cap, size_t needed) {
  if (*slot == nil || *cap < needed) {
    size_t sz = (needed + 16383) & ~16383;
    *slot = [g_device newBufferWithLength:sz
                                  options:MTLResourceStorageModeShared];
    *cap = sz;
  }
  return *slot;
}

int aljabr_metal_matvec_tb_batch(void *C, const void *A, const void *B, int K,
                                 int N, int type, int batchSize) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N <= 0 || batchSize <= 0)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t aBytes = (size_t)batchSize * (size_t)K * sizeof(float);
    size_t cBytes = (size_t)batchSize * (size_t)N * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_single_A,
                                           &g_scratch_single_A_bytes, aBytes);
    id<MTLBuffer> bufC = ensure_shared_buf(&g_scratch_single_C,
                                           &g_scratch_single_C_bytes, cBytes);
    if (bufA == nil || bufC == nil)
      return -4;

    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;
    uint32_t bSz = (uint32_t)batchSize;

    if (!encode_matvec_sub(enc, pipelines, bufC, bufA, B, kk, (uint32_t)N, type,
                           bSz)) {
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
                                const void *B0, const void *B1, int K, int N0,
                                int N1, int type0, int type1) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t aBytes = (size_t)K * sizeof(float);
    size_t c0Bytes = (size_t)N0 * sizeof(float);
    size_t c1Bytes = (size_t)N1 * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_triple_A,
                                           &g_scratch_triple_A_bytes, aBytes);
    id<MTLBuffer> bufC0 = ensure_shared_buf(
        &g_scratch_triple_C0, &g_scratch_triple_C0_bytes, c0Bytes);
    id<MTLBuffer> bufC1 = ensure_shared_buf(
        &g_scratch_triple_C1, &g_scratch_triple_C1_bytes, c1Bytes);
    if (bufA == nil || bufC0 == nil || bufC1 == nil)
      return -4;

    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;

    if (!encode_matvec_sub(enc, pipelines, bufC0, bufA, B0, kk, (uint32_t)N0,
                           type0, 1) ||
        !encode_matvec_sub(enc, pipelines, bufC1, bufA, B1, kk, (uint32_t)N1,
                           type1, 1)) {
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
                                        const void *B1, const void *B2, int K,
                                        int N0, int N1, int N2, int type0,
                                        int type1, int type2, int batchSize) {
  if (!g_initialized)
    return -1;
  if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0 || batchSize <= 0)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    size_t bSz = (size_t)batchSize;
    size_t aBytes = bSz * (size_t)K * sizeof(float);
    size_t c0Bytes = bSz * (size_t)N0 * sizeof(float);
    size_t c1Bytes = bSz * (size_t)N1 * sizeof(float);
    size_t c2Bytes = bSz * (size_t)N2 * sizeof(float);

    id<MTLBuffer> bufA = ensure_shared_buf(&g_scratch_triple_A,
                                           &g_scratch_triple_A_bytes, aBytes);
    id<MTLBuffer> bufC0 = ensure_shared_buf(
        &g_scratch_triple_C0, &g_scratch_triple_C0_bytes, c0Bytes);
    id<MTLBuffer> bufC1 = ensure_shared_buf(
        &g_scratch_triple_C1, &g_scratch_triple_C1_bytes, c1Bytes);
    id<MTLBuffer> bufC2 = ensure_shared_buf(
        &g_scratch_triple_C2, &g_scratch_triple_C2_bytes, c2Bytes);
    if (bufA == nil || bufC0 == nil || bufC1 == nil || bufC2 == nil)
      return -4;

    uint64_t t_start = monotonic_nanos();
    memcpy([bufA contents], A, aBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t kk = (uint32_t)K;
    uint32_t batch32 = (uint32_t)batchSize;

    if (!encode_matvec_sub(enc, pipelines, bufC0, bufA, B0, kk, (uint32_t)N0,
                           type0, batch32) ||
        !encode_matvec_sub(enc, pipelines, bufC1, bufA, B1, kk, (uint32_t)N1,
                           type1, batch32) ||
        !encode_matvec_sub(enc, pipelines, bufC2, bufA, B2, kk, (uint32_t)N2,
                           type2, batch32)) {
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
        ALJABR_METAL_LOG("[NATIVE TRIPLE BATCH #%d] K=%d N0=%d N1=%d N2=%d bSz=%d "
                         "types=(%d,%d,%d) prep: %.2f ms, gpu: %.2f ms, copy_out: %.2f "
                         "ms, total: %.2f ms\n",
                         triple_batch_call_count, K, N0, N1, N2, batchSize, type0, type1,
                         type2, (t_commit - t_start) / 1e6, (t_done - t_commit) / 1e6,
                         (t_end - t_done) / 1e6, (t_end - t_start) / 1e6);
      }
      return 0;
    }
    return -1;
  }
}

int aljabr_metal_matvec_tb_triple(void *C0, void *C1, void *C2, const void *A,
                                  const void *B0, const void *B1,
                                  const void *B2, int K, int N0, int N1, int N2,
                                  int type0, int type1, int type2) {
  return aljabr_metal_matvec_tb_triple_batch(C0, C1, C2, A, B0, B1, B2, K, N0,
                                             N1, N2, type0, type1, type2, 1);
}

static id<MTLBuffer> g_scratch_gate = nil;
static id<MTLBuffer> g_scratch_up = nil;
static id<MTLBuffer> g_scratch_hb = nil;
static id<MTLBuffer> g_scratch_inA = nil;
static id<MTLBuffer> g_scratch_outDown = nil;
static size_t g_scratch_ffn_bytes = 0;
static size_t g_scratch_dim_bytes = 0;

int aljabr_metal_ffn_gated_fused_batch(void *outDown, const void *inA,
                                       const void *gateW, const void *upW,
                                       const void *downW, int dim, int ffnDim,
                                       int typeGate, int typeUp, int typeDown,
                                       int activationKind, int batchSize) {
  if (!g_initialized)
    return -1;
  if (dim <= 0 || ffnDim <= 0 || batchSize <= 0)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  @autoreleasepool {
    uint64_t t0 = monotonic_nanos();
    size_t bSz = (size_t)batchSize;
    size_t dimBytes = bSz * (size_t)dim * sizeof(float);
    if (g_scratch_inA == nil || g_scratch_dim_bytes < dimBytes) {
      size_t sz = (dimBytes + 16383) & ~16383;
      g_scratch_inA =
          [g_device newBufferWithLength:sz
                                options:MTLResourceStorageModeShared];
      g_scratch_outDown =
          [g_device newBufferWithLength:sz
                                options:MTLResourceStorageModeShared];
      g_scratch_dim_bytes = sz;
    }
    if (g_scratch_inA == nil || g_scratch_outDown == nil)
      return -4;

    memcpy([g_scratch_inA contents], inA, dimBytes);

    size_t needed = bSz * (size_t)ffnDim * sizeof(float);
    if (g_scratch_gate == nil || g_scratch_ffn_bytes < needed) {
      size_t sz = (needed + 16383) & ~16383;
      g_scratch_gate =
          [g_device newBufferWithLength:sz
                                options:MTLResourceStorageModePrivate];
      g_scratch_up =
          [g_device newBufferWithLength:sz
                                options:MTLResourceStorageModePrivate];
      g_scratch_hb =
          [g_device newBufferWithLength:sz
                                options:MTLResourceStorageModePrivate];
      g_scratch_ffn_bytes = sz;
    }
    if (g_scratch_gate == nil || g_scratch_up == nil || g_scratch_hb == nil)
      return -4;

    id<MTLComputePipelineState> actPipeline =
        (activationKind == 2) ? pipelines->gelu_ffn : pipelines->silu_ffn;

    // Resolve weight buffers
    size_t gateBytes = 0, upBytes = 0, downBytes = 0;
    id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
    BOOL gate_mr8 = NO, up_mr8 = NO, down_mr8 = NO;
    BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

    // Gate
    if (typeGate == 12) {
      if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
        pGate = pipelines->matvec_tb_q4_k_mr8;
        gate_mr8 = YES;
      } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
        pGate = pipelines->matvec_tb_q4_k_mr4;
        gate_mr4 = YES;
      } else {
        pGate = pipelines->matvec_tb_q4_k;
      }
      gateBytes = (size_t)ffnDim * dim / 256 * 144;
    } else if (typeGate == 14) {
      if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
        pGate = pipelines->matvec_tb_q6_k_mr8;
        gate_mr8 = YES;
      } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
        pGate = pipelines->matvec_tb_q6_k_mr4;
        gate_mr4 = YES;
      } else {
        pGate = pipelines->matvec_tb_q6_k;
      }
      gateBytes = (size_t)ffnDim * dim / 256 * 210;
    } else if (typeGate == 8) {
      pGate = pipelines->matvec_tb_q8_0;
      gateBytes = (size_t)ffnDim * dim / 32 * 34;
    } else if (typeGate == 1) {
      pGate = pipelines->matvec_half;
      gateBytes = (size_t)ffnDim * dim * sizeof(uint16_t);
    } else if (typeGate == 30 || typeGate == 16) {
      pGate = pipelines->matvec_bf16;
      gateBytes = (size_t)ffnDim * dim * sizeof(uint16_t);
    }

    // Up
    if (typeUp == 12) {
      if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
        pUp = pipelines->matvec_tb_q4_k_mr8;
        up_mr8 = YES;
      } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
        pUp = pipelines->matvec_tb_q4_k_mr4;
        up_mr4 = YES;
      } else {
        pUp = pipelines->matvec_tb_q4_k;
      }
      upBytes = (size_t)ffnDim * dim / 256 * 144;
    } else if (typeUp == 14) {
      if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
        pUp = pipelines->matvec_tb_q6_k_mr8;
        up_mr8 = YES;
      } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
        pUp = pipelines->matvec_tb_q6_k_mr4;
        up_mr4 = YES;
      } else {
        pUp = pipelines->matvec_tb_q6_k;
      }
      upBytes = (size_t)ffnDim * dim / 256 * 210;
    } else if (typeUp == 8) {
      pUp = pipelines->matvec_tb_q8_0;
      upBytes = (size_t)ffnDim * dim / 32 * 34;
    } else if (typeUp == 1) {
      pUp = pipelines->matvec_half;
      upBytes = (size_t)ffnDim * dim * sizeof(uint16_t);
    } else if (typeUp == 30 || typeUp == 16) {
      pUp = pipelines->matvec_bf16;
      upBytes = (size_t)ffnDim * dim * sizeof(uint16_t);
    }

    // Down: matrix is (dim rows) x (ffnDim cols)
    if (typeDown == 12) {
      if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
        pDown = pipelines->matvec_tb_q4_k_mr8;
        down_mr8 = YES;
      } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
        pDown = pipelines->matvec_tb_q4_k_mr4;
        down_mr4 = YES;
      } else {
        pDown = pipelines->matvec_tb_q4_k;
      }
      downBytes = (size_t)dim * ffnDim / 256 * 144;
    } else if (typeDown == 14) {
      if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
        pDown = pipelines->matvec_tb_q6_k_mr8;
        down_mr8 = YES;
      } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
        pDown = pipelines->matvec_tb_q6_k_mr4;
        down_mr4 = YES;
      } else {
        pDown = pipelines->matvec_tb_q6_k;
      }
      downBytes = (size_t)dim * ffnDim / 256 * 210;
    } else if (typeDown == 8) {
      pDown = pipelines->matvec_tb_q8_0;
      downBytes = (size_t)dim * ffnDim / 32 * 34;
    } else if (typeDown == 1) {
      pDown = pipelines->matvec_half;
      downBytes = (size_t)dim * ffnDim * sizeof(uint16_t);
    } else if (typeDown == 30 || typeDown == 16) {
      pDown = pipelines->matvec_bf16;
      downBytes = (size_t)dim * ffnDim * sizeof(uint16_t);
    }

    BOOL can_fuse_gate_up =
        ((typeGate == 12 && typeUp == 12 &&
          ((pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled()) ||
           pipelines->ffn_gate_up_act_q4_k_mr4 != nil)) ||
         (typeGate == 14 && typeUp == 14 &&
          pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
    if (!can_fuse_gate_up && actPipeline == nil) {
      fprintf(stderr,
              "[aljabr-metal] actPipeline is nil for activationKind=%d\n",
              activationKind);
      return -3;
    }
    if (pGate == nil || pUp == nil || pDown == nil) {
      fprintf(stderr,
              "[aljabr-metal] pipeline nil: pGate=%p pUp=%p pDown=%p (types: "
              "%d, %d, %d)\n",
              pGate, pUp, pDown, typeGate, typeUp, typeDown);
      return -3;
    }
    size_t gateOff = 0, upOff = 0, downOff = 0;
    id<MTLBuffer> bufGateW = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUpW = wrap_weight_ptr_offset(upW, upBytes, &upOff);
    id<MTLBuffer> bufDownW = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    if (bufGateW == nil || bufUpW == nil || bufDownW == nil)
      return -4;

    uint32_t kDim = (uint32_t)dim;
    uint32_t nFfn = (uint32_t)ffnDim;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

    if (can_fuse_gate_up) {
      // 1. Fused Gate + Up + Activation (GeGLU / SwiGLU) directly into
      // g_scratch_hb
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      BOOL use_b4 =
          (bSz > 1 && bSz <= kMaxValidatedB4BatchSize &&
           ((typeGate == 12 && typeUp == 12 &&
             pipelines->ffn_gate_up_act_q4_k_b4_mr4 != nil) ||
            (typeGate == 14 && typeUp == 14 &&
             pipelines->ffn_gate_up_act_q6_k_b4_mr4 != nil)));
      BOOL use_mr8 = (!use_b4 && typeGate == 12 &&
                      pipelines->ffn_gate_up_act_q4_k_mr8 != nil);
      id<MTLComputePipelineState> fuseP;
      if (use_b4) {
        fuseP = (typeGate == 14) ? pipelines->ffn_gate_up_act_q6_k_b4_mr4
                                 : pipelines->ffn_gate_up_act_q4_k_b4_mr4;
      } else if (use_mr8) {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr8;
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
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4,
                                              (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else if (use_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, bSz,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 2. MatVec Down directly from g_scratch_hb to g_scratch_outDown
      if (bSz > 1 && bSz <= kMaxValidatedB4BatchSize && typeDown == 12 &&
          pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q4_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4,
                                              (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else if (bSz > 1 && bSz <= kMaxValidatedB4BatchSize && typeDown == 14 &&
                 pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q6_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4,
                                              (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc setComputePipelineState:pDown];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        if (down_mr8) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 7) / 8, bSz,
                                                1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (down_mr4) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, bSz,
                                                1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
          [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)dim, bSz, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
      }
      [enc endEncoding];
    } else {
      // High-occupancy 1-encoder FFN pipeline: Gate + Up + Act + Down
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      // 1. Gate
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_scratch_gate offset:0 atIndex:0];
      [enc setBuffer:g_scratch_inA offset:0 atIndex:1];
      [enc setBuffer:bufGateW offset:gateOff atIndex:2];
      [enc setBytes:&kDim length:sizeof(kDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (gate_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, bSz,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz,
                                              1)
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
      if (up_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, bSz,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, bSz,
                                              1)
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
      NSUInteger threads =
          MIN((NSUInteger)256, actPipeline.maxTotalThreadsPerThreadgroup);
      NSUInteger totalElem = bSz * (NSUInteger)ffnDim;
      NSUInteger groups = (totalElem + threads - 1) / threads;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 4. MatVec Down: (dim rows) x (ffnDim cols)
      if (bSz > 1 && bSz <= kMaxValidatedB4BatchSize && typeDown == 12 &&
          pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q4_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4,
                                              (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else if (bSz > 1 && bSz <= kMaxValidatedB4BatchSize && typeDown == 14 &&
                 pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
        [enc setComputePipelineState:pipelines->matvec_tb_q6_k_b4_mr4];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        uint32_t bSz32 = (uint32_t)bSz;
        [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4,
                                              (bSz + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc setComputePipelineState:pDown];
        [enc setBuffer:g_scratch_outDown offset:0 atIndex:0];
        [enc setBuffer:g_scratch_hb offset:0 atIndex:1];
        [enc setBuffer:bufDownW offset:downOff atIndex:2];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
        [enc setBytes:&kDim length:sizeof(kDim) atIndex:4];
        if (down_mr8) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 7) / 8, bSz,
                                                1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (down_mr4) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim + 3) / 4, bSz,
                                                1)
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
    if (getenv("ALJABR_METAL_DEBUG_FFN") != NULL &&
        (ffn_call_count < 10 || (ffn_call_count % 48 == 0))) {
      printf("[NATIVE FFN %d] bSz=%d, wrap+encode: %.2f ms, gpu_wait: %.2f ms, "
             "total: %.2f ms (mr4: gate=%d up=%d down=%d)\n",
             ffn_call_count, (int)bSz, (t_encoded - t0) / 1e6,
             (t_done - t_encoded) / 1e6, (t_done - t0) / 1e6, (int)gate_mr4,
             (int)up_mr4, (int)down_mr4);
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
                                 const void *gateW, const void *upW,
                                 const void *downW, int dim, int ffnDim,
                                 int typeGate, int typeUp, int typeDown,
                                 int activationKind) {
  return aljabr_metal_ffn_gated_fused_batch(outDown, inA, gateW, upW, downW,
                                            dim, ffnDim, typeGate, typeUp,
                                            typeDown, activationKind, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// aljabr_metal_wo_ffn_fused — fused wO + norms + residuals + FFN
// Eliminates 2 waitUntilCompleted per layer → saves ~28 ms per decode token
// ─────────────────────────────────────────────────────────────────────────────

static id<MTLBuffer> g_woff_attnFull = nil;
static id<MTLBuffer> g_woff_xb = nil;
static id<MTLBuffer> g_woff_x = nil;
static id<MTLBuffer> g_woff_normW1 = nil;
static id<MTLBuffer> g_woff_normW2 = nil;
static id<MTLBuffer> g_woff_normW3 = nil;
static id<MTLBuffer> g_woff_ffn_gate = nil;
static id<MTLBuffer> g_woff_ffn_up = nil;
static id<MTLBuffer> g_woff_ffn_hb = nil;
static size_t g_woff_attnFull_cap = 0;
static size_t g_woff_dim_cap = 0;
static size_t g_woff_ffn_cap = 0;
static BOOL g_woff_x_valid = NO;

static id<MTLBuffer> g_woff_nextQ = nil;
static id<MTLBuffer> g_woff_nextK = nil;
static id<MTLBuffer> g_woff_nextV = nil;
static id<MTLBuffer> g_woff_nextAttnNormW = nil;
static size_t g_woff_nextQ_cap = 0;
static size_t g_woff_nextK_cap = 0;
static size_t g_woff_nextV_cap = 0;
static size_t g_woff_nextAttnNormW_cap = 0;

static BOOL ensure_woff_next_qkv(size_t qBytes, size_t kBytes, size_t vBytes,
                                 size_t normBytes) {
  if (qBytes > g_woff_nextQ_cap) {
    size_t sz = (qBytes + 16383) & ~16383;
    g_woff_nextQ = [g_device newBufferWithLength:sz
                                         options:MTLResourceStorageModeShared];
    g_woff_nextQ_cap = sz;
    if (!g_woff_nextQ)
      return NO;
  }
  if (kBytes > g_woff_nextK_cap) {
    size_t sz = (kBytes + 16383) & ~16383;
    g_woff_nextK = [g_device newBufferWithLength:sz
                                         options:MTLResourceStorageModeShared];
    g_woff_nextK_cap = sz;
    if (!g_woff_nextK)
      return NO;
  }
  if (vBytes > g_woff_nextV_cap) {
    size_t sz = (vBytes + 16383) & ~16383;
    g_woff_nextV = [g_device newBufferWithLength:sz
                                         options:MTLResourceStorageModeShared];
    g_woff_nextV_cap = sz;
    if (!g_woff_nextV)
      return NO;
  }
  if (normBytes > g_woff_nextAttnNormW_cap) {
    size_t sz = (normBytes + 16383) & ~16383;
    g_woff_nextAttnNormW =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_nextAttnNormW_cap = sz;
    if (!g_woff_nextAttnNormW)
      return NO;
  }
  return YES;
}

static BOOL ensure_woff_dim_bufs(size_t attnKBytes, size_t dimBytes) {
  if (attnKBytes > g_woff_attnFull_cap) {
    size_t sz = (attnKBytes + 16383) & ~16383;
    g_woff_attnFull =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_attnFull_cap = sz;
    if (!g_woff_attnFull)
      return NO;
  }
  if (dimBytes > g_woff_dim_cap) {
    size_t sz = (dimBytes + 16383) & ~16383;
    g_woff_xb = [g_device newBufferWithLength:sz
                                      options:MTLResourceStorageModeShared];
    g_woff_x = [g_device newBufferWithLength:sz
                                     options:MTLResourceStorageModeShared];
    g_woff_normW1 = [g_device newBufferWithLength:sz
                                          options:MTLResourceStorageModeShared];
    g_woff_normW2 = [g_device newBufferWithLength:sz
                                          options:MTLResourceStorageModeShared];
    g_woff_normW3 = [g_device newBufferWithLength:sz
                                          options:MTLResourceStorageModeShared];
    g_woff_dim_cap = sz;
    if (!g_woff_xb || !g_woff_x || !g_woff_normW1 || !g_woff_normW2 ||
        !g_woff_normW3)
      return NO;
  }
  return YES;
}

static BOOL ensure_woff_ffn_bufs(size_t ffnBytes) {
  if (ffnBytes > g_woff_ffn_cap) {
    size_t sz = (ffnBytes + 16383) & ~16383;
    g_woff_ffn_gate =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_up =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_hb =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    g_woff_ffn_cap = sz;
    if (!g_woff_ffn_gate || !g_woff_ffn_up || !g_woff_ffn_hb)
      return NO;
  }
  return YES;
}

static void encode_woff_rmsnorm(id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> bufOut, id<MTLBuffer> bufIn,
                                id<MTLBuffer> bufW, size_t wOffset, uint32_t N,
                                float eps, uint32_t addOne,
                                AljabrMetalPipelines *pipelines) {
  id<MTLComputePipelineState> p =
      (N % 4 == 0 && pipelines->rmsnorm_float4 != nil)
          ? pipelines->rmsnorm_float4
          : pipelines->rmsnorm;
  if (p == nil)
    return;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufOut offset:0 atIndex:0];
  [enc setBuffer:bufIn offset:0 atIndex:1];
  [enc setBuffer:bufW offset:wOffset atIndex:2];
  [enc setBytes:&N length:sizeof(N) atIndex:3];
  [enc setBytes:&eps length:sizeof(eps) atIndex:4];
  [enc setBytes:&addOne length:sizeof(addOne) atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_woff_add(id<MTLComputeCommandEncoder> enc,
                            id<MTLBuffer> bufC, id<MTLBuffer> bufA,
                            id<MTLBuffer> bufB, uint32_t N,
                            AljabrMetalPipelines *pipelines) {
  id<MTLComputePipelineState> p = pipelines->add;
  if (p == nil)
    return;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufC offset:0 atIndex:0];
  [enc setBuffer:bufA offset:0 atIndex:1];
  [enc setBuffer:bufB offset:0 atIndex:2];
  [enc setBytes:&N length:sizeof(N) atIndex:3];
  NSUInteger threads = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void encode_post_attn_add_ffn_norm(
    id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bufXB, id<MTLBuffer> bufX,
    id<MTLBuffer> bufPostAttnW, size_t postAttnOff, id<MTLBuffer> bufFfnNormW,
    size_t ffnNormOff, uint32_t N, float eps, BOOL hasPostAttn,
    AljabrMetalPipelines *pipelines, uint32_t batchSize) {
  id<MTLComputePipelineState> p = pipelines->post_attn_add_ffn_norm;
  if (p == nil || bufXB == nil || bufX == nil || bufFfnNormW == nil)
    return;
  uint32_t hasPost = (hasPostAttn && bufPostAttnW != nil) ? 1 : 0;
  id<MTLBuffer> postW = (hasPost && bufPostAttnW != nil) ? bufPostAttnW : bufXB;
  size_t pOff = hasPost ? postAttnOff : 0;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufXB offset:0 atIndex:0];
  [enc setBuffer:bufX offset:0 atIndex:1];
  [enc setBuffer:postW offset:pOff atIndex:2];
  [enc setBuffer:bufFfnNormW offset:ffnNormOff atIndex:3];
  [enc setBytes:&N length:sizeof(N) atIndex:4];
  [enc setBytes:&eps length:sizeof(eps) atIndex:5];
  [enc setBytes:&hasPost length:sizeof(hasPost) atIndex:6];
  NSUInteger bSz = (batchSize > 0) ? (NSUInteger)batchSize : 1;
  [enc dispatchThreadgroups:MTLSizeMake(1, bSz, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_post_ffn_add_out_scale(
    id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bufXB, id<MTLBuffer> bufX,
    id<MTLBuffer> bufPostFfnW, size_t postFfnOff, uint32_t N, float eps,
    BOOL hasPostFfn, float outScale, AljabrMetalPipelines *pipelines,
    uint32_t batchSize) {
  id<MTLComputePipelineState> p = pipelines->post_ffn_add_out_scale;
  if (p == nil || bufXB == nil || bufX == nil)
    return;
  uint32_t hasPost = (hasPostFfn && bufPostFfnW != nil) ? 1 : 0;
  id<MTLBuffer> postW = (hasPost && bufPostFfnW != nil) ? bufPostFfnW : bufXB;
  size_t pOff = hasPost ? postFfnOff : 0;
  [enc setComputePipelineState:p];
  [enc setBuffer:bufXB offset:0 atIndex:0];
  [enc setBuffer:bufX offset:0 atIndex:1];
  [enc setBuffer:postW offset:pOff atIndex:2];
  [enc setBytes:&N length:sizeof(N) atIndex:3];
  [enc setBytes:&eps length:sizeof(eps) atIndex:4];
  [enc setBytes:&hasPost length:sizeof(hasPost) atIndex:5];
  [enc setBytes:&outScale length:sizeof(outScale) atIndex:6];
  NSUInteger bSz = (batchSize > 0) ? (NSUInteger)batchSize : 1;
  [enc dispatchThreadgroups:MTLSizeMake(1, bSz, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int aljabr_metal_wo_ffn_next_qkv_fused(
    float *x, const float *attn_full, const void *wO, int typeWO, int wO_K,
    int wO_N, const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW, const float *postFfnW, int hasPostFfnNorm, float eps,
    float outScale, const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown, int ffnDim, int activationKind,
    int hasNext, const float *nextAttnNormW, float *nextQ, float *nextK,
    float *nextV, const void *nextWQ, const void *nextWK, const void *nextWV,
    int nextTypeQ, int nextTypeK, int nextTypeV, int nextN0, int nextN1,
    int nextN2) {
  if (!g_initialized)
    return -1;
  if (wO_K <= 0 || wO_N <= 0 || ffnDim <= 0)
    return -2;

  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->rmsnorm == nil && pipelines->rmsnorm_float4 == nil)
    return -3;
  if (pipelines->add == nil)
    return -3;

  // ── Resolve wO pipeline ────────────────────────────────────────────────────
  id<MTLComputePipelineState> pWO = nil;
  size_t wOBytes = 0;
  BOOL wO_mr8 = NO, wO_mr4 = NO;
  if (typeWO == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pWO = pipelines->matvec_tb_q4_k_mr8;
      wO_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pWO = pipelines->matvec_tb_q4_k_mr4;
      wO_mr4 = YES;
    } else
      pWO = pipelines->matvec_tb_q4_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 144;
  } else if (typeWO == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pWO = pipelines->matvec_tb_q6_k_mr8;
      wO_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pWO = pipelines->matvec_tb_q6_k_mr4;
      wO_mr4 = YES;
    } else
      pWO = pipelines->matvec_tb_q6_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 210;
  } else if (typeWO == 8) {
    pWO = pipelines->matvec_tb_q8_0;
    wOBytes = (size_t)wO_N * wO_K / 32 * 34;
  } else if (typeWO == 1) {
    pWO = pipelines->matvec_half;
    wOBytes = (size_t)wO_N * wO_K * 2;
  } else if (typeWO == 30 || typeWO == 16) {
    pWO = pipelines->matvec_bf16;
    wOBytes = (size_t)wO_N * wO_K * 2;
  }
  if (pWO == nil)
    return -3;

  // ── Resolve FFN pipelines ──────────────────────────────────────────────────
  BOOL can_fuse_gate_up =
      ((typeGate == 12 && typeUp == 12 &&
        ((pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled()) ||
         pipelines->ffn_gate_up_act_q4_k_mr4 != nil)) ||
       (typeGate == 14 && typeUp == 14 &&
        pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
  BOOL gate_up_mr8 =
      (typeGate == 12 && typeUp == 12 &&
       pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled());
  id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
  size_t gateBytes = 0, upBytes = 0, downBytes = 0;
  BOOL gate_mr8 = NO, up_mr8 = NO, down_mr8 = NO;
  BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

  if (typeGate == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pGate = pipelines->matvec_tb_q4_k_mr8;
      gate_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pGate = pipelines->matvec_tb_q4_k_mr4;
      gate_mr4 = YES;
    } else
      pGate = pipelines->matvec_tb_q4_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeGate == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pGate = pipelines->matvec_tb_q6_k_mr8;
      gate_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pGate = pipelines->matvec_tb_q6_k_mr4;
      gate_mr4 = YES;
    } else
      pGate = pipelines->matvec_tb_q6_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 210;
  } else if (typeGate == 8) {
    pGate = pipelines->matvec_tb_q8_0;
    gateBytes = (size_t)ffnDim * wO_N / 32 * 34;
  } else if (typeGate == 1) {
    pGate = pipelines->matvec_half;
    gateBytes = (size_t)ffnDim * wO_N * 2;
  } else if (typeGate == 30 || typeGate == 16) {
    pGate = pipelines->matvec_bf16;
    gateBytes = (size_t)ffnDim * wO_N * 2;
  }

  if (typeUp == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pUp = pipelines->matvec_tb_q4_k_mr8;
      up_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pUp = pipelines->matvec_tb_q4_k_mr4;
      up_mr4 = YES;
    } else
      pUp = pipelines->matvec_tb_q4_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeUp == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pUp = pipelines->matvec_tb_q6_k_mr8;
      up_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pUp = pipelines->matvec_tb_q6_k_mr4;
      up_mr4 = YES;
    } else
      pUp = pipelines->matvec_tb_q6_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 210;
  } else if (typeUp == 8) {
    pUp = pipelines->matvec_tb_q8_0;
    upBytes = (size_t)ffnDim * wO_N / 32 * 34;
  } else if (typeUp == 1) {
    pUp = pipelines->matvec_half;
    upBytes = (size_t)ffnDim * wO_N * 2;
  } else if (typeUp == 30 || typeUp == 16) {
    pUp = pipelines->matvec_bf16;
    upBytes = (size_t)ffnDim * wO_N * 2;
  }

  if (typeDown == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pDown = pipelines->matvec_tb_q6_k_mr8;
      down_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pDown = pipelines->matvec_tb_q6_k_mr4;
      down_mr4 = YES;
    } else
      pDown = pipelines->matvec_tb_q6_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 210;
  } else if (typeDown == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pDown = pipelines->matvec_tb_q4_k_mr8;
      down_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pDown = pipelines->matvec_tb_q4_k_mr4;
      down_mr4 = YES;
    } else
      pDown = pipelines->matvec_tb_q4_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 144;
  } else if (typeDown == 8) {
    pDown = pipelines->matvec_tb_q8_0;
    downBytes = (size_t)wO_N * ffnDim / 32 * 34;
  } else if (typeDown == 1) {
    pDown = pipelines->matvec_half;
    downBytes = (size_t)wO_N * ffnDim * 2;
  } else if (typeDown == 30 || typeDown == 16) {
    pDown = pipelines->matvec_bf16;
    downBytes = (size_t)wO_N * ffnDim * 2;
  }

  if (!can_fuse_gate_up && (pGate == nil || pUp == nil))
    return -3;
  if (pDown == nil)
    return -3;

  // ── Ensure scratch buffers ─────────────────────────────────────────────────
  size_t attnKBytes = (size_t)wO_K * sizeof(float);
  size_t dimBytes = (size_t)wO_N * sizeof(float);
  size_t ffnBytes = (size_t)ffnDim * sizeof(float);
  if (!ensure_woff_dim_bufs(attnKBytes, dimBytes))
    return -4;
  if (!ensure_woff_ffn_bufs(ffnBytes))
    return -4;

  if (hasNext && nextWQ != NULL) {
    size_t qBytes = (size_t)nextN0 * sizeof(float);
    size_t kBytes = (size_t)nextN1 * sizeof(float);
    size_t vBytes = (size_t)nextN2 * sizeof(float);
    if (!ensure_woff_next_qkv(qBytes, kBytes, vBytes, dimBytes))
      return -4;
  }

  @autoreleasepool {
    // ── Copy inputs to GPU-accessible Shared buffers ───────────────────────
    memcpy([g_woff_attnFull contents], attn_full, attnKBytes);
    if (!g_woff_x_valid) {
      memcpy([g_woff_x contents], x, dimBytes);
      g_woff_x_valid = YES;
    }
    // Copy norm weight data into pre-allocated shared scratch buffers.
    // (Norms are dequantized Java-heap float[] arrays — not in the master Metal
    // model
    //  buffer — so zero-copy wrapping is not possible. Direct memcpy into the
    //  small pre-allocated g_woff_normW* buffers is cheaper than MTLBuffer
    //  allocation.)
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
    id<MTLBuffer> bufWO = wrap_weight_ptr_offset(wO, wOBytes, &wOOff);
    id<MTLBuffer> bufGate = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUp = wrap_weight_ptr_offset(upW, upBytes, &upOff);
    id<MTLBuffer> bufDown = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    if (!bufWO || !bufGate || !bufUp || !bufDown)
      return -4;

    uint32_t kDim = (uint32_t)wO_K;
    uint32_t nDim = (uint32_t)wO_N;
    uint32_t nFfn = (uint32_t)ffnDim;
    uint32_t actK = (uint32_t)activationKind;

    // ── Single command buffer, single encoder ──────────────────────────────
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Step 1: xb = wO(attnFull)
    [enc setComputePipelineState:pWO];
    [enc setBuffer:g_woff_xb offset:0 atIndex:0];
    [enc setBuffer:g_woff_attnFull offset:0 atIndex:1];
    [enc setBuffer:bufWO offset:wOOff atIndex:2];
    [enc setBytes:&kDim length:sizeof(kDim) atIndex:3];
    [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
    if (wO_mr8) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (wO_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 2 (optional): xb = postAttnNorm(xb)
    if (hasPostAttnNorm && bufPostAttn) {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostAttn, postAttnOff,
                          nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 3: x += xb  (attn residual)
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 4: xb = ffnNorm(x)
    {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufFfnNorm, ffnNormOff,
                          nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 5+6: FFN gate+up+act → hb
    if (can_fuse_gate_up) {
      id<MTLComputePipelineState> fuseP;
      if (gate_up_mr8) {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr8;
      } else if (typeGate == 14) {
        fuseP = pipelines->ffn_gate_up_act_q6_k_mr4;
      } else {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr4;
      }
      [enc setComputePipelineState:fuseP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBuffer:bufUp offset:upOff atIndex:3];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
      [enc setBytes:&actK length:sizeof(actK) atIndex:6];
      if (gate_up_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
    } else {
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (gate_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
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
      if (up_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      id<MTLComputePipelineState> actP =
          (activationKind == 2) ? pipelines->gelu_ffn : pipelines->silu_ffn;
      if (actP == nil) {
        [enc endEncoding];
        return -3;
      }
      [enc setComputePipelineState:actP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:1];
      [enc setBuffer:g_woff_ffn_up offset:0 atIndex:2];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
      NSUInteger actTh =
          MIN((NSUInteger)256, actP.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(actTh, 1, 1)];
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 7: xb = down(hb)   [FFN down projection]
    [enc setComputePipelineState:pDown];
    [enc setBuffer:g_woff_xb offset:0 atIndex:0];
    [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:1];
    [enc setBuffer:bufDown offset:downOff atIndex:2];
    [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
    [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
    if (down_mr8) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (down_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Step 8 (optional): xb = postFfnNorm(xb)
    if (hasPostFfnNorm && bufPostFfn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostFfn, postFfnOff,
                          nDim, eps, 0, pipelines);
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
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:2];
      NSUInteger threads =
          MIN((NSUInteger)256, pipelines->scale.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)nDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // ── Step 10+: Chain Next Layer QKV directly into the same command buffer
    // ──
    if (hasNext && nextWQ != NULL && bufNextAttn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // Step 10: xb = rmsNorm(x, nextAttnNormW)
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufNextAttn, nextAttnOff,
                          nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // Step 11, 12, 13: next Q, K, V matvecs
      uint32_t nextKK = nDim;
      encode_matvec_sub(enc, pipelines, g_woff_nextQ, g_woff_xb, nextWQ, nextKK,
                        (uint32_t)nextN0, nextTypeQ, 1);
      if (nextWK != NULL) {
        encode_matvec_sub(enc, pipelines, g_woff_nextK, g_woff_xb, nextWK,
                          nextKK, (uint32_t)nextN1, nextTypeK, 1);
      }
      if (nextWV != NULL) {
        encode_matvec_sub(enc, pipelines, g_woff_nextV, g_woff_xb, nextWV,
                          nextKK, (uint32_t)nextN2, nextTypeV, 1);
      }
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      g_woff_x_valid = NO;
      return -1;
    }

    // Return fully-updated residual x to caller only when finishing the layer
    // chain
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

int aljabr_metal_wo_ffn_fused(float *x, const float *attn_full, const void *wO,
                              int typeWO, int wO_K, int wO_N,
                              const float *postAttnW, int hasPostAttnNorm,
                              int addOnePostAttn, const float *ffnNormW,
                              int addOneFfnNorm, const float *postFfnW,
                              int hasPostFfnNorm, int addOnePostFfn, float eps,
                              const void *gateW, const void *upW,
                              const void *downW, int typeGate, int typeUp,
                              int typeDown, int ffnDim, int activationKind) {
  return aljabr_metal_wo_ffn_next_qkv_fused(
      x, attn_full, wO, typeWO, wO_K, wO_N, postAttnW, hasPostAttnNorm,
      ffnNormW, postFfnW, hasPostFfnNorm, eps, 1.0f, gateW, upW, downW,
      typeGate, typeUp, typeDown, ffnDim, activationKind, 0, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0);
}

void aljabr_metal_wo_ffn_reset_x(void) { g_woff_x_valid = NO; }

static id<MTLBuffer> g_woff_logits = nil;
static size_t g_woff_logits_cap = 0;
static id<MTLBuffer> g_woff_outputNormW = nil;
static size_t g_woff_outputNormW_cap = 0;

static BOOL ensure_woff_logits(size_t logitsBytes, size_t normBytes) {
  if (logitsBytes > g_woff_logits_cap) {
    size_t sz = (logitsBytes + 16383) & ~16383;
    g_woff_logits = [g_device newBufferWithLength:sz
                                          options:MTLResourceStorageModeShared];
    g_woff_logits_cap = sz;
    if (!g_woff_logits)
      return NO;
  }
  if (normBytes > g_woff_outputNormW_cap) {
    size_t sz = (normBytes + 16383) & ~16383;
    g_woff_outputNormW =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_woff_outputNormW_cap = sz;
    if (!g_woff_outputNormW)
      return NO;
  }
  return YES;
}

int aljabr_metal_wo_ffn_logits_fused(
    float *x, const float *attn_full, const void *wO, int typeWO, int wO_K,
    int wO_N, const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW, const float *postFfnW, int hasPostFfnNorm, float eps,
    float outScale, const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown, int ffnDim, int activationKind,
    const float *outputNormW, const void *outputW, int typeOutputW, int outputK,
    int outputN, float *logits, float cap) {
  if (!g_initialized)
    return -1;
  if (wO_K <= 0 || wO_N <= 0 || ffnDim <= 0 || outputK <= 0 || outputN <= 0)
    return -2;

  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->rmsnorm == nil && pipelines->rmsnorm_float4 == nil)
    return -3;
  if (pipelines->add == nil)
    return -3;

  // Resolve wO pipeline
  id<MTLComputePipelineState> pWO = nil;
  size_t wOBytes = 0;
  BOOL wO_mr8 = NO, wO_mr4 = NO;
  if (typeWO == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pWO = pipelines->matvec_tb_q4_k_mr8;
      wO_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pWO = pipelines->matvec_tb_q4_k_mr4;
      wO_mr4 = YES;
    } else
      pWO = pipelines->matvec_tb_q4_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 144;
  } else if (typeWO == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pWO = pipelines->matvec_tb_q6_k_mr8;
      wO_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pWO = pipelines->matvec_tb_q6_k_mr4;
      wO_mr4 = YES;
    } else
      pWO = pipelines->matvec_tb_q6_k;
    wOBytes = (size_t)wO_N * wO_K / 256 * 210;
  }
  if (pWO == nil)
    return -3;

  // Resolve FFN pipelines
  BOOL can_fuse_gate_up =
      ((typeGate == 12 && typeUp == 12 &&
        ((pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled()) ||
         pipelines->ffn_gate_up_act_q4_k_mr4 != nil)) ||
       (typeGate == 14 && typeUp == 14 &&
        pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
  BOOL gate_up_mr8 =
      (typeGate == 12 && typeUp == 12 &&
       pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled());
  id<MTLComputePipelineState> pGate = nil, pUp = nil, pDown = nil;
  size_t gateBytes = 0, upBytes = 0, downBytes = 0;
  BOOL gate_mr8 = NO, up_mr8 = NO, down_mr8 = NO;
  BOOL gate_mr4 = NO, up_mr4 = NO, down_mr4 = NO;

  if (typeGate == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pGate = pipelines->matvec_tb_q4_k_mr8;
      gate_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pGate = pipelines->matvec_tb_q4_k_mr4;
      gate_mr4 = YES;
    } else
      pGate = pipelines->matvec_tb_q4_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeGate == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pGate = pipelines->matvec_tb_q6_k_mr8;
      gate_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pGate = pipelines->matvec_tb_q6_k_mr4;
      gate_mr4 = YES;
    } else
      pGate = pipelines->matvec_tb_q6_k;
    gateBytes = (size_t)ffnDim * wO_N / 256 * 210;
  }

  if (typeUp == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pUp = pipelines->matvec_tb_q4_k_mr8;
      up_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pUp = pipelines->matvec_tb_q4_k_mr4;
      up_mr4 = YES;
    } else
      pUp = pipelines->matvec_tb_q4_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 144;
  } else if (typeUp == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pUp = pipelines->matvec_tb_q6_k_mr8;
      up_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pUp = pipelines->matvec_tb_q6_k_mr4;
      up_mr4 = YES;
    } else
      pUp = pipelines->matvec_tb_q6_k;
    upBytes = (size_t)ffnDim * wO_N / 256 * 210;
  }

  if (typeDown == 12) {
    if (pipelines->matvec_tb_q4_k_mr8 != nil && mr8_enabled()) {
      pDown = pipelines->matvec_tb_q4_k_mr8;
      down_mr8 = YES;
    } else if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      pDown = pipelines->matvec_tb_q4_k_mr4;
      down_mr4 = YES;
    } else
      pDown = pipelines->matvec_tb_q4_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 144;
  } else if (typeDown == 14) {
    if (pipelines->matvec_tb_q6_k_mr8 != nil && mr8_enabled()) {
      pDown = pipelines->matvec_tb_q6_k_mr8;
      down_mr8 = YES;
    } else if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      pDown = pipelines->matvec_tb_q6_k_mr4;
      down_mr4 = YES;
    } else
      pDown = pipelines->matvec_tb_q6_k;
    downBytes = (size_t)wO_N * ffnDim / 256 * 210;
  }

  if (!can_fuse_gate_up && (pGate == nil || pUp == nil))
    return -3;
  if (pDown == nil)
    return -3;

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
  if (pLogits == nil || outWBytes == 0)
    return -3;

  size_t attnKBytes = (size_t)wO_K * sizeof(float);
  size_t dimBytes = (size_t)wO_N * sizeof(float);
  size_t ffnBytes = (size_t)ffnDim * sizeof(float);
  size_t logitsBytes = (size_t)outputN * sizeof(float);

  if (!ensure_woff_dim_bufs(attnKBytes, dimBytes))
    return -4;
  if (!ensure_woff_ffn_bufs(ffnBytes))
    return -4;
  if (!ensure_woff_logits(logitsBytes, dimBytes))
    return -4;

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
    id<MTLBuffer> bufWO = wrap_weight_ptr_offset(wO, wOBytes, &wOOff);
    id<MTLBuffer> bufGate = wrap_weight_ptr_offset(gateW, gateBytes, &gateOff);
    id<MTLBuffer> bufUp = wrap_weight_ptr_offset(upW, upBytes, &upOff);
    id<MTLBuffer> bufDown = wrap_weight_ptr_offset(downW, downBytes, &downOff);
    id<MTLBuffer> bufOutW =
        wrap_weight_ptr_offset(outputW, outWBytes, &outWOff);
    if (!bufWO || !bufGate || !bufUp || !bufDown || !bufOutW)
      return -4;

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
    if (wO_mr8) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (wO_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 2: xb = postAttnNorm(xb)
    if (hasPostAttnNorm && bufPostAttn) {
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostAttn, postAttnOff,
                          nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Step 3: x += xb
    encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 4: xb = ffnNorm(x)
    encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufFfnNorm, ffnNormOff, nDim,
                        eps, 0, pipelines);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Step 5+6: gate+up+act
    if (can_fuse_gate_up) {
      id<MTLComputePipelineState> fuseP;
      if (gate_up_mr8) {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr8;
      } else if (typeGate == 14) {
        fuseP = pipelines->ffn_gate_up_act_q6_k_mr4;
      } else {
        fuseP = pipelines->ffn_gate_up_act_q4_k_mr4;
      }
      [enc setComputePipelineState:fuseP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBuffer:bufUp offset:upOff atIndex:3];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
      [enc setBytes:&actK length:sizeof(actK) atIndex:6];
      if (gate_up_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
    } else {
      [enc setComputePipelineState:pGate];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:0];
      [enc setBuffer:g_woff_xb offset:0 atIndex:1];
      [enc setBuffer:bufGate offset:gateOff atIndex:2];
      [enc setBytes:&nDim length:sizeof(nDim) atIndex:3];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:4];
      if (gate_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (gate_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
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
      if (up_mr8) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 7) / 8, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else if (up_mr4) {
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)ffnDim + 3) / 4, 1,
                                              1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ffnDim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      id<MTLComputePipelineState> actP =
          (activationKind == 2) ? pipelines->gelu_ffn : pipelines->silu_ffn;
      if (actP == nil) {
        [enc endEncoding];
        return -3;
      }
      [enc setComputePipelineState:actP];
      [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
      [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:1];
      [enc setBuffer:g_woff_ffn_up offset:0 atIndex:2];
      [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
      NSUInteger actTh =
          MIN((NSUInteger)256, actP.maxTotalThreadsPerThreadgroup);
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
    if (down_mr8) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 7) / 8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (down_mr4) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)wO_N + 3) / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)wO_N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Step 8: postFfnNorm
    if (hasPostFfnNorm && bufPostFfn) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostFfn, postFfnOff,
                          nDim, eps, 0, pipelines);
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
      NSUInteger threads =
          MIN((NSUInteger)256, pipelines->scale.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreads:MTLSizeMake((NSUInteger)nDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // ── Step 10: Output RMSNorm directly into g_woff_xb ──
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufOutNorm, outNormOff, nDim,
                        eps, 0, pipelines);
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

// ── Unified 48-Layer GPU Decode Pipeline ──────────────────────────────────
#define MAX_DECODE_LAYERS 64

static id<MTLBuffer> g_kv_k[MAX_DECODE_LAYERS];
static id<MTLBuffer> g_kv_v[MAX_DECODE_LAYERS];
static uint32_t g_kv_stride[MAX_DECODE_LAYERS];
static int g_kv_layers = 0;
static int g_kv_max_seq = 0;

static id<MTLBuffer> g_norm_buffers[6]; // 0=postAttn, 1=ffnNorm, 2=postFfn,
                                        // 3=attnNorm, 4=attnQNorm, 5=attnKNorm
static size_t g_norm_buffer_bytes[6];

typedef struct {
  const void *wQ;
  const void *wK;
  const void *wV;
  const void *wO;
  const void *gate;
  const void *up;
  const void *down;
  int typeQ;
  int typeK;
  int typeV;
  int typeWO;
  int typeGate;
  int typeUp;
  int typeDown;
  int n0, n1, n2, wO_K, wO_N, ffnDim;
  int nH, nKVH, headDim, stride;
  float freqBase;
  int isSwa;
  float outScale;
  float attnScale;
  int hasPostAttnNorm;
  int hasPostFfnNorm;
  int hasQNorm;
  int hasKNorm;
  int hasUnitVNorm;
  id<MTLBuffer> bufWQ;
  size_t offWQ;
  id<MTLBuffer> bufWK;
  size_t offWK;
  id<MTLBuffer> bufWV;
  size_t offWV;
  id<MTLBuffer> bufWO;
  size_t offWO;
  id<MTLBuffer> bufGate;
  size_t offGate;
  id<MTLBuffer> bufUp;
  size_t offUp;
  id<MTLBuffer> bufDown;
  size_t offDown;
  // Optional rope_freqs.weight divisors (non-null only for layers that use
  // proportional RoPE)
  id<MTLBuffer> bufRopeFreqs;
  uint32_t ropeFreqsLen;
  // Index of the KV cache layer to attend to. Equals 'layer' when this layer
  // has its own wK/wV weights; equals a previous layer index when wK==NULL
  // (Gemma 4 KV-sharing).
  int kvLayerIndex;
  int resolvedKvLayer;
} DecodeLayerConfig;

static DecodeLayerConfig g_decode_layers[MAX_DECODE_LAYERS];
static int g_num_decode_layers = 0;

static size_t g_decode_dimBytes = 0;
static size_t g_decode_maxAttnBytes = 0;
static size_t g_decode_maxFfnBytes = 0;
static size_t g_decode_maxQ = 0;
static size_t g_decode_maxK = 0;
static size_t g_decode_maxV = 0;
static uint32_t g_decode_maxHeadDim = 0;
static BOOL g_decode_geometry_ready = NO;

static void prepare_decode_geometry(void) {
  if (g_num_decode_layers == 0)
    return;
  DecodeLayerConfig *cfg0 = &g_decode_layers[0];
  g_decode_dimBytes = (size_t)cfg0->wO_N * sizeof(float);
  g_decode_maxAttnBytes = (size_t)cfg0->wO_K * sizeof(float);
  g_decode_maxFfnBytes = (size_t)cfg0->ffnDim * sizeof(float);
  g_decode_maxQ = (size_t)cfg0->n0 * sizeof(float);
  g_decode_maxK = (size_t)cfg0->n1 * sizeof(float);
  g_decode_maxV = (size_t)cfg0->n2 * sizeof(float);
  g_decode_maxHeadDim = (uint32_t)cfg0->headDim;

  for (int l = 0; l < g_num_decode_layers; l++) {
    DecodeLayerConfig *cfg = &g_decode_layers[l];
    if ((size_t)cfg->wO_K * sizeof(float) > g_decode_maxAttnBytes)
      g_decode_maxAttnBytes = (size_t)cfg->wO_K * sizeof(float);
    if ((size_t)cfg->ffnDim * sizeof(float) > g_decode_maxFfnBytes)
      g_decode_maxFfnBytes = (size_t)cfg->ffnDim * sizeof(float);
    if ((uint32_t)cfg->headDim > g_decode_maxHeadDim)
      g_decode_maxHeadDim = (uint32_t)cfg->headDim;
    if ((size_t)cfg->n0 * sizeof(float) > g_decode_maxQ)
      g_decode_maxQ = (size_t)cfg->n0 * sizeof(float);
    if ((size_t)cfg->n1 * sizeof(float) > g_decode_maxK)
      g_decode_maxK = (size_t)cfg->n1 * sizeof(float);
    if ((size_t)cfg->n2 * sizeof(float) > g_decode_maxV)
      g_decode_maxV = (size_t)cfg->n2 * sizeof(float);

    int kvLayer = cfg->kvLayerIndex;
    int hops = 0;
    while (kvLayer >= 0 && kvLayer < g_num_decode_layers &&
           hops++ < g_num_decode_layers) {
      DecodeLayerConfig *candidate = &g_decode_layers[kvLayer];
      if (candidate->wK)
        break;
      if (candidate->kvLayerIndex == kvLayer)
        break;
      kvLayer = candidate->kvLayerIndex;
    }
    cfg->resolvedKvLayer = kvLayer;
  }
  ensure_woff_dim_bufs(g_decode_maxAttnBytes, g_decode_dimBytes);
  ensure_woff_ffn_bufs(g_decode_maxFfnBytes);
  ensure_woff_next_qkv(g_decode_maxQ, g_decode_maxK, g_decode_maxV,
                       g_decode_dimBytes);
  g_decode_geometry_ready = YES;
}

static const float *g_output_norm_w = NULL;
static const void *g_output_w = NULL;
static int g_output_type_w = 0;
static int g_output_K = 0;
static int g_output_N = 0;
static float g_output_logit_cap = 0.0f;
static id<MTLBuffer> g_buf_output_norm = nil;
static size_t g_off_output_norm = 0;
static id<MTLBuffer> g_buf_output_w = nil;
static size_t g_off_output_w = 0;
static id<MTLBuffer> g_buf_output_logits = nil;
static size_t g_cap_output_logits = 0;

int aljabr_metal_init_kv_cache(int nLayers, int maxSeq,
                               const int *layerStrides) {
  if (!g_initialized)
    return -1;
  if (g_device == nil || nLayers <= 0 || nLayers > MAX_DECODE_LAYERS ||
      maxSeq <= 0 || layerStrides == NULL)
    return -1;
  g_kv_layers = nLayers;
  g_kv_max_seq = maxSeq;
  for (int i = 0; i < nLayers; i++) {
    uint32_t stride = (uint32_t)layerStrides[i];
    g_kv_stride[i] = stride;
    size_t bytes = (size_t)maxSeq * stride * sizeof(float);
    size_t allocBytes = (bytes + 16383) & ~16383;
    g_kv_k[i] = [g_device newBufferWithLength:allocBytes
                                      options:MTLResourceStorageModeShared];
    g_kv_v[i] = [g_device newBufferWithLength:allocBytes
                                      options:MTLResourceStorageModeShared];
    if (g_kv_k[i] == nil || g_kv_v[i] == nil)
      return -2;
  }
  return 0;
}

int aljabr_metal_sync_kv_cache(int layer, int seqLen, const float *kData,
                               const float *vData) {
  if (layer < 0 || layer >= g_kv_layers || seqLen <= 0 || seqLen > g_kv_max_seq)
    return -1;
  size_t bytes = (size_t)seqLen * g_kv_stride[layer] * sizeof(float);
  if (kData && g_kv_k[layer])
    memcpy([g_kv_k[layer] contents], kData, bytes);
  if (vData && g_kv_v[layer])
    memcpy([g_kv_v[layer] contents], vData, bytes);
  return 0;
}

int aljabr_metal_register_norm_buffer(int slot, const void *ptr, size_t bytes) {
  if (slot < 0 || slot >= 6 || ptr == NULL || bytes == 0)
    return -1;
  if (!g_initialized)
    return -1;
  g_norm_buffers[slot] = wrap_ptr((void *)ptr, bytes);
  g_norm_buffer_bytes[slot] = bytes;
  return (g_norm_buffers[slot] != nil) ? 0 : -2;
}

int aljabr_metal_register_rope_freqs(int layer, const float *freqData,
                                     int freqLen) {
  if (layer < 0 || layer >= MAX_DECODE_LAYERS || freqData == NULL ||
      freqLen <= 0)
    return -1;
  if (!g_initialized)
    return -1;
  size_t bytes = (size_t)freqLen * sizeof(float);
  id<MTLBuffer> buf =
      [g_device newBufferWithBytes:freqData
                            length:bytes
                           options:MTLResourceStorageModeShared];
  if (buf == nil)
    return -2;
  g_decode_layers[layer].bufRopeFreqs = buf;
  g_decode_layers[layer].ropeFreqsLen = (uint32_t)freqLen;
  return 0;
}

int aljabr_metal_register_decode_layer(
    int layer, int kvLayerIndex, const void *wQ, const void *wK, const void *wV,
    const void *wO, const void *gate, const void *up, const void *down,
    int typeQ, int typeK, int typeV, int typeWO, int typeGate, int typeUp,
    int typeDown, int n0, int n1, int n2, int wO_K, int wO_N, int ffnDim,
    int nH, int nKVH, int headDim, int stride, float freqBase, int isSwa,
    float outScale, float attnScale, int hasPostAttnNorm,
    int hasPostFfnNorm, int hasQNorm, int hasKNorm, int hasUnitVNorm) {
  if (layer < 0 || layer >= MAX_DECODE_LAYERS)
    return -1;
  if (kvLayerIndex < 0 || kvLayerIndex >= MAX_DECODE_LAYERS ||
      kvLayerIndex > layer)
    return -1;
  if (layer >= g_num_decode_layers)
    g_num_decode_layers = layer + 1;
  g_decode_geometry_ready = NO;
  DecodeLayerConfig *cfg = &g_decode_layers[layer];
  cfg->kvLayerIndex = kvLayerIndex;
  cfg->wQ = wQ;
  cfg->wK = wK;
  cfg->wV = wV;
  cfg->wO = wO;
  cfg->gate = gate;
  cfg->up = up;
  cfg->down = down;
  cfg->typeQ = typeQ;
  cfg->typeK = typeK;
  cfg->typeV = typeV;
  cfg->typeWO = typeWO;
  cfg->typeGate = typeGate;
  cfg->typeUp = typeUp;
  cfg->typeDown = typeDown;
  cfg->n0 = n0;
  cfg->n1 = n1;
  cfg->n2 = n2;
  cfg->wO_K = wO_K;
  cfg->wO_N = wO_N;
  cfg->ffnDim = ffnDim;
  cfg->nH = nH;
  cfg->nKVH = nKVH;
  cfg->headDim = headDim;
  cfg->stride = stride;
  cfg->freqBase = freqBase;
  cfg->isSwa = isSwa;
  cfg->kvLayerIndex = (kvLayerIndex >= 0 && kvLayerIndex < MAX_DECODE_LAYERS)
                          ? kvLayerIndex
                          : layer;
  cfg->outScale = outScale;
  cfg->attnScale = attnScale;
  cfg->hasPostAttnNorm = hasPostAttnNorm;
  cfg->hasPostFfnNorm = hasPostFfnNorm;
  cfg->hasQNorm = hasQNorm;
  cfg->hasKNorm = hasKNorm;
  cfg->hasUnitVNorm = hasUnitVNorm;

  if (wQ)
    cfg->bufWQ = wrap_weight_ptr_offset(
        wQ, calc_matvec_weight_bytes(typeQ, n0, wO_N), &cfg->offWQ);
  if (wK)
    cfg->bufWK = wrap_weight_ptr_offset(
        wK, calc_matvec_weight_bytes(typeK, n1, wO_N), &cfg->offWK);
  if (wV)
    cfg->bufWV = wrap_weight_ptr_offset(
        wV, calc_matvec_weight_bytes(typeV, n2, wO_N), &cfg->offWV);
  if (wO)
    cfg->bufWO = wrap_weight_ptr_offset(
        wO, calc_matvec_weight_bytes(typeWO, wO_N, wO_K), &cfg->offWO);
  if (gate)
    cfg->bufGate = wrap_weight_ptr_offset(
        gate, calc_matvec_weight_bytes(typeGate, ffnDim, wO_N), &cfg->offGate);
  if (up)
    cfg->bufUp = wrap_weight_ptr_offset(
        up, calc_matvec_weight_bytes(typeUp, ffnDim, wO_N), &cfg->offUp);
  if (down)
    cfg->bufDown = wrap_weight_ptr_offset(
        down, calc_matvec_weight_bytes(typeDown, wO_N, ffnDim), &cfg->offDown);
  return 0;
}

int aljabr_metal_register_output(const float *outputNormW, const void *outputW,
                                 int typeOutputW, int outputK, int outputN,
                                 float cap) {
  g_output_norm_w = outputNormW;
  g_output_w = outputW;
  g_output_type_w = typeOutputW;
  g_output_K = outputK;
  g_output_N = outputN;
  g_output_logit_cap = cap;
  if (outputNormW) {
    g_buf_output_norm =
        wrap_ptr((void *)outputNormW, (size_t)outputK * sizeof(float));
    g_off_output_norm = 0;
  }
  if (outputW) {
    g_buf_output_w = wrap_weight_ptr_offset(
        outputW, calc_matvec_weight_bytes(typeOutputW, outputN, outputK),
        &g_off_output_w);
  }
  size_t logitsBytes = (size_t)outputN * sizeof(float);
  if (logitsBytes > g_cap_output_logits || g_buf_output_logits == nil) {
    size_t sz = (logitsBytes + 16383) & ~16383;
    g_buf_output_logits =
        [g_device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    g_cap_output_logits = sz;
  }
  return 0;
}

static inline void encode_rope_single_token(id<MTLComputeCommandEncoder> enc,
                                            id<MTLBuffer> bufX, uint32_t nHeads,
                                            uint32_t headDim, uint32_t pos,
                                            float freqBase,
                                            AljabrMetalPipelines *pipelines) {
  if (pipelines->rope_single_token == nil || bufX == nil)
    return;
  uint32_t half_dim = headDim / 2;
  uint32_t total = nHeads * half_dim;
  [enc setComputePipelineState:pipelines->rope_single_token];
  [enc setBuffer:bufX offset:0 atIndex:0];
  [enc setBytes:&nHeads length:sizeof(nHeads) atIndex:1];
  [enc setBytes:&headDim length:sizeof(headDim) atIndex:2];
  [enc setBytes:&pos length:sizeof(pos) atIndex:3];
  [enc setBytes:&freqBase length:sizeof(freqBase) atIndex:4];
  NSUInteger threads = MIN((NSUInteger)256, (NSUInteger)total);
  NSUInteger groups = (total + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static inline void encode_rope_single_token_freqfact(
    id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bufX, uint32_t nHeads,
    uint32_t headDim, uint32_t pos, float freqBase,
    id<MTLBuffer> bufFreqFactors, AljabrMetalPipelines *pipelines) {
  if (bufFreqFactors == nil || pipelines->rope_single_token_freqfact == nil) {
    encode_rope_single_token(enc, bufX, nHeads, headDim, pos, freqBase,
                             pipelines);
    return;
  }
  if (bufX == nil)
    return;
  uint32_t half_dim = headDim / 2;
  uint32_t total = nHeads * half_dim;
  [enc setComputePipelineState:pipelines->rope_single_token_freqfact];
  [enc setBuffer:bufX offset:0 atIndex:0];
  [enc setBytes:&nHeads length:sizeof(nHeads) atIndex:1];
  [enc setBytes:&headDim length:sizeof(headDim) atIndex:2];
  [enc setBytes:&pos length:sizeof(pos) atIndex:3];
  [enc setBytes:&freqBase length:sizeof(freqBase) atIndex:4];
  [enc setBuffer:bufFreqFactors offset:0 atIndex:5];
  NSUInteger threads = MIN((NSUInteger)256, (NSUInteger)total);
  NSUInteger groups = (total + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static inline void encode_kv_copy_flat(id<MTLComputeCommandEncoder> enc,
                                       id<MTLBuffer> bufDst,
                                       id<MTLBuffer> bufSrc, uint32_t count,
                                       AljabrMetalPipelines *pipelines) {
  if (pipelines->kv_copy_flat == nil || bufDst == nil || bufSrc == nil ||
      count == 0)
    return;
  [enc setComputePipelineState:pipelines->kv_copy_flat];
  [enc setBuffer:bufDst offset:0 atIndex:0];
  [enc setBuffer:bufSrc offset:0 atIndex:1];
  [enc setBytes:&count length:sizeof(count) atIndex:2];
  NSUInteger threads = MIN((NSUInteger)256, (NSUInteger)count);
  if (threads == 0)
    threads = 1;
  NSUInteger groups = (count + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static inline void encode_kv_store_flat(id<MTLComputeCommandEncoder> enc,
                                        id<MTLBuffer> bufCache,
                                        id<MTLBuffer> bufVec, uint32_t pos,
                                        uint32_t stride,
                                        AljabrMetalPipelines *pipelines) {
  if (pipelines->kv_store_flat == nil || bufCache == nil || bufVec == nil)
    return;
  [enc setComputePipelineState:pipelines->kv_store_flat];
  [enc setBuffer:bufCache offset:0 atIndex:0];
  [enc setBuffer:bufVec offset:0 atIndex:1];
  [enc setBytes:&pos length:sizeof(pos) atIndex:2];
  [enc setBytes:&stride length:sizeof(stride) atIndex:3];
  NSUInteger threads = MIN((NSUInteger)256, (NSUInteger)stride);
  NSUInteger groups = (stride + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static inline void
encode_sdpa_flat(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bufOut,
                 id<MTLBuffer> bufQ, id<MTLBuffer> bufKCache,
                 id<MTLBuffer> bufVCache, uint32_t H, uint32_t H_kv, uint32_t D,
                 uint32_t stride, uint32_t pos, uint32_t start_pos, float scale,
                 float soft_cap, AljabrMetalPipelines *pipelines) {
  if (pipelines->sdpa_flat == nil || bufOut == nil || bufQ == nil ||
      bufKCache == nil || bufVCache == nil)
    return;
  [enc setComputePipelineState:pipelines->sdpa_flat];
  [enc setBuffer:bufOut offset:0 atIndex:0];
  [enc setBuffer:bufQ offset:0 atIndex:1];
  [enc setBuffer:bufKCache offset:0 atIndex:2];
  [enc setBuffer:bufVCache offset:0 atIndex:3];
  [enc setBytes:&H length:sizeof(H) atIndex:4];
  [enc setBytes:&H_kv length:sizeof(H_kv) atIndex:5];
  [enc setBytes:&D length:sizeof(D) atIndex:6];
  [enc setBytes:&stride length:sizeof(stride) atIndex:7];
  [enc setBytes:&pos length:sizeof(pos) atIndex:8];
  [enc setBytes:&start_pos length:sizeof(start_pos) atIndex:9];
  [enc setBytes:&scale length:sizeof(scale) atIndex:10];
  [enc setBytes:&soft_cap length:sizeof(soft_cap) atIndex:11];
  [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static id<MTLBuffer> g_token_embd_buffer = nil;
static size_t g_token_embd_offset = 0;
static int g_token_embd_dim = 0;
static int g_token_embd_vocab = 0;
static float g_token_embd_scale = 1.0f;
static id<MTLBuffer> g_token_embd_output = nil;
static size_t g_token_embd_output_capacity = 0;

static int aljabr_metal_decode_step_internal(int pos, int slidingWindow, float eps,
                             const float *x_in, float *logits_out, int tokenId, BOOL useTokenEmbedding) {
  if (!g_initialized || g_device == nil || g_queue == nil ||
      g_num_decode_layers == 0) {
    fprintf(stderr,
            "[aljabr-metal] decode_step rejected pos=%d initialized=%d device=%d queue=%d registered_layers=%d\n",
            pos, (int)g_initialized, g_device != nil, g_queue != nil,
            g_num_decode_layers);
    return -1;
  }
  BOOL profileDecode = env_truthy("ALJABR_METAL_DECODE_PROFILE");
  uint64_t profileStartNanos =
      profileDecode ? monotonic_nanos() : 0;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();

  if (!g_decode_geometry_ready) {
    prepare_decode_geometry();
  }
  size_t dimBytes = g_decode_dimBytes;
  uint32_t maxHeadDim = g_decode_maxHeadDim;

  @autoreleasepool {
    if (useTokenEmbedding) {
      if (g_token_embd_buffer == nil || g_token_embd_dim != (int)(dimBytes / sizeof(float)) ||
          tokenId < 0 || tokenId >= g_token_embd_vocab)
        return -6;
    } else {
      if (x_in == NULL)
        return -5;
      memcpy([g_woff_x contents], x_in, dimBytes);
    }

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (cmd == nil || enc == nil)
      return -5;

    // Encode the embedding first into the same vector consumed by layer zero.
    // Subsequent commands in this encoder observe it without an intermediate
    // commit, wait, or CPU-side copy.
    if (useTokenEmbedding) {
      AljabrMetalPipelines *embedPipelines = aljabr_metal_pipelines();
      if (embedPipelines == NULL || embedPipelines->embed_token_q4_k == nil) {
        [enc endEncoding];
        return -3;
      }
      uint32_t token = (uint32_t)tokenId;
      uint32_t K = (uint32_t)g_token_embd_dim;
      float scale = g_token_embd_scale;
      [enc setComputePipelineState:embedPipelines->embed_token_q4_k];
      [enc setBuffer:g_woff_x offset:0 atIndex:0];
      [enc setBuffer:g_token_embd_buffer offset:g_token_embd_offset atIndex:1];
      [enc setBytes:&token length:sizeof(token) atIndex:2];
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&scale length:sizeof(scale) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(K / 256u), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      // Make the embedding writes visible to layer-zero kernels in this encoder.
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    id<MTLBuffer> bufPostAttnAll = g_norm_buffers[0]; // slot 0 = postAttnNorm
    id<MTLBuffer> bufFfnNormAll = g_norm_buffers[1];  // slot 1 = ffnNorm
    id<MTLBuffer> bufPostFfnAll = g_norm_buffers[2];  // slot 2 = postFfnNorm
    id<MTLBuffer> bufAttnNormAll = g_norm_buffers[3]; // slot 3 = attnNorm
    id<MTLBuffer> bufQNormAll = g_norm_buffers[4];    // slot 4 = attnQNorm
    id<MTLBuffer> bufKNormAll = g_norm_buffers[5];    // slot 5 = attnKNorm
    if (!bufAttnNormAll || !bufFfnNormAll) {
      fprintf(stderr,
              "[aljabr-metal] decode_step missing required norm buffer pos=%d attn_norm=%d ffn_norm=%d\n",
              pos, bufAttnNormAll != nil, bufFfnNormAll != nil);
      [enc endEncoding];
      return -3;
    }

    BOOL debugLayers = (getenv("ALJABR_DEBUG_LAYERS") != NULL && pos == 27);
    for (int l = 0; l < g_num_decode_layers; l++) {
      if (debugLayers && l > 0) {
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        float *x_ptr = (float *)[g_woff_x contents];
        float sumSq = 0.0f;
        for (int k = 0; k < (int)(dimBytes / sizeof(float)); k++)
          sumSq += x_ptr[k] * x_ptr[k];
        printf("[METAL L%02d] x[0]=%+.4f, norm=%.4f\n", l - 1, x_ptr[0],
               sqrtf(sumSq));
        cmd = [g_queue commandBuffer];
        enc = [cmd computeCommandEncoder];
      }
      DecodeLayerConfig *cfg = &g_decode_layers[l];
      uint32_t nDim = (uint32_t)cfg->wO_N;
      uint32_t nFfn = (uint32_t)cfg->ffnDim;
      uint32_t kDim = (uint32_t)cfg->wO_K;
      size_t normOff = (size_t)l * dimBytes;

      // 1. xb = attnNorm(x)
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufAttnNormAll, normOff,
                          nDim, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 2. Q, K, V matvecs from xb
      encode_matvec_buf(enc, pipelines, g_woff_nextQ, g_woff_xb, cfg->bufWQ,
                        cfg->offWQ, nDim, (uint32_t)cfg->n0, cfg->typeQ, 1);
      if (cfg->wK) {
        encode_matvec_buf(enc, pipelines, g_woff_nextK, g_woff_xb, cfg->bufWK,
                          cfg->offWK, nDim, (uint32_t)cfg->n1, cfg->typeK, 1);
      }
      if (cfg->wV) {
        encode_matvec_buf(enc, pipelines, g_woff_nextV, g_woff_xb, cfg->bufWV,
                          cfg->offWV, nDim, (uint32_t)cfg->n2, cfg->typeV, 1);
      }

      // K-as-V uses the raw K projection for V. Copy before K normalization
      // and RoPE, both of which mutate g_woff_nextK. Using a compute copy kernel
      // avoids switching to a blit encoder and stays in the same compute command pass.
      if (cfg->wK && !cfg->wV) {
        size_t kvBytes = (size_t)cfg->n1 * sizeof(float);
        if (cfg->n1 <= 0 || cfg->n1 != cfg->n2 ||
            [g_woff_nextK length] < kvBytes ||
            [g_woff_nextV length] < kvBytes) {
          fprintf(stderr,
                  "[aljabr-metal] decode_step invalid K-as-V dimensions pos=%d layer=%d nK=%d nV=%d K_bytes=%lu V_bytes=%lu\n",
                  pos, l, cfg->n1, cfg->n2,
                  (unsigned long)[g_woff_nextK length],
                  (unsigned long)[g_woff_nextV length]);
          [enc endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          return -6;
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_kv_copy_flat(enc, g_woff_nextV, g_woff_nextK, (uint32_t)cfg->n1, pipelines);
      }
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 2b. Q & K Head Norms (if present, e.g. Gemma 4)
      if (cfg->hasQNorm && bufQNormAll && pipelines->rmsnorm_rows != nil) {
        uint32_t headDim = (uint32_t)cfg->headDim;
        uint32_t nH = (uint32_t)cfg->nH;
        uint32_t addOne = 0;
        size_t qNormOff = (size_t)l * (size_t)maxHeadDim * sizeof(float);
        [enc setComputePipelineState:pipelines->rmsnorm_rows];
        [enc setBuffer:g_woff_nextQ offset:0 atIndex:0];
        [enc setBuffer:g_woff_nextQ offset:0 atIndex:1];
        [enc setBuffer:bufQNormAll offset:qNormOff atIndex:2];
        [enc setBytes:&headDim length:sizeof(headDim) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc setBytes:&addOne length:sizeof(addOne) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(nH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (cfg->hasKNorm && bufKNormAll && cfg->wK &&
          pipelines->rmsnorm_rows != nil) {
        uint32_t headDim = (uint32_t)cfg->headDim;
        uint32_t nKVH = (uint32_t)cfg->nKVH;
        uint32_t addOne = 0;
        size_t kNormOff = (size_t)l * (size_t)maxHeadDim * sizeof(float);
        [enc setComputePipelineState:pipelines->rmsnorm_rows];
        [enc setBuffer:g_woff_nextK offset:0 atIndex:0];
        [enc setBuffer:g_woff_nextK offset:0 atIndex:1];
        [enc setBuffer:bufKNormAll offset:kNormOff atIndex:2];
        [enc setBytes:&headDim length:sizeof(headDim) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc setBytes:&addOne length:sizeof(addOne) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(nKVH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      // 2c. Unit V Norm (if present, e.g. Gemma 4)
      if (cfg->hasUnitVNorm && (cfg->wV || cfg->wK) &&
          pipelines->rmsnorm_unit_rows != nil) {
        uint32_t headDim = (uint32_t)cfg->headDim;
        uint32_t nKVH = (uint32_t)cfg->nKVH;
        [enc setComputePipelineState:pipelines->rmsnorm_unit_rows];
        [enc setBuffer:g_woff_nextV offset:0 atIndex:0];
        [enc setBuffer:g_woff_nextV offset:0 atIndex:1];
        [enc setBytes:&headDim length:sizeof(headDim) atIndex:2];
        [enc setBytes:&eps length:sizeof(eps) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nKVH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (cfg->hasQNorm || cfg->hasKNorm || cfg->hasUnitVNorm) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      // 3. RoPE on Q and K
      if (cfg->bufRopeFreqs != nil) {
        encode_rope_single_token_freqfact(
            enc, g_woff_nextQ, (uint32_t)cfg->nH, (uint32_t)cfg->headDim,
            (uint32_t)pos, cfg->freqBase, cfg->bufRopeFreqs, pipelines);
        if (cfg->wK) {
          encode_rope_single_token_freqfact(
              enc, g_woff_nextK, (uint32_t)cfg->nKVH, (uint32_t)cfg->headDim,
              (uint32_t)pos, cfg->freqBase, cfg->bufRopeFreqs, pipelines);
        }
      } else {
        encode_rope_single_token(enc, g_woff_nextQ, (uint32_t)cfg->nH,
                                 (uint32_t)cfg->headDim, (uint32_t)pos,
                                 cfg->freqBase, pipelines);
        if (cfg->wK) {
          encode_rope_single_token(enc, g_woff_nextK, (uint32_t)cfg->nKVH,
                                   (uint32_t)cfg->headDim, (uint32_t)pos,
                                   cfg->freqBase, pipelines);
        }
      }
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 4. Store K and V to KV cache at position pos.
      // Any layer with a K projection owns its cache. For K-as-V, nextV was
      // copied from the raw K projection above before K was normalized/RoPE'd.
      // Shared-KV layers must not store stale nextK/V scratch values.
      if (cfg->wK && g_kv_k[l] && g_kv_v[l]) {
        encode_kv_store_flat(enc, g_kv_k[l], g_woff_nextK, (uint32_t)pos,
                             (uint32_t)cfg->stride, pipelines);
        encode_kv_store_flat(enc, g_kv_v[l], g_woff_nextV, (uint32_t)pos,
                             (uint32_t)cfg->stride, pipelines);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      // 5. Use the pre-resolved KV cache owner (avoids pointer chasing in decode loop).
      int kvLayer = cfg->resolvedKvLayer;
      if (kvLayer < 0 || kvLayer >= g_kv_layers ||
          !g_kv_k[kvLayer] || !g_kv_v[kvLayer]) {
        fprintf(stderr,
                "[aljabr-metal] decode_step invalid KV owner pos=%d layer=%d configured_owner=%d resolved_owner=%d decode_layers=%d kv_layers=%d K_buffer=%d V_buffer=%d\n",
                pos, l, cfg->kvLayerIndex, kvLayer, g_num_decode_layers,
                g_kv_layers,
                (kvLayer >= 0 && kvLayer < g_kv_layers && g_kv_k[kvLayer] != nil),
                (kvLayer >= 0 && kvLayer < g_kv_layers && g_kv_v[kvLayer] != nil));
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return -4;
      }
      DecodeLayerConfig *kvCfg = &g_decode_layers[kvLayer];
      if (!kvCfg->wK) {
        fprintf(stderr,
                "[aljabr-metal] decode_step KV owner has no K projection pos=%d layer=%d owner=%d hasK=%d hasV=%d\n",
                pos, l, kvLayer, kvCfg->wK != NULL, kvCfg->wV != NULL);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return -4;
      }
      if (kvCfg->headDim != cfg->headDim ||
          kvCfg->nKVH != cfg->nKVH ||
          g_kv_stride[kvLayer] <
              (uint32_t)(cfg->nKVH * cfg->headDim)) {
        fprintf(stderr,
                "[aljabr-metal] decode_step KV geometry mismatch pos=%d layer=%d owner=%d layer_head_dim=%d owner_head_dim=%d layer_kv_heads=%d owner_kv_heads=%d owner_stride=%u required_stride=%u\n",
                pos, l, kvLayer, cfg->headDim, kvCfg->headDim, cfg->nKVH,
                kvCfg->nKVH, g_kv_stride[kvLayer],
                (uint32_t)(cfg->nKVH * cfg->headDim));
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return -5;
      }

      // 6. SDPA attention -> g_woff_attnFull
      uint32_t seqLen = (uint32_t)(pos + 1);
      uint32_t startPos = 0;
      if (cfg->isSwa && slidingWindow > 0 && seqLen > (uint32_t)slidingWindow) {
        startPos = seqLen - (uint32_t)slidingWindow;
      }
      float scale = cfg->attnScale;
      if (scale <= 0.0f) {
        scale = (cfg->hasQNorm || cfg->hasKNorm)
                    ? 1.0f
                    : (1.0f / sqrtf((float)cfg->headDim));
      }
      // Attend to the resolved KV owner, not the current layer's cache.
      // Layers without K/V projections do not append the current position to
      // their own cache, so reading g_kv_k[l]/g_kv_v[l] omits the new token.
      encode_sdpa_flat(enc, g_woff_attnFull, g_woff_nextQ,
                       g_kv_k[kvLayer], g_kv_v[kvLayer],
                       (uint32_t)cfg->nH, (uint32_t)kvCfg->nKVH,
                       (uint32_t)cfg->headDim, g_kv_stride[kvLayer],
                       (uint32_t)pos, startPos, scale, 0.0f, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      if (debugLayers && l == 0) {
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        float *a_ptr = (float *)[g_woff_attnFull contents];
        float aNorm = 0;
        for (int i = 0; i < (int)(cfg->nH * cfg->headDim); i++)
          aNorm += a_ptr[i] * a_ptr[i];
        printf("[METAL L00 POST-SDPA] attn[0]=%+.4f norm=%.4f\n", a_ptr[0],
               sqrtf(aNorm));
        cmd = [g_queue commandBuffer];
        enc = [cmd computeCommandEncoder];
      }

      // 6. xb = wO(attnFull)
      encode_matvec_buf(enc, pipelines, g_woff_xb, g_woff_attnFull, cfg->bufWO,
                        cfg->offWO, kDim, nDim, cfg->typeWO, 1);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 7 & 8. Fused: PostAttnNorm(xb) + (x += xb) + FfnNorm(x) -> xb
      if (pipelines->post_attn_add_ffn_norm != nil) {
        encode_post_attn_add_ffn_norm(
            enc, g_woff_xb, g_woff_x, bufPostAttnAll, normOff, bufFfnNormAll,
            normOff, nDim, eps, cfg->hasPostAttnNorm != 0, pipelines, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      } else {
        if (cfg->hasPostAttnNorm && bufPostAttnAll) {
          encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostAttnAll, normOff,
                              nDim, eps, 0, pipelines);
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, bufFfnNormAll, normOff,
                            nDim, eps, 0, pipelines);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      // 9. Gate + Up + Act (GeGLU) -> g_woff_ffn_hb
      BOOL can_fuse =
          ((cfg->typeGate == 12 && cfg->typeUp == 12 &&
            pipelines->ffn_gate_up_act_q4_k_mr4 != nil) ||
           (cfg->typeGate == 14 && cfg->typeUp == 14 &&
            pipelines->ffn_gate_up_act_q6_k_mr4 != nil));
      BOOL gate_up_mr8 =
          (cfg->typeGate == 12 && cfg->typeUp == 12 &&
           pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled());
      if (can_fuse) {
        id<MTLComputePipelineState> fuseP;
        if (gate_up_mr8) {
          fuseP = pipelines->ffn_gate_up_act_q4_k_mr8;
        } else if (cfg->typeGate == 14) {
          fuseP = pipelines->ffn_gate_up_act_q6_k_mr4;
        } else {
          fuseP = pipelines->ffn_gate_up_act_q4_k_mr4;
        }
        uint32_t actK = 2; // GeGLU
        [enc setComputePipelineState:fuseP];
        [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
        [enc setBuffer:g_woff_xb offset:0 atIndex:1];
        [enc setBuffer:cfg->bufGate offset:cfg->offGate atIndex:2];
        [enc setBuffer:cfg->bufUp offset:cfg->offUp atIndex:3];
        [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
        [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
        [enc setBytes:&actK length:sizeof(actK) atIndex:6];
        if (gate_up_mr8) {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nFfn + 7) / 8, 1,
                                                1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
          [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nFfn + 3) / 4, 1,
                                                1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
      } else {
        encode_matvec_sub(enc, pipelines, g_woff_ffn_gate, g_woff_xb, cfg->gate,
                          nDim, nFfn, cfg->typeGate, 1);
        encode_matvec_sub(enc, pipelines, g_woff_ffn_up, g_woff_xb, cfg->up,
                          nDim, nFfn, cfg->typeUp, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        id<MTLComputePipelineState> actP = pipelines->gelu_ffn;
        if (actP != nil) {
          [enc setComputePipelineState:actP];
          [enc setBuffer:g_woff_ffn_hb offset:0 atIndex:0];
          [enc setBuffer:g_woff_ffn_gate offset:0 atIndex:1];
          [enc setBuffer:g_woff_ffn_up offset:0 atIndex:2];
          [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:3];
          NSUInteger actTh =
              MIN((NSUInteger)256, actP.maxTotalThreadsPerThreadgroup);
          [enc dispatchThreads:MTLSizeMake((NSUInteger)nFfn, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(actTh, 1, 1)];
        }
      }
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 10. xb = down(hb)
      encode_matvec_buf(enc, pipelines, g_woff_xb, g_woff_ffn_hb, cfg->bufDown,
                        cfg->offDown, nFfn, nDim, cfg->typeDown, 1);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 10b, 11 & 12. Fused: PostFfnNorm(xb) + (x += xb) + OutScale(x)
      if (pipelines->post_ffn_add_out_scale != nil) {
        encode_post_ffn_add_out_scale(
            enc, g_woff_xb, g_woff_x, bufPostFfnAll, normOff, nDim, eps,
            cfg->hasPostFfnNorm != 0, cfg->outScale, pipelines, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      } else {
        if (cfg->hasPostFfnNorm && bufPostFfnAll) {
          encode_woff_rmsnorm(enc, g_woff_xb, g_woff_xb, bufPostFfnAll, normOff,
                              nDim, eps, 0, pipelines);
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        encode_woff_add(enc, g_woff_x, g_woff_x, g_woff_xb, nDim, pipelines);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (cfg->outScale > 0.0f && cfg->outScale != 1.0f &&
            pipelines->scale != nil) {
          float os = cfg->outScale;
          [enc setComputePipelineState:pipelines->scale];
          [enc setBuffer:g_woff_x offset:0 atIndex:0];
          [enc setBytes:&os length:sizeof(os) atIndex:1];
          [enc setBytes:&nDim length:sizeof(nDim) atIndex:2];
          NSUInteger th = MIN((NSUInteger)256, (NSUInteger)nDim);
          [enc dispatchThreads:MTLSizeMake((NSUInteger)nDim, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
      }
    }

    if (debugLayers) {
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      float *x_ptr = (float *)[g_woff_x contents];
      float sumSq = 0.0f;
      for (int k = 0; k < (int)(dimBytes / sizeof(float)); k++)
        sumSq += x_ptr[k] * x_ptr[k];
      printf("[METAL L%02d] x[0]=%+.4f, norm=%.4f\n", g_num_decode_layers - 1,
             x_ptr[0], sqrtf(sumSq));
      cmd = [g_queue commandBuffer];
      enc = [cmd computeCommandEncoder];
    }

    // ── Final output logits projection ──────────────────────────────────────
    if (g_output_w != NULL && g_buf_output_norm != nil &&
        g_buf_output_w != nil && g_buf_output_logits != nil) {
      uint32_t outK = (uint32_t)g_output_K;
      uint32_t outN = (uint32_t)g_output_N;
      encode_woff_rmsnorm(enc, g_woff_xb, g_woff_x, g_buf_output_norm,
                          g_off_output_norm, outK, eps, 0, pipelines);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_matvec_buf(enc, pipelines, g_buf_output_logits, g_woff_xb,
                        g_buf_output_w, g_off_output_w, outK, outN,
                        g_output_type_w, 1);
      if (g_output_logit_cap > 0.0f && pipelines->softcap != nil) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        float cap = g_output_logit_cap;
        [enc setComputePipelineState:pipelines->softcap];
        [enc setBuffer:g_buf_output_logits offset:0 atIndex:0];
        [enc setBytes:&cap length:sizeof(cap) atIndex:1];
        [enc setBytes:&outN length:sizeof(outN) atIndex:2];
        NSUInteger th = MIN((NSUInteger)256,
                            pipelines->softcap.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)outN, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(th, 1, 1)];
      }
    }

    [enc endEncoding];
    uint64_t profileEncodedNanos =
        profileDecode ? monotonic_nanos() : 0;
    uint64_t profileSubmitNanos =
        profileDecode ? monotonic_nanos() : 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    uint64_t profileCompletedNanos =
        profileDecode ? monotonic_nanos() : 0;

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      NSError *commandError = [cmd error];
      fprintf(stderr,
              "[aljabr-metal] decode_step command buffer failed pos=%d status=%ld error_domain=%s error_code=%ld error=%s\n",
              pos, (long)[cmd status],
              commandError != nil ? [[commandError domain] UTF8String] : "(none)",
              commandError != nil ? (long)[commandError code] : 0L,
              commandError != nil ? [[commandError localizedDescription] UTF8String] : "(none)");
      return -4;
    }

    if (profileDecode) {
      double gpuMs = 0.0;
      if (cmd.GPUStartTime > 0.0 && cmd.GPUEndTime >= cmd.GPUStartTime)
        gpuMs = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
      double wallMs =
          (double)(profileCompletedNanos - profileStartNanos) / 1000000.0;
      double encodeMs =
          (double)(profileEncodedNanos - profileStartNanos) / 1000000.0;
      double submitWaitMs =
          (double)(profileCompletedNanos - profileSubmitNanos) / 1000000.0;
      fprintf(stderr,
              "[METAL DECODE PROFILE] pos=%d layers=%d path=%s wall_ms=%.3f encode_ms=%.3f submit_wait_ms=%.3f gpu_ms=%.3f\n",
              pos, g_num_decode_layers, useTokenEmbedding ? "token" : "vector",
              wallMs, encodeMs, submitWaitMs, gpuMs);
    }

    int best_token = 0;
    if (g_buf_output_logits != nil && g_output_N > 0) {
      best_token = aljabr_metal_argmax_f32([g_buf_output_logits contents],
                                           g_output_N, -1, -1, -1, -1, -1, -1,
                                           -1, -1);
    }

    if (logits_out != NULL && g_buf_output_logits != nil) {
      memcpy(logits_out, [g_buf_output_logits contents],
             (size_t)g_output_N * sizeof(float));
      if (debugLayers) {
        int best = 0;
        for (int i = 1; i < (int)g_output_N; i++)
          if (logits_out[i] > logits_out[best])
            best = i;
        printf("[METAL LOGITS pos=27] argmax=%d score=%.4f (target 236828 "
               "score=%.4f, 236772 score=%.4f)\n",
               best, logits_out[best], logits_out[236828], logits_out[236772]);
      }
    }
    return (best_token >= 0) ? best_token : 0;
  }
}

 
// ── Q4_K token embedding entry points ───────────────────────────────────────
// The token path encodes embedding and transformer decode into one command
// buffer. The embedding kernel writes directly into the decoder's input scratch
// vector, avoiding an intermediate shared buffer, CPU wait, and second submit.

int aljabr_metal_register_token_embd(const void *embdW, int typeEmbdW,
                                     int dim, float scale) {
  if (!g_initialized || g_device == nil || g_queue == nil)
    return -1;
  // The current shader decodes GGML Q4_K blocks (256 elements, 144 bytes).
  if (embdW == NULL || typeEmbdW != 12 || dim <= 0 || (dim % 256) != 0 ||
      !isfinite(scale) || g_output_N <= 0)
    return -2;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines == NULL || pipelines->embed_token_q4_k == nil)
    return -3;

  size_t rowBytes = (size_t)(dim / 256) * 144u;
  if ((size_t)g_output_N > SIZE_MAX / rowBytes)
    return -4;
  size_t weightBytes = (size_t)g_output_N * rowBytes;
  size_t offset = 0;
  id<MTLBuffer> weights =
      wrap_weight_ptr_offset(embdW, weightBytes, &offset);
  if (weights == nil || offset > weights.length ||
      weightBytes > weights.length - offset)
    return -5;

  g_token_embd_buffer = weights;
  g_token_embd_offset = offset;
  g_token_embd_dim = dim;
  g_token_embd_vocab = g_output_N;
  g_token_embd_scale = scale;
  return 0;
}

int aljabr_metal_decode_step(int pos, int slidingWindow, float eps,
                             const float *x_in, float *logits_out) {
  return aljabr_metal_decode_step_internal(pos, slidingWindow, eps, x_in,
                                           logits_out, -1, NO);
}

int aljabr_metal_decode_step_token(int pos, int slidingWindow, float eps,
                                   int tokenId, float *logits_out) {
  if (tokenId < 0 || tokenId >= g_token_embd_vocab ||
      g_token_embd_buffer == nil || g_token_embd_dim <= 0)
    return -1;
  if (!g_initialized || g_queue == nil)
    return -2;

  // Diagnostic fallback: preserve the former two-command-buffer path so a
  // developer can compare logits and latency with the fused path on-device.
  // The fused path remains the default and does not allocate/copy this vector.
  if (env_truthy("ALJABR_METAL_DISABLE_FUSED_TOKEN_EMBEDDING")) {
    // Allocate the diagnostic output lazily so the default fused path has no
    // extra shared-buffer allocation or retained host-visible memory.
    size_t outputBytes = (size_t)g_token_embd_dim * sizeof(float);
    if (g_token_embd_output == nil ||
        g_token_embd_output_capacity < outputBytes) {
      id<MTLBuffer> output =
          [g_device newBufferWithLength:outputBytes
                                options:MTLResourceStorageModeShared];
      if (output == nil)
        return -6;
      g_token_embd_output = output;
      g_token_embd_output_capacity = outputBytes;
    }
    if (g_num_decode_layers <= 0 || g_decode_layers[0].wO_N <= 0 ||
        g_token_embd_dim != g_decode_layers[0].wO_N)
      return -6;

    @autoreleasepool {
      AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
      if (pipelines == NULL || pipelines->embed_token_q4_k == nil)
        return -3;
      uint32_t token = (uint32_t)tokenId;
      uint32_t K = (uint32_t)g_token_embd_dim;
      float scale = g_token_embd_scale;
      id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
      if (cmd == nil)
        return -4;
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      if (enc == nil)
        return -4;
      [enc setComputePipelineState:pipelines->embed_token_q4_k];
      [enc setBuffer:g_token_embd_output offset:0 atIndex:0];
      [enc setBuffer:g_token_embd_buffer offset:g_token_embd_offset atIndex:1];
      [enc setBytes:&token length:sizeof(token) atIndex:2];
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&scale length:sizeof(scale) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(K / 256u), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if ([cmd status] != MTLCommandBufferStatusCompleted)
        return -5;

      return aljabr_metal_decode_step_internal(
          pos, slidingWindow, eps,
          (const float *)[g_token_embd_output contents], logits_out, -1, NO);
    }
  }

  return aljabr_metal_decode_step_internal(pos, slidingWindow, eps, NULL,
                                           logits_out, tokenId, YES);
}

// ── Batched Fused wO + Norms + Residuals + FFN (Prefill Step) ──────────────

static id<MTLBuffer> g_scratch_wo_ffn_x = nil;
static size_t g_scratch_wo_ffn_x_bytes = 0;
static id<MTLBuffer> g_scratch_wo_ffn_attn = nil;
static size_t g_scratch_wo_ffn_attn_bytes = 0;
static id<MTLBuffer> g_scratch_wo_ffn_xb = nil;
static size_t g_scratch_wo_ffn_xb_bytes = 0;

int aljabr_metal_wo_ffn_fused_layer_batch(
    float *x, const float *attn_full,
    int layer, int batchSize, float eps, int activationKind) {
  if (!g_initialized || layer < 0 || layer >= g_num_decode_layers || batchSize <= 0) {
    ALJABR_METAL_LOG("[wo_ffn_batch] rc=-1: init=%d layer=%d num_layers=%d bSz=%d\n",
                     g_initialized, layer, g_num_decode_layers, batchSize);
    return -1;
  }
  DecodeLayerConfig *cfg = &g_decode_layers[layer];
  if (cfg->bufWO == nil || cfg->bufDown == nil) {
    ALJABR_METAL_LOG("[wo_ffn_batch] rc=-2: bufWO=%p bufDown=%p\n", cfg->bufWO, cfg->bufDown);
    return -2;
  }
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->post_attn_add_ffn_norm == nil || pipelines->post_ffn_add_out_scale == nil) {
    ALJABR_METAL_LOG("[wo_ffn_batch] rc=-3: post_attn=%p post_ffn=%p\n",
                     pipelines->post_attn_add_ffn_norm, pipelines->post_ffn_add_out_scale);
    return -3;
  }

  @autoreleasepool {
    size_t bSz = (size_t)batchSize;
    uint32_t nDim = (uint32_t)cfg->wO_N;
    uint32_t nFfn = (uint32_t)cfg->ffnDim;
    uint32_t kDim = (uint32_t)cfg->wO_K;
    size_t dimBytes = bSz * (size_t)nDim * sizeof(float);
    size_t attnBytes = bSz * (size_t)kDim * sizeof(float);
    size_t ffnBytes = bSz * (size_t)nFfn * sizeof(float);

    id<MTLBuffer> bufX = ensure_shared_buf(&g_scratch_wo_ffn_x, &g_scratch_wo_ffn_x_bytes, dimBytes);
    id<MTLBuffer> bufAttn = ensure_shared_buf(&g_scratch_wo_ffn_attn, &g_scratch_wo_ffn_attn_bytes, attnBytes);
    id<MTLBuffer> bufXB = ensure_shared_buf(&g_scratch_wo_ffn_xb, &g_scratch_wo_ffn_xb_bytes, dimBytes);
    if (!bufX || !bufAttn || !bufXB) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-4: shared bufs failed\n");
      return -4;
    }

    if (g_scratch_hb == nil || g_scratch_ffn_bytes < ffnBytes) {
      size_t sz = (ffnBytes + 16383) & ~16383;
      g_scratch_hb = [g_device newBufferWithLength:sz options:MTLResourceStorageModePrivate];
      g_scratch_ffn_bytes = sz;
    }
    if (!g_scratch_hb) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-4: g_scratch_hb failed\n");
      return -4;
    }

    memcpy([bufX contents], x, dimBytes);
    memcpy([bufAttn contents], attn_full, attnBytes);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // 1. wO projection: bufAttn -> bufXB
    if (!encode_matvec_buf(enc, pipelines, bufXB, bufAttn, cfg->bufWO, cfg->offWO,
                           kDim, nDim, cfg->typeWO, (uint32_t)bSz)) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-5: encode wO failed typeWO=%d\n", cfg->typeWO);
      [enc endEncoding];
      return -5;
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 2. Fused PostAttnNorm(xb) + (x += xb) + FfnNorm(x) -> xb
    id<MTLBuffer> bufPostAttnAll = g_norm_buffers[0];
    id<MTLBuffer> bufFfnNormAll = g_norm_buffers[1];
    id<MTLBuffer> bufPostFfnAll = g_norm_buffers[2];
    size_t normOff = (size_t)layer * (size_t)nDim * sizeof(float);

    encode_post_attn_add_ffn_norm(enc, bufXB, bufX, bufPostAttnAll, normOff,
                                 bufFfnNormAll, normOff, nDim, eps,
                                 cfg->hasPostAttnNorm != 0, pipelines, (uint32_t)bSz);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 3. FFN Gate + Up + Act -> g_scratch_hb
    BOOL use_b4 = (bSz > 1 && bSz <= kMaxValidatedB4BatchSize &&
                   ((cfg->typeGate == 12 && cfg->typeUp == 12 && pipelines->ffn_gate_up_act_q4_k_b4_mr4 != nil) ||
                    (cfg->typeGate == 14 && cfg->typeUp == 14 && pipelines->ffn_gate_up_act_q6_k_b4_mr4 != nil)));
    id<MTLComputePipelineState> fuseP;
    if (use_b4) {
      fuseP = (cfg->typeGate == 14) ? pipelines->ffn_gate_up_act_q6_k_b4_mr4 : pipelines->ffn_gate_up_act_q4_k_b4_mr4;
    } else if (cfg->typeGate == 12 && pipelines->ffn_gate_up_act_q4_k_mr8 != nil && mr8_enabled()) {
      fuseP = pipelines->ffn_gate_up_act_q4_k_mr8;
    } else if (cfg->typeGate == 14) {
      fuseP = pipelines->ffn_gate_up_act_q6_k_mr4;
    } else {
      fuseP = pipelines->ffn_gate_up_act_q4_k_mr4;
    }
    if (fuseP == nil) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-6: fuseP nil typeGate=%d typeUp=%d use_b4=%d\n",
                       cfg->typeGate, cfg->typeUp, (int)use_b4);
      [enc endEncoding];
      return -6;
    }

    [enc setComputePipelineState:fuseP];
    [enc setBuffer:g_scratch_hb offset:0 atIndex:0];
    [enc setBuffer:bufXB offset:0 atIndex:1];
    [enc setBuffer:cfg->bufGate offset:cfg->offGate atIndex:2];
    [enc setBuffer:cfg->bufUp offset:cfg->offUp atIndex:3];
    [enc setBytes:&nDim length:sizeof(nDim) atIndex:4];
    [enc setBytes:&nFfn length:sizeof(nFfn) atIndex:5];
    uint32_t actKind = (uint32_t)activationKind;
    [enc setBytes:&actKind length:sizeof(actKind) atIndex:6];
    if (use_b4) {
      uint32_t bSz32 = (uint32_t)bSz;
      [enc setBytes:&bSz32 length:sizeof(bSz32) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nFfn + 3) / 4, (bSz + 1) / 2, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (fuseP == pipelines->ffn_gate_up_act_q4_k_mr8) {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nFfn + 7) / 8, bSz, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)nFfn + 3) / 4, bSz, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 4. Down projection: g_scratch_hb -> bufXB
    if (!encode_matvec_buf(enc, pipelines, bufXB, g_scratch_hb, cfg->bufDown, cfg->offDown,
                           nFfn, nDim, cfg->typeDown, (uint32_t)bSz)) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-7: encode Down failed typeDown=%d\n", cfg->typeDown);
      [enc endEncoding];
      return -7;
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // 5. PostFfnNorm(xb) + (x += xb) + OutScale(x) -> bufX
    encode_post_ffn_add_out_scale(enc, bufXB, bufX, bufPostFfnAll, normOff, nDim, eps,
                                 cfg->hasPostFfnNorm != 0, cfg->outScale, pipelines, (uint32_t)bSz);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      ALJABR_METAL_LOG("[wo_ffn_batch] rc=-8: cmd failed status=%lu\n", (unsigned long)[cmd status]);
      return -8;
    }

    memcpy(x, [bufX contents], dimBytes);
    return 0;
  }
}
