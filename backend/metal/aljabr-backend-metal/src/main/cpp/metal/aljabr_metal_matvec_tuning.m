/**
 * aljabr_metal_matvec_tuning.m — matvec dispatch policy and autotune
 * state.
 */

#import "aljabr_metal_matvec_tuning.h"
#import "aljabr_metal_support.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>
#include <stdlib.h>

static NSMutableDictionary<NSString *, NSNumber *> *g_matvec_thread_choices =
    nil;

BOOL aljabr_metal_prefer_matvec_128(int K, int max_output) {
  const char *override = getenv("ALJABR_METAL_MATVEC_THREADS");
  if (override != NULL && override[0] != '\0') {
    long parsed = strtol(override, NULL, 10);
    if (parsed == 128)
      return YES;
    if (parsed == 256)
      return NO;
  }
  if (aljabr_metal_env_truthy("ALJABR_METAL_DISABLE_MATVEC_128")) {
    return NO;
  }
  // Keep the 128-thread path conservative: it helps skinny projections
  // such as compact Qwen FFNs, while large-output Gemma FFN/logit shapes
  // are memory-bandwidth bound and usually prefer the wider 256 reduction.
  int max_inner = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MATVEC_128_MAX_INNER", 768);
  int max_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MATVEC_128_MAX_OUTPUT", 4096);
  return (max_inner <= 0 || K <= max_inner) &&
         (max_out <= 0 || max_output <= max_out);
}

BOOL aljabr_metal_use_matvec_128(id<MTLComputePipelineState> pipeline128,
                                      int K, int max_output) {
  return pipeline128 != nil &&
         aljabr_metal_prefer_matvec_128(K, max_output);
}

BOOL aljabr_metal_should_use_bf16_matvec_x4(int K, int max_output) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_BF16_MATVEC_X4")) {
    return NO;
  }
  const char *override = getenv("ALJABR_METAL_MATVEC_THREADS");
  if (override != NULL && override[0] != '\0') {
    long parsed = strtol(override, NULL, 10);
    if (parsed == 128)
      return NO;
  }
  int min_inner = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X4_MIN_INNER", 512);
  int min_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X4_MIN_OUTPUT", 1024);
  // x4 reduces dispatch pressure for decode FFN projections. Keep it below
  // the huge vocab/logit range, where x8 is the dedicated large-output path.
  int max_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X4_MAX_OUTPUT", 8192);
  return (min_inner <= 0 || K >= min_inner) &&
         (min_out <= 0 || max_output >= min_out) &&
         (max_out <= 0 || max_output <= max_out);
}

BOOL aljabr_metal_should_use_bf16_matvec_x8(int K, int max_output) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_BF16_MATVEC_X8")) {
    return NO;
  }
  if (!aljabr_metal_env_truthy(
          "ALJABR_METAL_ENABLE_BF16_MATVEC_X8")) {
    return NO;
  }
  const char *override = getenv("ALJABR_METAL_MATVEC_THREADS");
  if (override != NULL && override[0] != '\0') {
    long parsed = strtol(override, NULL, 10);
    if (parsed == 128)
      return NO;
  }
  int min_inner = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X8_MIN_INNER", 512);
  int min_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X8_MIN_OUTPUT", 16384);
  int max_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_MATVEC_X8_MAX_OUTPUT", 0);
  return (min_inner <= 0 || K >= min_inner) &&
         (min_out <= 0 || max_output >= min_out) &&
         (max_out <= 0 || max_output <= max_out);
}

BOOL aljabr_metal_should_use_simdgroup_reduction(void) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_SIMDGROUP_REDUCTION")) {
    return NO;
  }
  return aljabr_metal_env_truthy(
      "ALJABR_METAL_ENABLE_SIMDGROUP_REDUCTION");
}

BOOL aljabr_metal_should_use_bf16_pair_simd_reduction(int K,
                                                           int max_output) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_BF16_PAIR_SIMD")) {
    return NO;
  }
  if (aljabr_metal_env_truthy("ALJABR_METAL_ENABLE_BF16_PAIR_SIMD")) {
    return YES;
  }
  const char *override = getenv("ALJABR_METAL_MATVEC_THREADS");
  if (override != NULL && override[0] != '\0') {
    long parsed = strtol(override, NULL, 10);
    if (parsed == 128)
      return NO;
  }
  int min_inner = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_PAIR_SIMD_MIN_INNER", 1024);
  int min_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_PAIR_SIMD_MIN_OUTPUT", 4096);
  int max_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_BF16_PAIR_SIMD_MAX_OUTPUT", 0);
  return (min_inner <= 0 || K >= min_inner) &&
         (min_out <= 0 || max_output >= min_out) &&
         (max_out <= 0 || max_output <= max_out);
}

BOOL aljabr_metal_should_use_fused_gated_ffn_matvec(int is_bf16,
                                                         int input_dim,
                                                         int intermediate_dim) {
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_FUSED_GATED_FFN_MATVEC")) {
    return NO;
  }
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_ENABLE_FUSED_GATED_FFN_MATVEC")) {
    return YES;
  }
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_AUTO_FUSED_GATED_FFN_MATVEC")) {
    return NO;
  }
  if (input_dim <= 0 || intermediate_dim <= 0) {
    return NO;
  }
  if (is_bf16) {
    return NO;
  }
  return YES;
}

NSString *aljabr_metal_matvec_shape_key(const char *op, int K, int N0,
                                             int N1, int N2) {
  return [NSString stringWithFormat:@"%s:%d:%d:%d:%d", op, K, N0, N1, N2];
}

NSUInteger aljabr_metal_cached_matvec_threads(NSString *key) {
  @synchronized([MPSMatrixVectorMultiplication class]) {
    NSNumber *cached = g_matvec_thread_choices != nil
                           ? [g_matvec_thread_choices objectForKey:key]
                           : nil;
    if (cached != nil) {
      NSUInteger value = (NSUInteger)[cached unsignedIntegerValue];
      if (value == ALJABR_MATVEC_THREADS_128 ||
          value == ALJABR_MATVEC_THREADS_256) {
        return value;
      }
    }
  }
  return 0;
}

void aljabr_metal_cache_matvec_threads(NSString *key, NSUInteger threads) {
  if (threads != ALJABR_MATVEC_THREADS_128 &&
      threads != ALJABR_MATVEC_THREADS_256) {
    return;
  }
  @synchronized([MPSMatrixVectorMultiplication class]) {
    if (g_matvec_thread_choices == nil) {
      g_matvec_thread_choices = [[NSMutableDictionary alloc] init];
    }
    [g_matvec_thread_choices setObject:@(threads) forKey:key];
  }
}

NSUInteger aljabr_metal_forced_matvec_threads(void) {
  const char *override = getenv("ALJABR_METAL_MATVEC_THREADS");
  if (override == NULL || override[0] == '\0') {
    return 0;
  }
  long parsed = strtol(override, NULL, 10);
  if (parsed == 128)
    return ALJABR_MATVEC_THREADS_128;
  if (parsed == 256)
    return ALJABR_MATVEC_THREADS_256;
  return 0;
}

BOOL aljabr_metal_matvec_autotune_enabled(int K, int max_output,
                                               BOOL can128) {
  if (!can128 || aljabr_metal_forced_matvec_threads() != 0) {
    return NO;
  }
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_MATVEC_AUTOTUNE")) {
    return NO;
  }
  // Keep the default window conservative for one-shot CLI runs. Autotune is
  // per process, so wider FFN shapes should opt in until safetensor has a
  // persistent execution daemon that can amortize calibration.
  int max_inner = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MATVEC_AUTOTUNE_MAX_INNER", 2048);
  int max_out = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_MATVEC_AUTOTUNE_MAX_OUTPUT", 4096);
  return (max_inner <= 0 || K <= max_inner) &&
         (max_out <= 0 || max_output <= max_out);
}

NSUInteger aljabr_metal_default_matvec_threads(
    id<MTLComputePipelineState> pipeline128, int K, int max_output) {
  NSUInteger forced = aljabr_metal_forced_matvec_threads();
  if (forced == ALJABR_MATVEC_THREADS_128 && pipeline128 != nil) {
    return ALJABR_MATVEC_THREADS_128;
  }
  if (forced == ALJABR_MATVEC_THREADS_256) {
    return ALJABR_MATVEC_THREADS_256;
  }
  return aljabr_metal_use_matvec_128(pipeline128, K, max_output)
             ? ALJABR_MATVEC_THREADS_128
             : ALJABR_MATVEC_THREADS_256;
}

BOOL aljabr_metal_matvec_autotune_prefers_128(uint64_t nanos128,
                                                   uint64_t nanos256) {
  float margin = aljabr_metal_env_float_or_default(
      "ALJABR_METAL_MATVEC_AUTOTUNE_MARGIN", 0.03f);
  if (margin < 0.0f)
    margin = 0.0f;
  return (double)nanos128 < ((double)nanos256 * (1.0 - (double)margin));
}

void aljabr_metal_log_matvec_autotune(const char *op, int K, int N0,
                                           int N1, int N2, uint64_t nanos128,
                                           uint64_t nanos256,
                                           NSUInteger selected) {
  if (!aljabr_metal_env_truthy("ALJABR_METAL_MATVEC_DEBUG") &&
      !aljabr_metal_env_truthy(
          "ALJABR_METAL_MATVEC_AUTOTUNE_DEBUG")) {
    return;
  }
  fprintf(stderr,
          "[aljabr-metal] matvec autotune op=%s K=%d N0=%d N1=%d N2=%d "
          "t128=%.3fms t256=%.3fms selected=%lu\n",
          op, K, N0, N1, N2, (double)nanos128 / 1000000.0,
          (double)nanos256 / 1000000.0, (unsigned long)selected);
}
