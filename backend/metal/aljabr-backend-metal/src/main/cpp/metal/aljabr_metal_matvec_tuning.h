/**
 * aljabr_metal_matvec_tuning.h — matvec thread-width and fast-path policy
 * helpers.
 */

#ifndef ALJABR_METAL_MATVEC_TUNING_H
#define ALJABR_METAL_MATVEC_TUNING_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

static const NSUInteger ALJABR_MATVEC_THREADS_32 = 32;
static const NSUInteger ALJABR_MATVEC_THREADS_128 = 128;
static const NSUInteger ALJABR_MATVEC_THREADS_256 = 256;

BOOL aljabr_metal_prefer_matvec_128(int K, int max_output);
BOOL aljabr_metal_use_matvec_128(id<MTLComputePipelineState> pipeline128,
                                      int K, int max_output);
BOOL aljabr_metal_should_use_bf16_matvec_x4(int K, int max_output);
BOOL aljabr_metal_should_use_bf16_matvec_x8(int K, int max_output);
BOOL aljabr_metal_should_use_simdgroup_reduction(void);
BOOL aljabr_metal_should_use_bf16_pair_simd_reduction(int K,
                                                           int max_output);
BOOL aljabr_metal_should_use_fused_gated_ffn_matvec(int is_bf16,
                                                         int input_dim,
                                                         int intermediate_dim);

NSString *aljabr_metal_matvec_shape_key(const char *op, int K, int N0,
                                             int N1, int N2);
NSUInteger aljabr_metal_cached_matvec_threads(NSString *key);
void aljabr_metal_cache_matvec_threads(NSString *key, NSUInteger threads);
NSUInteger aljabr_metal_forced_matvec_threads(void);
BOOL aljabr_metal_matvec_autotune_enabled(int K, int max_output,
                                               BOOL can128);
NSUInteger aljabr_metal_default_matvec_threads(
    id<MTLComputePipelineState> pipeline128, int K, int max_output);
BOOL aljabr_metal_matvec_autotune_prefers_128(uint64_t nanos128,
                                                   uint64_t nanos256);
void aljabr_metal_log_matvec_autotune(const char *op, int K, int N0,
                                           int N1, int N2, uint64_t nanos128,
                                           uint64_t nanos256,
                                           NSUInteger selected);

#define prefer_matvec_128 aljabr_metal_prefer_matvec_128
#define use_matvec_128 aljabr_metal_use_matvec_128
#define should_use_bf16_matvec_x4 aljabr_metal_should_use_bf16_matvec_x4
#define should_use_bf16_matvec_x8 aljabr_metal_should_use_bf16_matvec_x8
#define should_use_simdgroup_reduction                                         \
  aljabr_metal_should_use_simdgroup_reduction
#define should_use_bf16_pair_simd_reduction                                    \
  aljabr_metal_should_use_bf16_pair_simd_reduction
#define should_use_fused_gated_ffn_matvec                                      \
  aljabr_metal_should_use_fused_gated_ffn_matvec
#define matvec_shape_key aljabr_metal_matvec_shape_key
#define cached_matvec_threads aljabr_metal_cached_matvec_threads
#define cache_matvec_threads aljabr_metal_cache_matvec_threads
#define forced_matvec_threads aljabr_metal_forced_matvec_threads
#define matvec_autotune_enabled aljabr_metal_matvec_autotune_enabled
#define default_matvec_threads aljabr_metal_default_matvec_threads
#define matvec_autotune_prefers_128                                            \
  aljabr_metal_matvec_autotune_prefers_128
#define log_matvec_autotune aljabr_metal_log_matvec_autotune

#endif
