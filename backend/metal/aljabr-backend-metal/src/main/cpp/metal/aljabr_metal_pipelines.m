/**
 * aljabr_metal_pipelines.m — runtime Metal pipeline compilation and
 * storage.
 */

#import "aljabr_metal_kernel_source.h"
#import "aljabr_metal_matvec_tuning.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_support.h"

#include <stdlib.h>
#include <string.h>

static AljabrMetalPipelines g_pipelines;

AljabrMetalPipelines *aljabr_metal_pipelines(void) { return &g_pipelines; }

static id<MTLComputePipelineState> compile_pipeline(id<MTLLibrary> library,
                                                    NSString *name) {
  NSError *error = nil;
  id<MTLFunction> fn = [library newFunctionWithName:name];
  if (fn == nil) {
    fprintf(stderr, "[aljabr-metal] function %s not found in library\n", [name UTF8String]);
    return nil;
  }
  id<MTLComputePipelineState> ps = [g_device newComputePipelineStateWithFunction:fn error:&error];
  if (ps == nil) {
    fprintf(stderr, "[aljabr-metal] pipeline %s failed to compile: %s\n", [name UTF8String], [[error localizedDescription] UTF8String]);
  }
  return ps;
}

static BOOL should_compile_128_matvec_pipelines(void) {
  if (aljabr_metal_env_truthy("ALJABR_METAL_DISABLE_MATVEC_128")) {
    return NO;
  }
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_ENABLE_MATVEC_128_PIPELINES") ||
      aljabr_metal_env_truthy(
          "ALJABR_METAL_ENABLE_MATVEC_AUTOTUNE")) {
    return YES;
  }
  const char *forced = getenv("ALJABR_METAL_MATVEC_THREADS");
  return forced != NULL && strcmp(forced, "128") == 0;
}

static BOOL should_compile_simdgroup_reduction_pipelines(void) {
  return !aljabr_metal_env_truthy(
             "ALJABR_METAL_DISABLE_SIMDGROUP_REDUCTION") &&
         aljabr_metal_env_truthy(
             "ALJABR_METAL_ENABLE_SIMDGROUP_REDUCTION");
}

static BOOL should_compile_bf16_fused_gated_pair_pipeline(void) {
  return aljabr_metal_env_truthy(
      "ALJABR_METAL_ENABLE_FUSED_GATED_FFN_MATVEC");
}

static BOOL should_compile_bf16_fused_gated_pair_x4_pipeline(void) {
  return aljabr_metal_env_truthy(
      "ALJABR_METAL_ENABLE_BF16_FUSED_GATED_PAIR_X4");
}

static BOOL should_compile_bf16_matvec_rows_pipelines(void) {
  return aljabr_metal_env_truthy(
             "ALJABR_METAL_ENABLE_BF16_FFN_MATVEC_ROWS") ||
         should_compile_bf16_fused_gated_pair_pipeline();
}

static BOOL should_compile_bf16_x8_pipeline(void) {
  return aljabr_metal_env_truthy(
             "ALJABR_METAL_ENABLE_BF16_MATVEC_X8") &&
         !aljabr_metal_env_truthy(
             "ALJABR_METAL_DISABLE_BF16_MATVEC_X8");
}

static void warmup_pipelines(AljabrMetalPipelines *pipelines) {
  if (g_queue == nil || g_device == nil) return;
  @autoreleasepool {
    id<MTLBuffer> dummy = [g_device newBufferWithLength:131072 options:MTLResourceStorageModeShared];
    if (dummy == nil) return;
    memset([dummy contents], 0, 131072);

    uint32_t K = 1024;
    uint32_t N = 4;
    uint32_t act = 2; // GeGLU

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Warmup matvec_tb_q4_k_mr4
    if (pipelines->matvec_tb_q4_k_mr4 != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q4_k_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup matvec_tb_q4_k_mr4_softcap
    if (pipelines->matvec_tb_q4_k_mr4_softcap != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q4_k_mr4_softcap];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      float cap = 30.0f;
      [enc setBytes:&cap length:sizeof(cap) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup matvec_tb_q4_k_b4_mr4
    if (pipelines->matvec_tb_q4_k_b4_mr4 != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q4_k_b4_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      uint32_t bSz = 4;
      [enc setBytes:&bSz length:sizeof(bSz) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup ffn_gate_up_act_q4_k_mr4
    if (pipelines->ffn_gate_up_act_q4_k_mr4 != nil) {
      [enc setComputePipelineState:pipelines->ffn_gate_up_act_q4_k_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // hb
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B_gate
      [enc setBuffer:dummy offset:32768 atIndex:3];  // B_up
      [enc setBytes:&K length:sizeof(K) atIndex:4];
      [enc setBytes:&N length:sizeof(N) atIndex:5];
      [enc setBytes:&act length:sizeof(act) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup ffn_gate_up_act_q4_k_b4_mr4
    if (pipelines->ffn_gate_up_act_q4_k_b4_mr4 != nil) {
      [enc setComputePipelineState:pipelines->ffn_gate_up_act_q4_k_b4_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // hb
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B_gate
      [enc setBuffer:dummy offset:32768 atIndex:3];  // B_up
      [enc setBytes:&K length:sizeof(K) atIndex:4];
      [enc setBytes:&N length:sizeof(N) atIndex:5];
      [enc setBytes:&act length:sizeof(act) atIndex:6];
      uint32_t bSz = 4;
      [enc setBytes:&bSz length:sizeof(bSz) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup matvec_tb_q6_k_mr4
    if (pipelines->matvec_tb_q6_k_mr4 != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q6_k_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:49152 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup matvec_tb_q6_k_mr4_softcap
    if (pipelines->matvec_tb_q6_k_mr4_softcap != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q6_k_mr4_softcap];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:49152 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      float cap = 30.0f;
      [enc setBytes:&cap length:sizeof(cap) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup matvec_tb_q6_k_b4_mr4
    if (pipelines->matvec_tb_q6_k_b4_mr4 != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q6_k_b4_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // C
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:49152 atIndex:2];  // B
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      uint32_t bSz = 4;
      [enc setBytes:&bSz length:sizeof(bSz) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup ffn_gate_up_act_q6_k_mr4
    if (pipelines->ffn_gate_up_act_q6_k_mr4 != nil) {
      [enc setComputePipelineState:pipelines->ffn_gate_up_act_q6_k_mr4];
      [enc setBuffer:dummy offset:0 atIndex:0];      // hb
      [enc setBuffer:dummy offset:4096 atIndex:1];   // A
      [enc setBuffer:dummy offset:16384 atIndex:2];  // B_gate
      [enc setBuffer:dummy offset:32768 atIndex:3];  // B_up
      [enc setBytes:&K length:sizeof(K) atIndex:4];
      [enc setBytes:&N length:sizeof(N) atIndex:5];
      [enc setBytes:&act length:sizeof(act) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // Warmup standard fallback kernels
    if (pipelines->matvec_tb_q4_k != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q4_k];
      [enc setBuffer:dummy offset:0 atIndex:0];
      [enc setBuffer:dummy offset:4096 atIndex:1];
      [enc setBuffer:dummy offset:16384 atIndex:2];
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    if (pipelines->matvec_tb_q6_k != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q6_k];
      [enc setBuffer:dummy offset:0 atIndex:0];
      [enc setBuffer:dummy offset:4096 atIndex:1];
      [enc setBuffer:dummy offset:49152 atIndex:2];
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    if (pipelines->matvec_tb_q8_0 != nil) {
      [enc setComputePipelineState:pipelines->matvec_tb_q8_0];
      [enc setBuffer:dummy offset:0 atIndex:0];
      [enc setBuffer:dummy offset:4096 atIndex:1];
      [enc setBuffer:dummy offset:16384 atIndex:2];
      [enc setBytes:&K length:sizeof(K) atIndex:3];
      [enc setBytes:&N length:sizeof(N) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    [enc endEncoding];
    uint64_t tw0 = aljabr_metal_monotonic_nanos();
    [cmd commit];
    [cmd waitUntilCompleted];
    uint64_t tw1 = aljabr_metal_monotonic_nanos();
    if ([cmd status] != MTLCommandBufferStatusCompleted) {
      NSLog(@"[aljabr-metal] warmup failed: %@", [cmd error]);
    } else {
      printf("[aljabr-metal] Pre-warmed MR4 pipelines on GPU in %.2f ms\n", (tw1 - tw0) / 1e6);
    }
  }
}

void aljabr_metal_compile_runtime_pipelines(
    AljabrMetalPipelines *pipelines) {
  if (pipelines == NULL || g_device == nil) {
    return;
  }

  uint64_t t_start = aljabr_metal_monotonic_nanos();
  NSError *error = nil;
  id<MTLLibrary> library =
      [g_device newLibraryWithSource:aljabr_metal_runtime_kernel_source()
                             options:nil
                               error:&error];
  uint64_t t_lib1 = aljabr_metal_monotonic_nanos();
  if (library == nil) {
    fprintf(stderr, "[aljabr-metal] failed to compile runtime kernel library: %s\n", [[error localizedDescription] UTF8String]);
    return;
  }

  pipelines->add = compile_pipeline(library, @"aljabr_add_kernel");
  pipelines->silu_ffn =
      compile_pipeline(library, @"aljabr_silu_ffn_kernel");
  pipelines->gelu_ffn =
      compile_pipeline(library, @"aljabr_gelu_ffn_kernel");
  pipelines->rope = compile_pipeline(library, @"aljabr_rope_kernel");
  pipelines->rope_float2 =
      compile_pipeline(library, @"aljabr_rope_float2_kernel");
  pipelines->rmsnorm = compile_pipeline(library, @"aljabr_rmsnorm_kernel");
  pipelines->rmsnorm_float4 =
      compile_pipeline(library, @"aljabr_rmsnorm_float4_kernel");
  pipelines->rmsnorm_rows =
      compile_pipeline(library, @"aljabr_rmsnorm_rows_kernel");
  pipelines->layernorm =
      compile_pipeline(library, @"aljabr_layernorm_kernel");
  pipelines->layernorm_float4 =
      compile_pipeline(library, @"aljabr_layernorm_float4_kernel");
  pipelines->layernorm_rows =
      compile_pipeline(library, @"aljabr_layernorm_rows_kernel");
  pipelines->silu = compile_pipeline(library, @"aljabr_silu_kernel");
  pipelines->gelu = compile_pipeline(library, @"aljabr_gelu_kernel");
  pipelines->softmax = compile_pipeline(library, @"aljabr_softmax_kernel");
  pipelines->softmax_rows =
      compile_pipeline(library, @"aljabr_softmax_rows_kernel");
  pipelines->softcap =
      compile_pipeline(library, @"aljabr_softcap_kernel");
  pipelines->scale =
      compile_pipeline(library, @"aljabr_scale_kernel");
  pipelines->matvec_half =
      compile_pipeline(library, @"aljabr_matvec_tb_half_kernel");
  pipelines->matvec_t_half =
      compile_pipeline(library, @"aljabr_matvec_t_half_kernel");
  pipelines->matvec_half_pair =
      compile_pipeline(library, @"aljabr_matvec_tb_half_pair_kernel");
  pipelines->matvec_half_gated_pair = compile_pipeline(
      library, @"aljabr_matvec_tb_half_gated_pair_kernel");
  pipelines->matvec_half_triple_mixed = compile_pipeline(
      library, @"aljabr_matvec_tb_half_triple_mixed_kernel");
  pipelines->decode_attention =
      compile_pipeline(library, @"aljabr_decode_attention_kernel");
  pipelines->flash_attention =
      compile_pipeline(library, @"aljabr_flash_attention_kernel");

  NSError *matvec256Error = nil;
  id<MTLLibrary> matvec256Library =
      [g_device newLibraryWithSource:aljabr_metal_matvec_kernel_source(
                                         ALJABR_MATVEC_THREADS_256)
                             options:nil
                               error:&matvec256Error];
  if (matvec256Library != nil) {
    pipelines->matvec_tb_int4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_int4_kernel");
    pipelines->matvec_tb_nf4 =
        compile_pipeline(matvec256Library, @"aljabr_matvec_tb_nf4_kernel");
  } else {
    fprintf(stderr, "[aljabr-metal] failed to compile matvec256 library: %s\n", [[matvec256Error localizedDescription] UTF8String]);
  }
  if (matvec256Library != nil) {
    // GGML block-quantized kernels
    pipelines->matvec_tb_q4_k = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q4_k_kernel");
    pipelines->matvec_tb_q4_k_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q4_k_mr4_kernel");
    pipelines->matvec_tb_q4_k_mr4_softcap = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q4_k_mr4_softcap_kernel");
    pipelines->matvec_tb_q4_k_b4_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q4_k_b4_mr4_kernel");
    pipelines->ffn_gate_up_act_q4_k_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_ffn_gate_up_act_q4_k_mr4_kernel");
    pipelines->ffn_gate_up_act_q4_k_b4_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_ffn_gate_up_act_q4_k_b4_mr4_kernel");
    pipelines->matvec_tb_q6_k = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q6_k_kernel");
    pipelines->matvec_tb_q6_k_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q6_k_mr4_kernel");
    pipelines->matvec_tb_q6_k_mr4_softcap = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q6_k_mr4_softcap_kernel");
    pipelines->matvec_tb_q6_k_b4_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q6_k_b4_mr4_kernel");
    pipelines->ffn_gate_up_act_q6_k_mr4 = compile_pipeline(
        matvec256Library, @"aljabr_ffn_gate_up_act_q6_k_mr4_kernel");
    pipelines->matvec_tb_q8_0 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_q8_0_kernel");
    pipelines->matvec_bf16 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_kernel");
    pipelines->matvec_bf16_pair = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_pair_kernel");
    pipelines->matvec_bf16_pair_simd = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_pair_simd_kernel");
    pipelines->matvec_bf16_x4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_x4_kernel");
    if (should_compile_bf16_x8_pipeline()) {
      pipelines->matvec_bf16_x8 = compile_pipeline(
          matvec256Library, @"aljabr_matvec_tb_bf16_x8_kernel");
    }
    pipelines->matvec_bf16_pair_x4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_pair_x4_kernel");
    if (should_compile_simdgroup_reduction_pipelines()) {
      pipelines->matvec_bf16_x4_simd = compile_pipeline(
          matvec256Library, @"aljabr_matvec_tb_bf16_x4_simd_kernel");
      pipelines->matvec_bf16_pair_x4_simd = compile_pipeline(
          matvec256Library, @"aljabr_matvec_tb_bf16_pair_x4_simd_kernel");
    }
    if (should_compile_bf16_fused_gated_pair_pipeline()) {
      pipelines->matvec_bf16_gated_pair = compile_pipeline(
          matvec256Library, @"aljabr_matvec_tb_bf16_gated_pair_kernel");
      if (!aljabr_metal_env_truthy(
              "ALJABR_METAL_DISABLE_BF16_FUSED_GATED_PAIR_SIMD")) {
        pipelines->matvec_bf16_gated_pair_simd = compile_pipeline(
            matvec256Library,
            @"aljabr_matvec_tb_bf16_gated_pair_simd_kernel");
      }
    }
    if (should_compile_bf16_fused_gated_pair_x4_pipeline()) {
      pipelines->matvec_bf16_gated_pair_x4 = compile_pipeline(
          matvec256Library, @"aljabr_matvec_tb_bf16_gated_pair_x4_kernel");
    }
    if (should_compile_bf16_matvec_rows_pipelines()) {
      pipelines->matvec_bf16_rows_gated_pair = compile_pipeline(
          matvec256Library,
          @"aljabr_matvec_rows_tb_bf16_gated_pair_kernel");
      pipelines->matvec_bf16_rows = compile_pipeline(
          matvec256Library, @"aljabr_matvec_rows_tb_bf16_kernel");
      pipelines->matvec_bf16_rows_gated_pair_x4 = compile_pipeline(
          matvec256Library,
          @"aljabr_matvec_rows_tb_bf16_gated_pair_x4_kernel");
      pipelines->matvec_bf16_rows_x4 = compile_pipeline(
          matvec256Library, @"aljabr_matvec_rows_tb_bf16_x4_kernel");
    }
    pipelines->matvec_bf16_triple_mixed = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_triple_mixed_kernel");
    pipelines->matvec_bf16_triple_mixed_x4 = compile_pipeline(
        matvec256Library, @"aljabr_matvec_tb_bf16_triple_mixed_x4_kernel");
  }

  if (should_compile_128_matvec_pipelines()) {
    NSError *matvec128Error = nil;
    id<MTLLibrary> matvec128Library =
        [g_device newLibraryWithSource:aljabr_metal_matvec_kernel_source(
                                           ALJABR_MATVEC_THREADS_128)
                               options:nil
                                 error:&matvec128Error];
    if (matvec128Library != nil) {
      pipelines->matvec_half_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_half_kernel");
      pipelines->matvec_t_half_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_t_half_kernel");
      pipelines->matvec_tb_int4_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_int4_kernel");
      pipelines->matvec_tb_nf4_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_nf4_kernel");
      pipelines->matvec_half_pair_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_half_pair_kernel");
      pipelines->matvec_half_gated_pair_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_half_gated_pair_kernel");
      pipelines->matvec_half_triple_mixed_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_half_triple_mixed_kernel");
      pipelines->matvec_bf16_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_bf16_kernel");
      pipelines->matvec_bf16_pair_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_bf16_pair_kernel");
      pipelines->matvec_bf16_gated_pair_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_bf16_gated_pair_kernel");
      pipelines->matvec_bf16_triple_mixed_128 = compile_pipeline(
          matvec128Library, @"aljabr_matvec_tb_bf16_triple_mixed_kernel");
    }
  }

  warmup_pipelines(pipelines);

  uint64_t t_end = aljabr_metal_monotonic_nanos();
  printf("[aljabr-metal] Runtime pipelines initialized in %.2f ms (runtime_lib: %.2f ms)\n",
         (t_end - t_start) / 1e6, (t_lib1 - t_start) / 1e6);
}

