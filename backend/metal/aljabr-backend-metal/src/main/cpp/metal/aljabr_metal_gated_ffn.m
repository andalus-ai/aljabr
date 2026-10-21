/**
 * aljabr_metal_gated_ffn.m — Metal gated FFN public kernels.
 */

#import "aljabr_metal_buffers.h"
#import "aljabr_metal_gated_ffn.h"
#import "aljabr_metal_matvec_tuning.h"
#import "aljabr_metal_mps_cache.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_support.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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

static int aljabr_metal_gated_ffn_half_impl(
    void *C, const void *A, const void *gateW, const void *upW,
    const void *downW, int M, int input_dim, int intermediate_dim,
    int output_dim, int is_bf16,
    id<MTLComputePipelineState> activation_pipeline) {
  if (!g_initialized)
    return -1;
  if (activation_pipeline == nil)
    return -3;
  if (M <= 0 || input_dim <= 0 || intermediate_dim <= 0 || output_dim <= 0)
    return -2;

  @autoreleasepool {
    MPSDataType b_mps_type;
    if (!resolve_half_weight_type(is_bf16, &b_mps_type)) {
      return -2;
    }
    int b_elem_size = 2;

    size_t activation_bytes = (size_t)M * intermediate_dim * sizeof(float);
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * output_dim * sizeof(float));
    id<MTLBuffer> bufA =
        wrap_ptr((void *)A, (size_t)M * input_dim * sizeof(float));
    id<MTLBuffer> bufGateW = wrap_weight_ptr(
        gateW, (size_t)intermediate_dim * input_dim * b_elem_size);
    id<MTLBuffer> bufUpW = wrap_weight_ptr(upW, (size_t)intermediate_dim *
                                                    input_dim * b_elem_size);
    id<MTLBuffer> bufDownW = wrap_weight_ptr(
        downW, (size_t)output_dim * intermediate_dim * b_elem_size);
    id<MTLBuffer> bufGate = nil;
    id<MTLBuffer> bufUp = nil;
    id<MTLBuffer> bufCombined = nil;
    if (!ensure_swiglu_scratch(activation_bytes, &bufGate, &bufUp,
                               &bufCombined)) {
      return -4;
    }
    if (bufC == nil || bufA == nil || bufGateW == nil || bufUpW == nil ||
        bufDownW == nil || bufGate == nil || bufUp == nil ||
        bufCombined == nil) {
      return -4;
    }

    MPSMatrixDescriptor *descInput = cached_matrix_descriptor(
        M, input_dim, input_dim * sizeof(float), MPSDataTypeFloat32);
    MPSMatrixDescriptor *descGateWeight = cached_matrix_descriptor(
        intermediate_dim, input_dim, input_dim * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descActivation = cached_matrix_descriptor(
        M, intermediate_dim, intermediate_dim * sizeof(float),
        MPSDataTypeFloat32);
    MPSMatrixDescriptor *descDownWeight =
        cached_matrix_descriptor(output_dim, intermediate_dim,
                                 intermediate_dim * b_elem_size, b_mps_type);
    MPSMatrixDescriptor *descOutput = cached_matrix_descriptor(
        M, output_dim, output_dim * sizeof(float), MPSDataTypeFloat32);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA
                                             descriptor:descInput];
    MPSMatrix *matGateW = [[MPSMatrix alloc] initWithBuffer:bufGateW
                                                 descriptor:descGateWeight];
    MPSMatrix *matUpW = [[MPSMatrix alloc] initWithBuffer:bufUpW
                                               descriptor:descGateWeight];
    MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate
                                                descriptor:descActivation];
    MPSMatrix *matUp = [[MPSMatrix alloc] initWithBuffer:bufUp
                                              descriptor:descActivation];
    MPSMatrix *matCombined = [[MPSMatrix alloc] initWithBuffer:bufCombined
                                                    descriptor:descActivation];
    MPSMatrix *matDownW = [[MPSMatrix alloc] initWithBuffer:bufDownW
                                                 descriptor:descDownWeight];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC
                                             descriptor:descOutput];

    MPSMatrixMultiplication *upProj =
        cached_mmul(NO, YES, M, intermediate_dim, input_dim, MPSDataTypeFloat32,
                    b_mps_type, MPSDataTypeFloat32, 1.0f, 0.0f);
    MPSMatrixMultiplication *downProj = cached_mmul(
        NO, YES, M, output_dim, intermediate_dim, MPSDataTypeFloat32,
        b_mps_type, MPSDataTypeFloat32, 1.0f, 0.0f);

    uint32_t activation_count = (uint32_t)(M * intermediate_dim);
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [upProj encodeToCommandBuffer:cmd
                       leftMatrix:matA
                      rightMatrix:matGateW
                     resultMatrix:matGate];
    [upProj encodeToCommandBuffer:cmd
                       leftMatrix:matA
                      rightMatrix:matUpW
                     resultMatrix:matUp];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:activation_pipeline];
    [enc setBuffer:bufCombined offset:0 atIndex:0];
    [enc setBuffer:bufGate offset:0 atIndex:1];
    [enc setBuffer:bufUp offset:0 atIndex:2];
    [enc setBytes:&activation_count length:sizeof(activation_count) atIndex:3];
    NSUInteger threads =
        MIN((NSUInteger)256, activation_pipeline.maxTotalThreadsPerThreadgroup);
    if (threads < 1)
      threads = 1;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)activation_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];

    [downProj encodeToCommandBuffer:cmd
                         leftMatrix:matCombined
                        rightMatrix:matDownW
                       resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int aljabr_metal_swiglu_ffn_half(void *C, const void *A, const void *gateW,
                                      const void *upW, const void *downW, int M,
                                      int input_dim, int intermediate_dim,
                                      int output_dim, int is_bf16) {
  return aljabr_metal_gated_ffn_half_impl(
      C, A, gateW, upW, downW, M, input_dim, intermediate_dim, output_dim,
      is_bf16, aljabr_metal_pipelines()->silu_ffn);
}

int aljabr_metal_geglu_ffn_half(void *C, const void *A, const void *gateW,
                                     const void *upW, const void *downW, int M,
                                     int input_dim, int intermediate_dim,
                                     int output_dim, int is_bf16) {
  return aljabr_metal_gated_ffn_half_impl(
      C, A, gateW, upW, downW, M, input_dim, intermediate_dim, output_dim,
      is_bf16, aljabr_metal_pipelines()->gelu_ffn);
}

static int aljabr_metal_gated_ffn_matvec_half_impl(
    void *C, const void *A, const void *gateW, const void *upW,
    const void *downW, int input_dim, int intermediate_dim, int output_dim,
    int is_bf16, int activation_kind,
    id<MTLComputePipelineState> activation_pipeline) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized)
    return -1;
  if (activation_pipeline == nil)
    return -3;
  BOOL useFusedGateProjection =
      should_use_fused_gated_ffn_matvec(is_bf16, input_dim, intermediate_dim);
  id<MTLComputePipelineState> pairPipeline256 =
      useFusedGateProjection ? (is_bf16 ? pipelines->matvec_bf16_gated_pair
                                        : pipelines->matvec_half_gated_pair)
                             : (is_bf16 ? pipelines->matvec_bf16_pair
                                        : pipelines->matvec_half_pair);
  id<MTLComputePipelineState> pairPipeline128 =
      useFusedGateProjection ? (is_bf16 ? pipelines->matvec_bf16_gated_pair_128
                                        : pipelines->matvec_half_gated_pair_128)
                             : (is_bf16 ? pipelines->matvec_bf16_pair_128
                                        : pipelines->matvec_half_pair_128);
  BOOL useBf16FusedGateX4 =
      is_bf16 &&
      aljabr_metal_env_truthy(
          "ALJABR_METAL_ENABLE_BF16_FUSED_GATED_PAIR_X4") &&
      !aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_BF16_FUSED_GATED_PAIR_X4") &&
      should_use_bf16_matvec_x4(input_dim, intermediate_dim) &&
      pipelines->matvec_bf16_gated_pair_x4 != nil;
  if (useBf16FusedGateX4) {
    useFusedGateProjection = YES;
    pairPipeline256 = pipelines->matvec_bf16_gated_pair_x4;
    pairPipeline128 = nil;
  }
  if (is_bf16 && useFusedGateProjection &&
      pipelines->matvec_bf16_gated_pair_simd != nil && !useBf16FusedGateX4 &&
      !aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_BF16_FUSED_GATED_PAIR_SIMD")) {
    pairPipeline256 = pipelines->matvec_bf16_gated_pair_simd;
    pairPipeline128 = nil;
  }
  if (pairPipeline256 == nil && useFusedGateProjection) {
    useFusedGateProjection = NO;
    pairPipeline256 =
        is_bf16 ? pipelines->matvec_bf16_pair : pipelines->matvec_half_pair;
    pairPipeline128 = is_bf16 ? pipelines->matvec_bf16_pair_128
                              : pipelines->matvec_half_pair_128;
  }
  BOOL useBf16PairX4 = is_bf16 && !useFusedGateProjection &&
                       should_use_bf16_matvec_x4(input_dim, intermediate_dim) &&
                       pipelines->matvec_bf16_pair_x4 != nil;
  if (useBf16PairX4) {
    pairPipeline256 = should_use_simdgroup_reduction() &&
                              pipelines->matvec_bf16_pair_x4_simd != nil
                          ? pipelines->matvec_bf16_pair_x4_simd
                          : pipelines->matvec_bf16_pair_x4;
    pairPipeline128 = nil;
  } else if (is_bf16 && !useFusedGateProjection &&
             should_use_bf16_pair_simd_reduction(input_dim, intermediate_dim) &&
             pipelines->matvec_bf16_pair_simd != nil) {
    pairPipeline256 = pipelines->matvec_bf16_pair_simd;
    pairPipeline128 = nil;
  }
  id<MTLComputePipelineState> downPipeline256 =
      is_bf16 ? pipelines->matvec_bf16 : pipelines->matvec_half;
  id<MTLComputePipelineState> downPipeline128 =
      is_bf16 ? pipelines->matvec_bf16_128 : pipelines->matvec_half_128;
  BOOL useBf16DownX4 =
      is_bf16 && should_use_bf16_matvec_x4(intermediate_dim, output_dim) &&
      pipelines->matvec_bf16_x4 != nil;
  if (useBf16DownX4) {
    downPipeline256 = should_use_simdgroup_reduction() &&
                              pipelines->matvec_bf16_x4_simd != nil
                          ? pipelines->matvec_bf16_x4_simd
                          : pipelines->matvec_bf16_x4;
    downPipeline128 = nil;
  }
  if (pairPipeline256 == nil || downPipeline256 == nil)
    return -3;
  if (input_dim <= 0 || intermediate_dim <= 0 || output_dim <= 0)
    return -2;

  @autoreleasepool {
    size_t activation_bytes = (size_t)intermediate_dim * sizeof(float);
    id<MTLBuffer> bufGate = nil;
    id<MTLBuffer> bufUp = nil;
    id<MTLBuffer> bufCombined = nil;
    if (!ensure_swiglu_scratch(activation_bytes, &bufGate, &bufUp,
                               &bufCombined)) {
      return -4;
    }

    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)output_dim * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)input_dim * sizeof(float));
    id<MTLBuffer> bufGateW = wrap_weight_ptr(
        gateW, (size_t)intermediate_dim * input_dim * sizeof(uint16_t));
    id<MTLBuffer> bufUpW = wrap_weight_ptr(
        upW, (size_t)intermediate_dim * input_dim * sizeof(uint16_t));
    id<MTLBuffer> bufDownW = wrap_weight_ptr(
        downW, (size_t)output_dim * intermediate_dim * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufGateW == nil || bufUpW == nil ||
        bufDownW == nil || bufGate == nil || bufUp == nil ||
        bufCombined == nil) {
      return -4;
    }

    NSUInteger pairThreads =
        default_matvec_threads(pairPipeline128, input_dim, intermediate_dim);
    id<MTLComputePipelineState> pairPipeline =
        pairThreads == ALJABR_MATVEC_THREADS_128 ? pairPipeline128
                                                      : pairPipeline256;
    if (pairPipeline == nil ||
        pairPipeline.maxTotalThreadsPerThreadgroup < pairThreads) {
      return -3;
    }

    NSUInteger downThreads =
        default_matvec_threads(downPipeline128, intermediate_dim, output_dim);
    id<MTLComputePipelineState> downPipeline =
        downThreads == ALJABR_MATVEC_THREADS_128 ? downPipeline128
                                                      : downPipeline256;
    if (downPipeline == nil ||
        downPipeline.maxTotalThreadsPerThreadgroup < downThreads) {
      return -3;
    }

    uint32_t inputK = (uint32_t)input_dim;
    uint32_t interN = (uint32_t)intermediate_dim;
    uint32_t outputN = (uint32_t)output_dim;
    uint32_t activationCount = (uint32_t)intermediate_dim;
    uint32_t activationKind = (uint32_t)activation_kind;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

    id<MTLComputeCommandEncoder> pairEnc = [cmd computeCommandEncoder];
    [pairEnc setComputePipelineState:pairPipeline];
    if (useFusedGateProjection) {
      [pairEnc setBuffer:bufCombined offset:0 atIndex:0];
      [pairEnc setBuffer:bufA offset:0 atIndex:1];
      [pairEnc setBuffer:bufGateW offset:0 atIndex:2];
      [pairEnc setBuffer:bufUpW offset:0 atIndex:3];
      [pairEnc setBytes:&inputK length:sizeof(inputK) atIndex:4];
      [pairEnc setBytes:&interN length:sizeof(interN) atIndex:5];
      [pairEnc setBytes:&activationKind
                 length:sizeof(activationKind)
                atIndex:6];
    } else {
      [pairEnc setBuffer:bufGate offset:0 atIndex:0];
      [pairEnc setBuffer:bufUp offset:0 atIndex:1];
      [pairEnc setBuffer:bufA offset:0 atIndex:2];
      [pairEnc setBuffer:bufGateW offset:0 atIndex:3];
      [pairEnc setBuffer:bufUpW offset:0 atIndex:4];
      [pairEnc setBytes:&inputK length:sizeof(inputK) atIndex:5];
      [pairEnc setBytes:&interN length:sizeof(interN) atIndex:6];
    }
    NSUInteger pairGroups = (useBf16PairX4 || useBf16FusedGateX4)
                                ? (((NSUInteger)intermediate_dim + 3u) / 4u)
                                : (NSUInteger)intermediate_dim;
    [pairEnc dispatchThreadgroups:MTLSizeMake(pairGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(pairThreads, 1, 1)];
    [pairEnc endEncoding];

    if (!useFusedGateProjection) {
      id<MTLComputeCommandEncoder> actEnc = [cmd computeCommandEncoder];
      [actEnc setComputePipelineState:activation_pipeline];
      [actEnc setBuffer:bufCombined offset:0 atIndex:0];
      [actEnc setBuffer:bufGate offset:0 atIndex:1];
      [actEnc setBuffer:bufUp offset:0 atIndex:2];
      [actEnc setBytes:&activationCount
                length:sizeof(activationCount)
               atIndex:3];
      NSUInteger activationThreads = MIN(
          (NSUInteger)256, activation_pipeline.maxTotalThreadsPerThreadgroup);
      if (activationThreads < 1)
        activationThreads = 1;
      [actEnc dispatchThreads:MTLSizeMake((NSUInteger)intermediate_dim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(activationThreads, 1, 1)];
      [actEnc endEncoding];
    }

    id<MTLComputeCommandEncoder> downEnc = [cmd computeCommandEncoder];
    [downEnc setComputePipelineState:downPipeline];
    [downEnc setBuffer:bufC offset:0 atIndex:0];
    [downEnc setBuffer:bufCombined offset:0 atIndex:1];
    [downEnc setBuffer:bufDownW offset:0 atIndex:2];
    [downEnc setBytes:&interN length:sizeof(interN) atIndex:3];
    [downEnc setBytes:&outputN length:sizeof(outputN) atIndex:4];
    NSUInteger downGroups = useBf16DownX4 ? (((NSUInteger)output_dim + 3u) / 4u)
                                          : (NSUInteger)output_dim;
    [downEnc dispatchThreadgroups:MTLSizeMake(downGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(downThreads, 1, 1)];
    [downEnc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

static int aljabr_metal_gated_ffn_matvec_rows_bf16_impl(
    void *C, const void *A, const void *gateW, const void *upW,
    const void *downW, int M, int input_dim, int intermediate_dim,
    int output_dim, int activation_kind) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized)
    return -1;
  if (M <= 0 || input_dim <= 0 || intermediate_dim <= 0 || output_dim <= 0)
    return -2;
  id<MTLComputePipelineState> pairPipeline =
      pipelines->matvec_bf16_rows_gated_pair_x4 != nil
          ? pipelines->matvec_bf16_rows_gated_pair_x4
          : pipelines->matvec_bf16_rows_gated_pair;
  id<MTLComputePipelineState> downPipeline =
      pipelines->matvec_bf16_rows_x4 != nil ? pipelines->matvec_bf16_rows_x4
                                            : pipelines->matvec_bf16_rows;
  if (pairPipeline == nil || downPipeline == nil)
    return -3;
  if (pairPipeline.maxTotalThreadsPerThreadgroup <
          ALJABR_MATVEC_THREADS_256 ||
      downPipeline.maxTotalThreadsPerThreadgroup <
          ALJABR_MATVEC_THREADS_256) {
    return -3;
  }
  BOOL usePairX4 = pairPipeline == pipelines->matvec_bf16_rows_gated_pair_x4;
  BOOL useDownX4 = downPipeline == pipelines->matvec_bf16_rows_x4;

  @autoreleasepool {
    size_t activation_bytes = (size_t)M * intermediate_dim * sizeof(float);
    id<MTLBuffer> bufCombined = nil;
    if (!ensure_swiglu_combined_scratch(activation_bytes, &bufCombined)) {
      return -4;
    }

    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * output_dim * sizeof(float));
    id<MTLBuffer> bufA =
        wrap_ptr((void *)A, (size_t)M * input_dim * sizeof(float));
    id<MTLBuffer> bufGateW = wrap_weight_ptr(
        gateW, (size_t)intermediate_dim * input_dim * sizeof(uint16_t));
    id<MTLBuffer> bufUpW = wrap_weight_ptr(
        upW, (size_t)intermediate_dim * input_dim * sizeof(uint16_t));
    id<MTLBuffer> bufDownW = wrap_weight_ptr(
        downW, (size_t)output_dim * intermediate_dim * sizeof(uint16_t));
    if (bufC == nil || bufA == nil || bufGateW == nil || bufUpW == nil ||
        bufDownW == nil || bufCombined == nil) {
      return -4;
    }

    uint32_t rows = (uint32_t)M;
    uint32_t inputK = (uint32_t)input_dim;
    uint32_t interN = (uint32_t)intermediate_dim;
    uint32_t outputN = (uint32_t)output_dim;
    uint32_t activationKind = (uint32_t)activation_kind;
    NSUInteger threads = ALJABR_MATVEC_THREADS_256;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

    id<MTLComputeCommandEncoder> pairEnc = [cmd computeCommandEncoder];
    [pairEnc setComputePipelineState:pairPipeline];
    [pairEnc setBuffer:bufCombined offset:0 atIndex:0];
    [pairEnc setBuffer:bufA offset:0 atIndex:1];
    [pairEnc setBuffer:bufGateW offset:0 atIndex:2];
    [pairEnc setBuffer:bufUpW offset:0 atIndex:3];
    [pairEnc setBytes:&inputK length:sizeof(inputK) atIndex:4];
    [pairEnc setBytes:&interN length:sizeof(interN) atIndex:5];
    [pairEnc setBytes:&activationKind length:sizeof(activationKind) atIndex:6];
    [pairEnc setBytes:&rows length:sizeof(rows) atIndex:7];
    NSUInteger pairGroups = usePairX4
                                ? (((NSUInteger)intermediate_dim + 3u) / 4u)
                                : (NSUInteger)intermediate_dim;
    [pairEnc dispatchThreadgroups:MTLSizeMake(pairGroups, (NSUInteger)M, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [pairEnc endEncoding];

    id<MTLComputeCommandEncoder> downEnc = [cmd computeCommandEncoder];
    [downEnc setComputePipelineState:downPipeline];
    [downEnc setBuffer:bufC offset:0 atIndex:0];
    [downEnc setBuffer:bufCombined offset:0 atIndex:1];
    [downEnc setBuffer:bufDownW offset:0 atIndex:2];
    [downEnc setBytes:&interN length:sizeof(interN) atIndex:3];
    [downEnc setBytes:&outputN length:sizeof(outputN) atIndex:4];
    [downEnc setBytes:&rows length:sizeof(rows) atIndex:5];
    NSUInteger downGroups = useDownX4 ? (((NSUInteger)output_dim + 3u) / 4u)
                                      : (NSUInteger)output_dim;
    [downEnc dispatchThreadgroups:MTLSizeMake(downGroups, (NSUInteger)M, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [downEnc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

int aljabr_metal_bf16_ffn_matvec_rows_variant(void) {
  if (!g_initialized)
    return -1;
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (pipelines->matvec_bf16_rows_gated_pair_x4 != nil &&
      pipelines->matvec_bf16_rows_x4 != nil) {
    return 2;
  }
  if (pipelines->matvec_bf16_rows_gated_pair != nil &&
      pipelines->matvec_bf16_rows != nil) {
    return 1;
  }
  return 0;
}

int aljabr_metal_swiglu_ffn_matvec_half(void *C, const void *A,
                                             const void *gateW, const void *upW,
                                             const void *downW, int input_dim,
                                             int intermediate_dim,
                                             int output_dim) {
  return aljabr_metal_gated_ffn_matvec_half_impl(
      C, A, gateW, upW, downW, input_dim, intermediate_dim, output_dim, 0, 1,
      aljabr_metal_pipelines()->silu_ffn);
}

int aljabr_metal_geglu_ffn_matvec_half(void *C, const void *A,
                                            const void *gateW, const void *upW,
                                            const void *downW, int input_dim,
                                            int intermediate_dim,
                                            int output_dim) {
  return aljabr_metal_gated_ffn_matvec_half_impl(
      C, A, gateW, upW, downW, input_dim, intermediate_dim, output_dim, 0, 2,
      aljabr_metal_pipelines()->gelu_ffn);
}

int aljabr_metal_swiglu_ffn_matvec_bf16(void *C, const void *A,
                                             const void *gateW, const void *upW,
                                             const void *downW, int input_dim,
                                             int intermediate_dim,
                                             int output_dim) {
  return aljabr_metal_gated_ffn_matvec_half_impl(
      C, A, gateW, upW, downW, input_dim, intermediate_dim, output_dim, 1, 1,
      aljabr_metal_pipelines()->silu_ffn);
}

int aljabr_metal_geglu_ffn_matvec_bf16(void *C, const void *A,
                                            const void *gateW, const void *upW,
                                            const void *downW, int input_dim,
                                            int intermediate_dim,
                                            int output_dim) {
  return aljabr_metal_gated_ffn_matvec_half_impl(
      C, A, gateW, upW, downW, input_dim, intermediate_dim, output_dim, 1, 2,
      aljabr_metal_pipelines()->gelu_ffn);
}

int aljabr_metal_swiglu_ffn_matvec_rows_bf16(
    void *C, const void *A, const void *gateW, const void *upW,
    const void *downW, int M, int input_dim, int intermediate_dim,
    int output_dim) {
  return aljabr_metal_gated_ffn_matvec_rows_bf16_impl(
      C, A, gateW, upW, downW, M, input_dim, intermediate_dim, output_dim, 1);
}

int aljabr_metal_geglu_ffn_matvec_rows_bf16(
    void *C, const void *A, const void *gateW, const void *upW,
    const void *downW, int M, int input_dim, int intermediate_dim,
    int output_dim) {
  return aljabr_metal_gated_ffn_matvec_rows_bf16_impl(
      C, A, gateW, upW, downW, M, input_dim, intermediate_dim, output_dim, 2);
}
