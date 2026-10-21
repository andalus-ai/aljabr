/**
 * aljabr_metal_elementwise.m — Metal elementwise public kernels.
 */

#import "aljabr_metal_buffers.h"
#import "aljabr_metal_cpu_fallback.h"
#import "aljabr_metal_elementwise.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_support.h"

static BOOL g_elementwise_enabled = NO;

void aljabr_metal_elementwise_set_enabled(BOOL enabled) {
  g_elementwise_enabled = enabled;
}

static int cpu_rmsnorm_rows(void *out, const void *x, const void *weight,
                            int rows, int N, float eps, int addOne) {
  size_t row_bytes = (size_t)N * sizeof(float);
  for (int row = 0; row < rows; row++) {
    int rc = aljabr_metal_cpu_rmsnorm(
        (char *)out + ((size_t)row * row_bytes),
        (const char *)x + ((size_t)row * row_bytes), weight, N, eps, addOne);
    if (rc != 0)
      return rc;
  }
  return 0;
}

int aljabr_metal_add(void *C, const void *A, const void *B, int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_add(C, A, B, N);
  if (!g_elementwise_enabled || pipelines->add == nil)
    return aljabr_metal_cpu_add(C, A, B, N);

  @autoreleasepool {
    id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
    id<MTLBuffer> bufA = wrap_ptr((void *)A, (size_t)N * sizeof(float));
    id<MTLBuffer> bufB = wrap_ptr((void *)B, (size_t)N * sizeof(float));
    if (bufC == nil || bufA == nil || bufB == nil)
      return aljabr_metal_cpu_add(C, A, B, N);

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->add];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger threads =
        MIN((NSUInteger)256, pipelines->add.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_rope(void *out, const void *x, int N, int headDim,
                           int posOffset, float freqBase, int isNeox) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0 || !g_elementwise_enabled ||
      pipelines->rope == nil) {
    return 1; // Fallback to CPU
  }

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil) {
      return 1; // Fallback to CPU
    }

    unsigned int n = (unsigned int)N;
    unsigned int h = (unsigned int)headDim;
    unsigned int p = (unsigned int)posOffset;
    unsigned int neox = (unsigned int)isNeox;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    // Use float2 kernel when N is even: reads each (x0,x1) pair as float2,
    // halving memory transactions with no change in correctness.
    BOOL useFloat2 = (N % 2 == 0) && (pipelines->rope_float2 != nil);
    id<MTLComputePipelineState> ropePipeline =
        useFloat2 ? pipelines->rope_float2 : pipelines->rope;
    [enc setComputePipelineState:ropePipeline];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&h length:sizeof(h) atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc setBytes:&freqBase length:sizeof(freqBase) atIndex:5];
    [enc setBytes:&neox length:sizeof(neox) atIndex:6];

    NSUInteger threads = (NSUInteger)(n / 2);
    NSUInteger maxThreads = ropePipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger tG = (threads < maxThreads) ? threads : maxThreads;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tG, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
  return 0;
}

int aljabr_metal_rmsnorm(void *out, const void *x, const void *weight,
                              int N, float eps, int addOne) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_rmsnorm(out, x, weight, N, eps, addOne);
  if (!g_elementwise_enabled || pipelines->rmsnorm == nil ||
      pipelines->rmsnorm.maxTotalThreadsPerThreadgroup < 256) {
    return aljabr_metal_cpu_rmsnorm(out, x, weight, N, eps, addOne);
  }

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    id<MTLBuffer> bufWeight =
        wrap_ptr((void *)weight, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil || bufWeight == nil) {
      return aljabr_metal_cpu_rmsnorm(out, x, weight, N, eps, addOne);
    }

    unsigned int n = (unsigned int)N;
    unsigned int add = addOne ? 1u : 0u;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (N % 4 == 0 && pipelines->rmsnorm_float4 != nil) {
      [enc setComputePipelineState:pipelines->rmsnorm_float4];
    } else {
      [enc setComputePipelineState:pipelines->rmsnorm];
    }
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufWeight offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc setBytes:&add length:sizeof(add) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_rmsnorm_rows(void *out, const void *x, const void *weight,
                                   int rows, int N, float eps, int addOne) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (rows <= 0 || N <= 0)
    return 0;
  if (!g_initialized || !g_elementwise_enabled ||
      pipelines->rmsnorm_rows == nil ||
      pipelines->rmsnorm_rows.maxTotalThreadsPerThreadgroup < 256) {
    return cpu_rmsnorm_rows(out, x, weight, rows, N, eps, addOne);
  }

  @autoreleasepool {
    size_t elements = (size_t)rows * (size_t)N;
    id<MTLBuffer> bufOut = wrap_ptr(out, elements * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, elements * sizeof(float));
    id<MTLBuffer> bufWeight =
        wrap_ptr((void *)weight, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil || bufWeight == nil) {
      return cpu_rmsnorm_rows(out, x, weight, rows, N, eps, addOne);
    }

    unsigned int n = (unsigned int)N;
    unsigned int add = addOne ? 1u : 0u;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->rmsnorm_rows];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufWeight offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc setBytes:&add length:sizeof(add) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_softmax(void *out, const void *x, int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return -5;
  if (!g_elementwise_enabled || pipelines->softmax == nil)
    return -5;

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil)
      return -5;

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->softmax];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_softmax_rows(void *out, const void *x, int rows, int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (rows <= 0 || N <= 0)
    return 0;
  if (!g_initialized || !g_elementwise_enabled ||
      pipelines->softmax_rows == nil)
    return -5;

  @autoreleasepool {
    size_t elements = (size_t)rows * (size_t)N;
    id<MTLBuffer> bufOut = wrap_ptr(out, elements * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, elements * sizeof(float));
    if (bufOut == nil || bufX == nil)
      return -5;

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->softmax_rows];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_silu_ffn(void *out, const void *gate, const void *up,
                               int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_silu_ffn(out, gate, up, N);
  if (!g_elementwise_enabled || pipelines->silu_ffn == nil)
    return aljabr_metal_cpu_silu_ffn(out, gate, up, N);

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufGate = wrap_ptr((void *)gate, (size_t)N * sizeof(float));
    id<MTLBuffer> bufUp = wrap_ptr((void *)up, (size_t)N * sizeof(float));
    if (bufOut == nil || bufGate == nil || bufUp == nil)
      return aljabr_metal_cpu_silu_ffn(out, gate, up, N);

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->silu_ffn];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufGate offset:0 atIndex:1];
    [enc setBuffer:bufUp offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger threads =
        MIN((NSUInteger)256, pipelines->silu_ffn.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_gelu_ffn(void *out, const void *gate, const void *up,
                               int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_gelu_ffn(out, gate, up, N);
  if (!g_elementwise_enabled || pipelines->gelu_ffn == nil)
    return aljabr_metal_cpu_gelu_ffn(out, gate, up, N);

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufGate = wrap_ptr((void *)gate, (size_t)N * sizeof(float));
    id<MTLBuffer> bufUp = wrap_ptr((void *)up, (size_t)N * sizeof(float));
    if (bufOut == nil || bufGate == nil || bufUp == nil)
      return aljabr_metal_cpu_gelu_ffn(out, gate, up, N);

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->gelu_ffn];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufGate offset:0 atIndex:1];
    [enc setBuffer:bufUp offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger threads =
        MIN((NSUInteger)256, pipelines->gelu_ffn.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

static int cpu_layernorm_rows(void *out, const void *x, const void *weight,
                              const void *bias, int rows, int N, float eps) {
  size_t row_bytes = (size_t)N * sizeof(float);
  for (int row = 0; row < rows; row++) {
    int rc = aljabr_metal_cpu_layernorm(
        (char *)out + ((size_t)row * row_bytes),
        (const char *)x + ((size_t)row * row_bytes), weight, bias, N, eps);
    if (rc != 0)
      return rc;
  }
  return 0;
}

int aljabr_metal_layernorm(void *out, const void *x, const void *weight,
                                const void *bias, int N, float eps) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_layernorm(out, x, weight, bias, N, eps);
  if (!g_elementwise_enabled || pipelines->layernorm == nil ||
      pipelines->layernorm.maxTotalThreadsPerThreadgroup < 256) {
    return aljabr_metal_cpu_layernorm(out, x, weight, bias, N, eps);
  }

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    id<MTLBuffer> bufWeight =
        weight ? wrap_ptr((void *)weight, (size_t)N * sizeof(float)) : nil;
    id<MTLBuffer> bufBias =
        bias ? wrap_ptr((void *)bias, (size_t)N * sizeof(float)) : nil;
    if (bufOut == nil || bufX == nil || (weight != NULL && bufWeight == nil) ||
        (bias != NULL && bufBias == nil)) {
      return aljabr_metal_cpu_layernorm(out, x, weight, bias, N, eps);
    }

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    // Use float4 kernel when N is a multiple of 4: reduces memory transactions
    // in both the reduction and write-back loops by 4x.
    BOOL useFloat4 = (N % 4 == 0) && (pipelines->layernorm_float4 != nil);
    [enc setComputePipelineState:useFloat4 ? pipelines->layernorm_float4
                                           : pipelines->layernorm];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    if (bufWeight != nil)
      [enc setBuffer:bufWeight offset:0 atIndex:2];
    if (bufBias != nil)
      [enc setBuffer:bufBias offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc setBytes:&eps length:sizeof(eps) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_layernorm_rows(void *out, const void *x,
                                     const void *weight, const void *bias,
                                     int rows, int N, float eps) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (rows <= 0 || N <= 0)
    return 0;
  if (!g_initialized || !g_elementwise_enabled ||
      pipelines->layernorm_rows == nil ||
      pipelines->layernorm_rows.maxTotalThreadsPerThreadgroup < 256) {
    return cpu_layernorm_rows(out, x, weight, bias, rows, N, eps);
  }

  @autoreleasepool {
    size_t elements = (size_t)rows * (size_t)N;
    id<MTLBuffer> bufOut = wrap_ptr(out, elements * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, elements * sizeof(float));
    id<MTLBuffer> bufWeight =
        weight ? wrap_ptr((void *)weight, (size_t)N * sizeof(float)) : nil;
    id<MTLBuffer> bufBias =
        bias ? wrap_ptr((void *)bias, (size_t)N * sizeof(float)) : nil;
    if (bufOut == nil || bufX == nil || (weight != NULL && bufWeight == nil) ||
        (bias != NULL && bufBias == nil)) {
      return cpu_layernorm_rows(out, x, weight, bias, rows, N, eps);
    }

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    // Rows variant: float4 kernel processes each row with 4-wide vector
    // loads/stores. Each threadgroup handles one row, so the dispatch shape
    // stays the same.
    BOOL useFloat4Rows = (N % 4 == 0) && (pipelines->layernorm_float4 != nil);
    [enc setComputePipelineState:useFloat4Rows ? pipelines->layernorm_float4
                                               : pipelines->layernorm_rows];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    if (bufWeight != nil)
      [enc setBuffer:bufWeight offset:0 atIndex:2];
    if (bufBias != nil)
      [enc setBuffer:bufBias offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc setBytes:&eps length:sizeof(eps) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_silu(void *out, const void *x, int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_silu(out, x, N);
  if (!g_elementwise_enabled || pipelines->silu == nil)
    return aljabr_metal_cpu_silu(out, x, N);

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil)
      return aljabr_metal_cpu_silu(out, x, N);

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->silu];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger threads =
        MIN((NSUInteger)256, pipelines->silu.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}

int aljabr_metal_gelu(void *out, const void *x, int N) {
  AljabrMetalPipelines *pipelines = aljabr_metal_pipelines();
  if (!g_initialized || N <= 0)
    return aljabr_metal_cpu_gelu(out, x, N);
  if (!g_elementwise_enabled || pipelines->gelu == nil)
    return aljabr_metal_cpu_gelu(out, x, N);

  @autoreleasepool {
    id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
    id<MTLBuffer> bufX = wrap_ptr((void *)x, (size_t)N * sizeof(float));
    if (bufOut == nil || bufX == nil)
      return aljabr_metal_cpu_gelu(out, x, N);

    unsigned int n = (unsigned int)N;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipelines->gelu];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger threads =
        MIN((NSUInteger)256, pipelines->gelu.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil ? 0 : -5;
  }
}
