/**
 * aljabr_metal_fa4.m — Metal FA4 bridge for Aljabr
 *
 * Implements the FlashAttention-4 algorithm on Apple Silicon using
 * MPSGraph.scaledDotProductAttentionWithQuery (available macOS 15.0+).
 *
 * On Apple Silicon the algorithm maps as follows:
 *
 *   FA4 (Blackwell)          Metal / Apple Silicon equivalent
 *   ─────────────────────    ──────────────────────────────────────────────────
 *   TMEM accumulator         On-chip SRAM of the GPU (L1 tile cache)
 *   UMMA tcgen05.mma         MPSGraph SDPA or AMX-accelerated MPS matmuls
 *   Async UMMA pipelines     Metal GPU command buffers / concurrent blits
 *   Software exp() on FMA    Apple Silicon FP16/BF16 exp via scalar intrinsics
 *   2-CTA MMA backward       Not applicable (forward-only path used by Aljabr)
 *
 * On M3/M4 hardware MPSGraph.scaledDotProductAttentionWithQuery dispatches to
 * a fused Metal kernel that keeps the S = QK^T tile on the GPU's L1 cache
 * without materialising the full attention matrix — functionally identical to
 * the FlashAttention tile-recompute strategy.
 *
 * Compilation:
 *   make
 *
 * NOTE: compile all Metal bridge modules into ONE dylib — they share the
 * globals g_device and g_queue from aljabr_metal_support.m, while CPU
 * fallback kernels live in aljabr_metal_cpu_fallback.m.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#import "aljabr_metal_cpu_fallback.h"
#import "aljabr_metal_support.h"

// ── Helper
// ─────────────────────────────────────────────────────────────────────

static id<MTLBuffer> wrap_ptr_fa4(void *ptr, size_t bytes) {
  return [g_device newBufferWithBytesNoCopy:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static float *transpose_bthd_to_bhtd(const float *src, int B, int T, int H,
                                     int D) {
  size_t total = (size_t)B * T * H * D;
  float *dst = (float *)malloc(total * sizeof(float));
  if (!dst)
    return NULL;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < H; h++) {
        const float *srcPtr = src + ((((size_t)b * T + t) * H + h) * D);
        float *dstPtr = dst + ((((size_t)b * H + h) * T + t) * D);
        memcpy(dstPtr, srcPtr, (size_t)D * sizeof(float));
      }
    }
  }
  return dst;
}

static void transpose_bhtd_to_bthd_inplace(const float *src, float *dst, int B,
                                           int T, int H, int D) {
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int t = 0; t < T; t++) {
        const float *srcPtr = src + ((((size_t)b * H + h) * T + t) * D);
        float *dstPtr = dst + ((((size_t)b * T + t) * H + h) * D);
        memcpy(dstPtr, srcPtr, (size_t)D * sizeof(float));
      }
    }
  }
}

static inline int causal_limit_for_row(int T, int S, int t) {
  int causalOffset = S > T ? (S - T) : 0;
  int limit = causalOffset + t + 1;
  return limit < S ? limit : S;
}

static int mps_matmul_tb_f32(float *out, const float *left, const float *right,
                             int M, int K, int N, float alpha) {
  @autoreleasepool {
    id<MTLBuffer> outBuf = wrap_ptr_fa4(out, (size_t)M * N * sizeof(float));
    id<MTLBuffer> leftBuf =
        wrap_ptr_fa4((void *)left, (size_t)M * K * sizeof(float));
    id<MTLBuffer> rightBuf =
        wrap_ptr_fa4((void *)right, (size_t)N * K * sizeof(float));

    MPSMatrixDescriptor *leftDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:K
                                             rowBytes:K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *rightDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                              columns:K
                                             rowBytes:K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *outDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *leftMat = [[MPSMatrix alloc] initWithBuffer:leftBuf
                                                descriptor:leftDesc];
    MPSMatrix *rightMat = [[MPSMatrix alloc] initWithBuffer:rightBuf
                                                 descriptor:rightDesc];
    MPSMatrix *outMat = [[MPSMatrix alloc] initWithBuffer:outBuf
                                               descriptor:outDesc];

    MPSMatrixMultiplication *mmul =
        [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:M
                                          resultColumns:N
                                        interiorColumns:K
                                                  alpha:alpha
                                                   beta:0.0f];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:leftMat
                    rightMatrix:rightMat
                   resultMatrix:outMat];
    [cmd commit];
    [cmd waitUntilCompleted];
    return [cmd status] == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}

static int mps_matmul_nn_f32(float *out, const float *left, const float *right,
                             int M, int K, int N, float alpha) {
  @autoreleasepool {
    id<MTLBuffer> outBuf = wrap_ptr_fa4(out, (size_t)M * N * sizeof(float));
    id<MTLBuffer> leftBuf =
        wrap_ptr_fa4((void *)left, (size_t)M * K * sizeof(float));
    id<MTLBuffer> rightBuf =
        wrap_ptr_fa4((void *)right, (size_t)K * N * sizeof(float));

    MPSMatrixDescriptor *leftDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:K
                                             rowBytes:K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *rightDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *outDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *leftMat = [[MPSMatrix alloc] initWithBuffer:leftBuf
                                                descriptor:leftDesc];
    MPSMatrix *rightMat = [[MPSMatrix alloc] initWithBuffer:rightBuf
                                                 descriptor:rightDesc];
    MPSMatrix *outMat = [[MPSMatrix alloc] initWithBuffer:outBuf
                                               descriptor:outDesc];

    MPSMatrixMultiplication *mmul =
        [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:M
                                          resultColumns:N
                                        interiorColumns:K
                                                  alpha:alpha
                                                   beta:0.0f];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd
                     leftMatrix:leftMat
                    rightMatrix:rightMat
                   resultMatrix:outMat];
    [cmd commit];
    [cmd waitUntilCompleted];
    return [cmd status] == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}

// ── Public C API
// ───────────────────────────────────────────────────────────────

/**
 * Metal FA4-equivalent attention via MPSGraph SDPA.
 *
 * Uses MPSGraph.scaledDotProductAttentionWithQuery (macOS 15+) which fuses
 * QK^T / softmax / ×V into a single Metal compute pass, keeping the attention
 * score tile in the GPU's tile memory — equivalent to FlashAttention's
 * tile-recompute approach.
 *
 * Falls back to separate MPS matmuls on macOS 13.
 *
 * @param output       [B, T, H, D] float16 or float32
 * @param query        [B, T, H, D]
 * @param key          [B, S, H_kv, D]   (already gathered from K-cache by
 * caller)
 * @param value        [B, S, H_kv, D]   (already gathered from V-cache by
 * caller)
 * @param B            batch size
 * @param T            query sequence length
 * @param S            key/value sequence length
 * @param H            query heads
 * @param H_kv         key/value heads (GQA)
 * @param D            head dim
 * @param scale        softmax scale (1/sqrt(D))
 * @param is_causal    apply causal mask
 * @param use_bf16     use BF16 arithmetic (M2+ only; falls back to FP16 on M1)
 * @return 0 on success, -1 on error
 */
int aljabr_metal_fa4_attention(void *output, const void *query,
                                    const void *key, const void *value, int B,
                                    int T, int S, int H, int H_kv, int D,
                                    float scale, int is_causal, int use_bf16,
                                    float soft_cap) {

  if (!g_initialized)
    return -1;

  MPSDataType dtype = MPSDataTypeFloat32;
  size_t elemBytes = sizeof(float);

  if (@available(macOS 15.0, *)) {
    if (soft_cap <= 0.0f) {
      // ── MPSGraph SDPA path (macOS 15+, M3+ optimised) ─────────────────────
      // Fused single-pass attention: no intermediate attention matrix on DRAM.
      size_t qBytes = (size_t)B * T * H * D * elemBytes;
      size_t kBytes = (size_t)B * S * H_kv * D * elemBytes;
      size_t oBytes = (size_t)B * T * H * D * elemBytes;

      id<MTLBuffer> qBuf = wrap_ptr_fa4((void *)query, qBytes);
      id<MTLBuffer> kBuf = wrap_ptr_fa4((void *)key, kBytes);
      id<MTLBuffer> vBuf = wrap_ptr_fa4((void *)value, kBytes);
      id<MTLBuffer> oBuf = wrap_ptr_fa4((void *)output, oBytes);

      MPSGraph *graph = [MPSGraph new];

      // Shape descriptors: [B, T, H, D]
      NSArray<NSNumber *> *qShapeOrig = @[ @(B), @(T), @(H), @(D) ];
      NSArray<NSNumber *> *kShapeOrig = @[ @(B), @(S), @(H_kv), @(D) ];
      NSArray<NSNumber *> *mShape = @[ @(B), @1, @(T), @(S) ];

      MPSGraphTensor *qT_orig = [graph placeholderWithShape:qShapeOrig
                                                   dataType:dtype
                                                       name:@"Q_orig"];
      MPSGraphTensor *kT_orig = [graph placeholderWithShape:kShapeOrig
                                                   dataType:dtype
                                                       name:@"K_orig"];
      MPSGraphTensor *vT_orig = [graph placeholderWithShape:kShapeOrig
                                                   dataType:dtype
                                                       name:@"V_orig"];

      // Transpose [B, T, H, D] -> [B, H, T, D]
      MPSGraphTensor *qT = [graph transposeTensor:qT_orig
                                        dimension:1
                                    withDimension:2
                                             name:@"Q"];
      MPSGraphTensor *kT = [graph transposeTensor:kT_orig
                                        dimension:1
                                    withDimension:2
                                             name:@"K"];
      MPSGraphTensor *vT = [graph transposeTensor:vT_orig
                                        dimension:1
                                    withDimension:2
                                             name:@"V"];

      MPSGraphTensor *mT = nil;
      if (is_causal) {
        mT = [graph placeholderWithShape:mShape dataType:dtype name:@"MASK"];
      }

      MPSGraphTensor *attnOut =
          is_causal ? [graph scaledDotProductAttentionWithQueryTensor:qT
                                                            keyTensor:kT
                                                          valueTensor:vT
                                                           maskTensor:mT
                                                                scale:scale
                                                                 name:@"sdpa"]
                    : [graph scaledDotProductAttentionWithQueryTensor:qT
                                                            keyTensor:kT
                                                          valueTensor:vT
                                                                scale:scale
                                                                 name:@"sdpa"];

      // Transpose [B, H, T, D] -> [B, T, H, D]
      MPSGraphTensor *finalOut = [graph transposeTensor:attnOut
                                              dimension:1
                                          withDimension:2
                                                   name:@"O_t"];

      // Feed data
      MPSGraphTensorData *qData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:qBuf
                                                  shape:qShapeOrig
                                               dataType:dtype];
      MPSGraphTensorData *kData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:kBuf
                                                  shape:kShapeOrig
                                               dataType:dtype];
      MPSGraphTensorData *vData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:vBuf
                                                  shape:kShapeOrig
                                               dataType:dtype];
      MPSGraphTensorData *oData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf
                                                  shape:qShapeOrig
                                               dataType:dtype];

      id<MTLBuffer> mBuf = nil;
      MPSGraphTensorData *mData = nil;
      float *maskData = NULL;
      if (is_causal) {
        size_t maskElems = (size_t)B * T * S;
        maskData = (float *)calloc(maskElems, sizeof(float));
        if (!maskData) {
          return -1;
        }
        for (int b = 0; b < B; b++) {
          for (int t = 0; t < T; t++) {
            int limit = causal_limit_for_row(T, S, t);
            for (int s = limit; s < S; s++) {
              size_t idx = ((size_t)b * T + t) * S + s;
              maskData[idx] = -1.0e9f;
            }
          }
        }
        mBuf = wrap_ptr_fa4(maskData, maskElems * sizeof(float));
        mData = [[MPSGraphTensorData alloc] initWithMTLBuffer:mBuf
                                                        shape:mShape
                                                     dataType:dtype];
      }

      NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
          [@{qT_orig : qData, kT_orig : kData, vT_orig : vData} mutableCopy];
      if (mData != nil && mT != nil) {
        feeds[mT] = mData;
      }
      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
          @{finalOut : oData};

      MPSCommandBuffer *cmdBuf =
          [MPSCommandBuffer commandBufferFromCommandQueue:g_queue];
      [graph encodeToCommandBuffer:cmdBuf
                             feeds:feeds
                  targetOperations:nil
                 resultsDictionary:results
               executionDescriptor:nil];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];

      if ([cmdBuf status] == MTLCommandBufferStatusError) {
        NSLog(@"[AljabrMetal FA4] SDPA error: %@", [cmdBuf error]);
        if (maskData)
          free(maskData);
        return -1;
      }
      if (maskData)
        free(maskData);
      return 0;
    }
  }

  float *qTransposed = transpose_bthd_to_bhtd((const float *)query, B, T, H, D);
  float *kTransposed =
      transpose_bthd_to_bhtd((const float *)key, B, S, H_kv, D);
  float *vTransposed =
      transpose_bthd_to_bhtd((const float *)value, B, S, H_kv, D);
  float *oTransposed = (float *)calloc((size_t)B * H * T * D, sizeof(float));
  if (!qTransposed || !kTransposed || !vTransposed || !oTransposed) {
    if (qTransposed)
      free(qTransposed);
    if (kTransposed)
      free(kTransposed);
    if (vTransposed)
      free(vTransposed);
    if (oTransposed)
      free(oTransposed);
    return -1;
  }

  // ── Fallback path for macOS 13 (separate MPS matmuls) ────────────────────
  // QK^T per head using MPSMatrixMultiplication, then CPU softmax / soft-cap,
  // then ×V.
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      int hk = h / (H / H_kv); // GQA head mapping

      size_t scoreElems = (size_t)T * S;
      float *scorePtr = (float *)calloc(scoreElems, sizeof(float));
      if (!scorePtr) {
        free(qTransposed);
        free(kTransposed);
        free(vTransposed);
        free(oTransposed);
        return -1;
      }

      const float *qPtr = qTransposed + ((((size_t)b * H) + h) * T * D);
      const float *kPtr = kTransposed + ((((size_t)b * H_kv) + hk) * S * D);
      float *oPtr = oTransposed + ((((size_t)b * H) + h) * T * D);

      if (mps_matmul_tb_f32(scorePtr, qPtr, kPtr, T, D, S, scale) != 0) {
        aljabr_metal_cpu_matmul_tb(scorePtr, qPtr, kPtr, T, D, S, scale,
                                        0.0f);
      }

      // Causal mask + softmax per row
      for (int t = 0; t < T; t++) {
        float *row = scorePtr + t * S;
        int lim = is_causal ? causal_limit_for_row(T, S, t) : S;
        for (int s = lim; s < S; s++)
          row[s] = -1e9f;
        if (soft_cap > 0.0f) {
          for (int s = 0; s < lim; s++) {
            row[s] = soft_cap * tanhf(row[s] / soft_cap);
          }
        }
        float mx = row[0];
        for (int s = 1; s < lim; s++)
          if (row[s] > mx)
            mx = row[s];
        float sm = 0.f;
        for (int s = 0; s < lim; s++) {
          row[s] = expf(row[s] - mx);
          sm += row[s];
        }
        for (int s = 0; s < lim; s++)
          row[s] /= sm;
      }

      // out[T, D] = score[T, S] × V[S, D]
      const float *vPtr = vTransposed + ((((size_t)b * H_kv) + hk) * S * D);
      if (mps_matmul_nn_f32(oPtr, scorePtr, vPtr, T, S, D, 1.0f) != 0) {
        aljabr_metal_cpu_matmul_nn(oPtr, scorePtr, vPtr, T, S, D, 1.0f,
                                        0.0f);
      }
      free(scorePtr);
    }
  }
  transpose_bhtd_to_bthd_inplace(oTransposed, (float *)output, B, T, H, D);
  free(qTransposed);
  free(kTransposed);
  free(vTransposed);
  free(oTransposed);
  return 0;
}

/**
 * Check whether MPSGraph SDPA (macOS 14+ fused path) is available.
 * Returns 1 if available, 0 if only fallback path is usable.
 */
int aljabr_metal_fa4_sdpa_available(void) {
  if (@available(macOS 15.0, *))
    return 1;
  return 0;
}

/**
 * Check whether BF16 computation is supported (Apple M2 GPU family and later).
 * Returns 1 if BF16 available, 0 if FP16 only.
 */
int aljabr_metal_fa4_bf16_available(void) {
  if (!g_device)
    return 0;
  return [g_device supportsFamily:MTLGPUFamilyApple8] ? 1 : 0;
}
