/**
 * aljabr_metal_attention.m — paged attention bridge implementation.
 */

#import "aljabr_metal_attention.h"
#import "aljabr_metal_buffers.h"
#import "aljabr_metal_cpu_fallback.h"
#import "aljabr_metal_pipelines.h"
#import "aljabr_metal_scratch.h"
#import "aljabr_metal_support.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int aljabr_metal_decode_attention_gpu(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int H, int H_kv,
    int D, int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap) {
  id<MTLComputePipelineState> pipeline =
      aljabr_metal_pipelines()->decode_attention;
  if (!g_initialized || pipeline == nil)
    return -3;
  if (aljabr_metal_env_truthy(
          "ALJABR_METAL_DISABLE_DECODE_ATTENTION_KERNEL"))
    return -9;
  if (B <= 0 || H <= 0 || H_kv <= 0 || D <= 0 || block_size <= 0 ||
      max_blocks <= 0)
    return -2;
  if (H % H_kv != 0)
    return -2;
  if (D > 256)
    return -8;

  int max_context = aljabr_metal_env_int_or_default(
      "ALJABR_METAL_DECODE_ATTENTION_MAX_CONTEXT", 4096);
  if (max_context <= 0 || max_context > 4096) {
    max_context = 4096;
  }

  int max_phys = -1;
  for (int b = 0; b < B; b++) {
    int ctx_len = context_lens[b];
    if (ctx_len <= 0)
      continue;
    int effective_query_start_pos = query_start_pos;
    if (is_causal && sliding_window <= 0 && query_start_pos == 0) {
      effective_query_start_pos = ctx_len - 1;
    }
    int max_pos = is_causal ? effective_query_start_pos : (ctx_len - 1);
    if (max_pos >= ctx_len)
      max_pos = ctx_len - 1;
    int min_pos = sliding_window > 0
                      ? (effective_query_start_pos - sliding_window + 1)
                      : 0;
    if (min_pos < 0)
      min_pos = 0;
    int valid = max_pos >= min_pos ? (max_pos - min_pos + 1) : 0;
    if (valid <= 0)
      continue;
    if (valid > max_context)
      return -8;

    int blocks = (ctx_len + block_size - 1) / block_size;
    if (blocks > max_blocks)
      return -2;
    for (int blk = 0; blk < blocks; blk++) {
      int phys = block_table[b * max_blocks + blk];
      if (phys < 0)
        return -2;
      if (phys > max_phys)
        max_phys = phys;
    }
  }
  if (max_phys < 0) {
    memset(out, 0, (size_t)B * H * D * sizeof(float));
    return 0;
  }

  @autoreleasepool {
    size_t q_bytes = (size_t)B * H * D * sizeof(float);
    size_t out_bytes = q_bytes;
    size_t kv_bytes =
        (size_t)(max_phys + 1) * H_kv * block_size * D * sizeof(float);
    size_t block_table_bytes = (size_t)B * max_blocks * sizeof(int);
    size_t context_lens_bytes = (size_t)B * sizeof(int);

    id<MTLBuffer> bufOut = wrap_ptr(out, out_bytes);
    id<MTLBuffer> bufQ = wrap_ptr((void *)Q, q_bytes);
    id<MTLBuffer> bufK = wrap_ptr((void *)K_cache, kv_bytes);
    id<MTLBuffer> bufV = wrap_ptr((void *)V_cache, kv_bytes);
    id<MTLBuffer> bufBlockTable =
        wrap_ptr((void *)block_table, block_table_bytes);
    id<MTLBuffer> bufContextLens =
        wrap_ptr((void *)context_lens, context_lens_bytes);
    if (bufOut == nil || bufQ == nil || bufK == nil || bufV == nil ||
        bufBlockTable == nil || bufContextLens == nil) {
      return -4;
    }

    uint32_t h = (uint32_t)H;
    uint32_t hkv = (uint32_t)H_kv;
    uint32_t d = (uint32_t)D;
    uint32_t bs = (uint32_t)block_size;
    uint32_t mb = (uint32_t)max_blocks;
    uint32_t causal = is_causal ? 1u : 0u;
    int32_t qpos = (int32_t)query_start_pos;
    int32_t window = (int32_t)sliding_window;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufQ offset:0 atIndex:1];
    [enc setBuffer:bufK offset:0 atIndex:2];
    [enc setBuffer:bufV offset:0 atIndex:3];
    [enc setBuffer:bufBlockTable offset:0 atIndex:4];
    [enc setBuffer:bufContextLens offset:0 atIndex:5];
    [enc setBytes:&h length:sizeof(h) atIndex:6];
    [enc setBytes:&hkv length:sizeof(hkv) atIndex:7];
    [enc setBytes:&d length:sizeof(d) atIndex:8];
    [enc setBytes:&bs length:sizeof(bs) atIndex:9];
    [enc setBytes:&mb length:sizeof(mb) atIndex:10];
    [enc setBytes:&scale length:sizeof(scale) atIndex:11];
    [enc setBytes:&causal length:sizeof(causal) atIndex:12];
    [enc setBytes:&qpos length:sizeof(qpos) atIndex:13];
    [enc setBytes:&window length:sizeof(window) atIndex:14];
    [enc setBytes:&soft_cap length:sizeof(soft_cap) atIndex:15];

    if (pipeline.maxTotalThreadsPerThreadgroup < 256) {
      return -3;
    }
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B * (NSUInteger)H, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
  }
}

static inline void resolve_attention_bounds(int ctx_len, int is_causal,
                                            int query_start_pos, int query_idx,
                                            int sliding_window, int *min_pos,
                                            int *max_pos) {
  int abs_query_pos = query_start_pos + query_idx;
  int upper = is_causal ? abs_query_pos : (ctx_len - 1);
  if (upper >= ctx_len)
    upper = ctx_len - 1;
  int lower = sliding_window > 0 ? (abs_query_pos - sliding_window + 1) : 0;
  if (lower < 0)
    lower = 0;
  *min_pos = lower;
  *max_pos = upper;
}

static int aljabr_metal_attention_impl(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int T, int H,
    int H_kv, int D, int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap) {

  if (!g_initialized)
    return -1;
  if (H_kv <= 0 || H % H_kv != 0)
    return -1;
  if (B <= 0 || T <= 0 || H <= 0 || D <= 0 || block_size <= 0 ||
      max_blocks <= 0)
    return -2;

  int gqa_group = H / H_kv;
  if (T == 1) {
    int gpu_rc = aljabr_metal_decode_attention_gpu(
        out, Q, K_cache, V_cache, block_table, context_lens, B, H, H_kv, D,
        block_size, max_blocks, scale, is_causal, query_start_pos,
        sliding_window, soft_cap);
    if (gpu_rc == 0) {
      return 0;
    }
    if (gpu_rc != -9 && aljabr_metal_env_truthy(
                            "ALJABR_METAL_DECODE_ATTENTION_DEBUG")) {
      fprintf(stderr,
              "[aljabr-metal] decode attention GPU kernel unavailable "
              "rc=%d; falling back to CPU attention bridge\n",
              gpu_rc);
    }
  }
  BOOL use_decode_sgemv =
      (T == 1) && !aljabr_metal_env_truthy(
                      "ALJABR_METAL_DISABLE_ATTENTION_SGEMV");

  for (int b = 0; b < B; b++) {
    int ctx_len = context_lens[b];
    float *batchOut = (float *)out + ((size_t)b * T * H * D);
    if (ctx_len <= 0) {
      memset(batchOut, 0, (size_t)T * H * D * sizeof(float));
      continue;
    }
    int num_blocks = (ctx_len + block_size - 1) / block_size;
    if (num_blocks > max_blocks)
      return -2;
    size_t kv_elements = (size_t)ctx_len * H_kv * D;
    size_t score_elements = (size_t)H * T * ctx_len;
    float *kPtr = NULL;
    float *vPtr = NULL;
    float *scoreBuf = NULL;
    size_t scratch_elements =
        score_elements > kv_elements ? score_elements : kv_elements;
    if (!aljabr_metal_ensure_attention_scratch(scratch_elements, &kPtr,
                                                    &vPtr, &scoreBuf)) {
      return -4;
    }
    const float *kcache = (const float *)K_cache;
    const float *vcache = (const float *)V_cache;

    int lda = D;
    if (block_table != NULL) {
      for (int blk = 0; blk < num_blocks; blk++) {
        int phys = block_table[b * max_blocks + blk];
        if (phys < 0)
          return -2;
        int tokens_in_blk =
            (blk == num_blocks - 1) ? (ctx_len - blk * block_size) : block_size;

        for (int h = 0; h < H_kv; h++) {
          const float *src_k =
              kcache + ((size_t)phys * H_kv + h) * block_size * D;
          const float *src_v =
              vcache + ((size_t)phys * H_kv + h) * block_size * D;
          float *dst_k = kPtr + ((size_t)h * ctx_len + blk * block_size) * D;
          float *dst_v = vPtr + ((size_t)h * ctx_len + blk * block_size) * D;
          memcpy(dst_k, src_k, (size_t)tokens_in_blk * D * sizeof(float));
          memcpy(dst_v, src_v, (size_t)tokens_in_blk * D * sizeof(float));
        }
      }
    } else {
      lda = H_kv * D;
    }

    const float *qPtr = (const float *)Q + (size_t)b * T * H * D;
    for (int h = 0; h < H; h++) {
      int kv_h = h / gqa_group;
      for (int t = 0; t < T; t++) {
        const float *qh = qPtr + (t * H + h) * D;
        float *row = scoreBuf + (h * T + t) * ctx_len;
        int min_pos = 0;
        int max_pos = ctx_len - 1;
        int effective_query_start_pos = query_start_pos;
        if (is_causal && sliding_window <= 0 && query_start_pos == 0 &&
            ctx_len >= T) {
          effective_query_start_pos = ctx_len - T;
        }
        resolve_attention_bounds(ctx_len, is_causal, effective_query_start_pos,
                                 t, sliding_window, &min_pos, &max_pos);

        float *oh = batchOut + (t * H + h) * D;
        memset(oh, 0, (size_t)D * sizeof(float));
        if (max_pos < min_pos) {
          continue;
        }

        if (use_decode_sgemv) {
          int valid_len = max_pos - min_pos + 1;
          const float *kBase;
          if (block_table != NULL) {
            kBase = kPtr + ((size_t)kv_h * ctx_len + min_pos) * D;
          } else {
            kBase =
                kcache + (((size_t)b * ctx_len + min_pos) * H_kv + kv_h) * D;
          }
          aljabr_metal_cpu_matvec_rows(row + min_pos, kBase, qh, valid_len,
                                            D, lda, scale, 0.0f);

          float mx = -1e30f;
          for (int s = min_pos; s <= max_pos; s++) {
            float score = row[s];
            if (soft_cap > 0.0f) {
              score = soft_cap * tanhf(score / soft_cap);
              row[s] = score;
            }
            if (score > mx)
              mx = score;
          }

          float sum = 0.0f;
          for (int s = min_pos; s <= max_pos; s++) {
            row[s] = expf(row[s] - mx);
            sum += row[s];
          }
          float inv_sum = 1.0f / (sum + 1e-9f);
          for (int s = min_pos; s <= max_pos; s++) {
            row[s] *= inv_sum;
          }

          const float *vBase;
          if (block_table != NULL) {
            vBase = vPtr + ((size_t)kv_h * ctx_len + min_pos) * D;
          } else {
            vBase =
                vcache + (((size_t)b * ctx_len + min_pos) * H_kv + kv_h) * D;
          }
          aljabr_metal_cpu_matvec_cols(oh, vBase, row + min_pos, valid_len,
                                            D, lda, 1.0f, 0.0f);
          continue;
        }

        float mx = -1e30f;
        for (int s = min_pos; s <= max_pos; s++) {
          const float *kh;
          if (block_table != NULL) {
            kh = kPtr + ((size_t)kv_h * ctx_len + s) * D;
          } else {
            kh = kcache + (((size_t)b * ctx_len + s) * H_kv + kv_h) * D;
          }
          float dot = 0.0f;
          for (int d = 0; d < D; d++)
            dot += qh[d] * kh[d];

          float score = dot * scale;
          if (soft_cap > 0.0f) {
            score = soft_cap * tanhf(score / soft_cap);
          }
          row[s] = score;
          if (score > mx)
            mx = score;
        }

        float sum = 0.0f;
        for (int s = min_pos; s <= max_pos; s++) {
          row[s] = expf(row[s] - mx);
          sum += row[s];
        }
        float inv_sum = 1.0f / (sum + 1e-9f);
        for (int s = min_pos; s <= max_pos; s++) {
          row[s] *= inv_sum;
        }

        for (int s = min_pos; s <= max_pos; s++) {
          const float *vh;
          if (block_table != NULL) {
            vh = vPtr + ((size_t)kv_h * ctx_len + s) * D;
          } else {
            vh = vcache + (((size_t)b * ctx_len + s) * H_kv + kv_h) * D;
          }
          float weight = row[s];
          for (int d = 0; d < D; d++)
            oh[d] += weight * vh[d];
        }
      }
    }
  }
  return 0;
}

int aljabr_metal_attention(void *out, const void *Q, const void *K_cache,
                                const void *V_cache, const int *block_table,
                                const int *context_lens, int B, int T, int H,
                                int D, int block_size, int max_blocks,
                                float scale, int is_causal, float soft_cap) {
  return aljabr_metal_attention_impl(
      out, Q, K_cache, V_cache, block_table, context_lens, B, T, H, H, D,
      block_size, max_blocks, scale, is_causal, 0, 0, soft_cap);
}

int aljabr_metal_attention_windowed(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int T, int H, int D,
    int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap) {
  return aljabr_metal_attention_impl(
      out, Q, K_cache, V_cache, block_table, context_lens, B, T, H, H, D,
      block_size, max_blocks, scale, is_causal, query_start_pos, sliding_window,
      soft_cap);
}

int aljabr_metal_attention_gqa(void *out, const void *Q,
                                    const void *K_cache, const void *V_cache,
                                    const int *block_table,
                                    const int *context_lens, int B, int T,
                                    int H, int H_kv, int D, int block_size,
                                    int max_blocks, float scale, int is_causal,
                                    float soft_cap) {
  return aljabr_metal_attention_impl(
      out, Q, K_cache, V_cache, block_table, context_lens, B, T, H, H_kv, D,
      block_size, max_blocks, scale, is_causal, 0, 0, soft_cap);
}

int aljabr_metal_attention_gqa_windowed(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int T, int H,
    int H_kv, int D, int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap) {
  return aljabr_metal_attention_impl(
      out, Q, K_cache, V_cache, block_table, context_lens, B, T, H, H_kv, D,
      block_size, max_blocks, scale, is_causal, query_start_pos, sliding_window,
      soft_cap);
}

int aljabr_metal_flash_attention(void *out, const void *q, const void *k,
                                      const void *v, int B, int seq_len,
                                      int num_heads, int head_dim) {
  id<MTLComputePipelineState> pipeline =
      aljabr_metal_pipelines()->flash_attention;
  if (!g_initialized || pipeline == nil)
    return -3;

  @autoreleasepool {
    size_t size = (size_t)B * seq_len * num_heads * head_dim * sizeof(float);

    id<MTLBuffer> bufOut = wrap_ptr(out, size);
    id<MTLBuffer> bufQ = wrap_ptr((void *)q, size);
    id<MTLBuffer> bufK = wrap_ptr((void *)k, size);
    id<MTLBuffer> bufV = wrap_ptr((void *)v, size);

    if (bufOut == nil || bufQ == nil || bufK == nil || bufV == nil)
      return -4;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBuffer:bufQ offset:0 atIndex:1];
    [enc setBuffer:bufK offset:0 atIndex:2];
    [enc setBuffer:bufV offset:0 atIndex:3];

    uint32_t slen = (uint32_t)seq_len;
    uint32_t hdim = (uint32_t)head_dim;
    [enc setBytes:&slen length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:5];

    uint32_t grid_x = (slen + 31) / 32;
    MTLSize grid = MTLSizeMake(grid_x, num_heads, B);
    MTLSize threads = MTLSizeMake(32, 1, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
  return 0;
}
