/**
 * aljabr_metal_attention.h — paged attention bridge entry points.
 */

#ifndef ALJABR_METAL_ATTENTION_H
#define ALJABR_METAL_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

int aljabr_metal_attention(void *out, const void *Q, const void *K_cache,
                                const void *V_cache, const int *block_table,
                                const int *context_lens, int B, int T, int H,
                                int D, int block_size, int max_blocks,
                                float scale, int is_causal, float soft_cap);

int aljabr_metal_attention_windowed(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int T, int H, int D,
    int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap);

int aljabr_metal_attention_gqa(void *out, const void *Q,
                                    const void *K_cache, const void *V_cache,
                                    const int *block_table,
                                    const int *context_lens, int B, int T,
                                    int H, int H_kv, int D, int block_size,
                                    int max_blocks, float scale, int is_causal,
                                    float soft_cap);

int aljabr_metal_attention_gqa_windowed(
    void *out, const void *Q, const void *K_cache, const void *V_cache,
    const int *block_table, const int *context_lens, int B, int T, int H,
    int H_kv, int D, int block_size, int max_blocks, float scale, int is_causal,
    int query_start_pos, int sliding_window, float soft_cap);

int aljabr_metal_flash_attention(void *out, const void *q, const void *k,
                                      const void *v, int B, int seq_len,
                                      int num_heads, int head_dim);

/* Decode one query against contiguous sequence-major borrowed MTLBuffer K/V. */
int aljabr_metal_sdpa_borrowed_kv(
    void *out, const void *q, void *k_buffer, void *v_buffer,
    int num_heads, int kv_heads, int head_dim, int sequence_stride,
    int position, int start_position, float scale, float soft_cap);

#ifdef __cplusplus
}
#endif

#endif
