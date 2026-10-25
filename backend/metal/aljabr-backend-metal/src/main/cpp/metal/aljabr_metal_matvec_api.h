/**
 * aljabr_metal_matvec_api.h — exported Metal matvec C API.
 */

#ifndef ALJABR_METAL_MATVEC_API_H
#define ALJABR_METAL_MATVEC_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply logit soft-capping in-place on the GPU: x[i] = cap * tanh(x[i] / cap).
 *
 * @param logits  float* to the logit buffer (in-place, caller-managed memory)
 * @param N       number of logit elements (e.g. vocab size = 262144)
 * @param cap     soft-cap scalar (e.g. 30.0f for Gemma-4)
 * @return 0 on success, negative on failure
 */
int aljabr_metal_softcap(void *logits, int N, float cap);

int aljabr_metal_matvec_tb_half(void *C, const void *A, const void *B,
                                     int K, int N);

int aljabr_metal_matvec_tb_half_mps(void *C, const void *A, const void *B,
                                         int K, int N);

int aljabr_metal_matvec_t_half(void *C, const void *A, const void *B,
                                    int K, int N);

int aljabr_metal_matvec_tb_bf16(void *C, const void *A, const void *B,
                                     int K, int N);

int aljabr_metal_matvec_tb_int4(void *C, const void *A,
                                     const void *B_packed, const void *scales,
                                     int K, int N, int blockSize);

// GGML-format quantized kernels (block formats from GGUF)
int aljabr_metal_matvec_tb_q4_k(void *C, const void *A, const void *B,
                                     int K, int N);
int aljabr_metal_matvec_tb_q4_k_softcap(void *C, const void *A, const void *B,
                                             int K, int N, float cap);
int aljabr_metal_matvec_tb_q6_k(void *C, const void *A, const void *B,
                                     int K, int N);
int aljabr_metal_matvec_tb_q6_k_softcap(void *C, const void *A, const void *B,
                                             int K, int N, float cap);
int aljabr_metal_matvec_tb_q8_0(void *C, const void *A, const void *B,
                                     int K, int N);

int aljabr_metal_matvec_tb_nf4(void *C, const void *A,
                                    const void *B_packed, const void *absmax,
                                    int K, int N, int blockSize);

int aljabr_metal_matvec_tb_half_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N);

int aljabr_metal_matvec_tb_bf16_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N);

int aljabr_metal_matvec_tb_half_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2);

int aljabr_metal_matvec_tb_bf16_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2);

int aljabr_metal_matvec_tb_pair(void *C0, void *C1, const void *A,
                                const void *B0, const void *B1, int K,
                                int N0, int N1, int type0, int type1);

int aljabr_metal_matvec_tb_triple(void *C0, void *C1, void *C2,
                                  const void *A, const void *B0,
                                  const void *B1, const void *B2,
                                  int K, int N0, int N1, int N2,
                                  int type0, int type1, int type2);

int aljabr_metal_matvec_tb_batch(void *C, const void *A, const void *B,
                                 int K, int N, int type, int batchSize);

int aljabr_metal_matvec_tb_triple_batch(void *C0, void *C1, void *C2,
                                        const void *A, const void *B0,
                                        const void *B1, const void *B2,
                                        int K, int N0, int N1, int N2,
                                        int type0, int type1, int type2,
                                        int batchSize);

int aljabr_metal_ffn_gated_fused(void *outDown, const void *inA,
                                const void *gateW, const void *upW, const void *downW,
                                int dim, int ffnDim,
                                int typeGate, int typeUp, int typeDown,
                                int activationKind);

int aljabr_metal_ffn_gated_fused_batch(void *outDown, const void *inA,
                                      const void *gateW, const void *upW, const void *downW,
                                      int dim, int ffnDim,
                                      int typeGate, int typeUp, int typeDown,
                                      int activationKind,
                                      int batchSize);

/**
 * Fused wO-projection + post-attn norm + attn residual + ffn-norm + FFN + post-ffn norm + ffn residual.
 *
 * Runs ALL of the following in a SINGLE Metal command buffer with in-encoder barriers,
 * eliminating 2 waitUntilCompleted roundtrips per layer (one for wO, one for FFN).
 *
 * GPU sequence:
 *   xb = wO(attnFull)                      [wO matvec, Q4_K/Q6_K/etc]
 *   if postAttnW: xb = rmsNorm(xb, postAttnW)
 *   x += xb                                [attn residual]
 *   xb = rmsNorm(x, ffnNormW)             [ffn pre-norm]
 *   xb = FFN(xb)                           [gate+up+act+down]
 *   if postFfnW: xb = rmsNorm(xb, postFfnW)
 *   x += xb                                [ffn residual — x is fully updated on return]
 *
 * @param x           [wO_N floats] residual vector — updated IN PLACE (both attn+ffn adds done on GPU)
 * @param attn_full   [wO_K floats] multi-head attention output (wO matvec input)
 * @param wO          wO weight matrix pointer
 * @param typeWO      quantization type of wO (12=Q4_K, 14=Q6_K, 8=Q8_0, etc.)
 * @param wO_K        inner dim of wO (= nH * headDim, e.g. 4096 for Gemma-4 12B)
 * @param wO_N        outer dim of wO (= embeddingDim = dim, e.g. 3840 for Gemma-4 12B)
 * @param postAttnW   [wO_N floats] post-attention RMSNorm weights, or NULL to skip
 * @param hasPostAttnNorm  1 if postAttnW should be applied, 0 to skip
 * @param addOnePostAttn   1 to apply (weight+1) style (Gemma), 0 for plain RMSNorm
 * @param ffnNormW    [wO_N floats] FFN pre-norm RMSNorm weights (required)
 * @param addOneFfnNorm    1 to apply (weight+1) style
 * @param postFfnW    [wO_N floats] post-FFN RMSNorm weights, or NULL to skip
 * @param hasPostFfnNorm   1 if postFfnW should be applied, 0 to skip
 * @param addOnePostFfn    1 to apply (weight+1) style
 * @param eps         RMSNorm epsilon
 * @param gateW       FFN gate weight matrix
 * @param upW         FFN up weight matrix
 * @param downW       FFN down weight matrix
 * @param typeGate    gate quantization type
 * @param typeUp      up quantization type
 * @param typeDown    down quantization type
 * @param ffnDim      FFN intermediate dimension
 * @param activationKind  1=SwiGLU, 2=GeGLU
 * @return 0 on success, negative on failure (caller should fall back to separate dispatch)
 */
int aljabr_metal_wo_ffn_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm, int addOnePostAttn,
    const float *ffnNormW, int addOneFfnNorm,
    const float *postFfnW, int hasPostFfnNorm, int addOnePostFfn,
    float eps,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind
);

/**
 * Fuses layer L's wO + norms + FFN + residual AND layer L+1's attnNorm + QKV in ONE command buffer.
 * Eliminates all CPU-GPU synchronization between layer L FFN and layer L+1 QKV.
 */
int aljabr_metal_wo_ffn_next_qkv_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW,
    const float *postFfnW, int hasPostFfnNorm,
    float eps,
    float outScale,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind,
    int hasNext,
    const float *nextAttnNormW,
    float *nextQ, float *nextK, float *nextV,
    const void *nextWQ, const void *nextWK, const void *nextWV,
    int nextTypeQ, int nextTypeK, int nextTypeV,
    int nextN0, int nextN1, int nextN2
);

/**
 * Fuses the final layer's wO + norms + FFN + residual + outputNorm + logits projection (with softcap)
 * in ONE command buffer, eliminating the final layer wait and CPU roundtrip before logits.
 */
int aljabr_metal_wo_ffn_logits_fused(
    float *x,
    const float *attn_full,
    const void *wO, int typeWO, int wO_K, int wO_N,
    const float *postAttnW, int hasPostAttnNorm,
    const float *ffnNormW,
    const float *postFfnW, int hasPostFfnNorm,
    float eps,
    float outScale,
    const void *gateW, const void *upW, const void *downW,
    int typeGate, int typeUp, int typeDown,
    int ffnDim, int activationKind,
    const float *outputNormW,
    const void *outputW, int typeOutputW, int outputK, int outputN,
    float *logits, float cap
);

void aljabr_metal_wo_ffn_reset_x(void);

/**
 * Initializes GPU-resident shared KV cache buffers for all layers.
 */
int aljabr_metal_init_kv_cache(int nLayers, int maxSeq, const int *layerStrides);

/**
 * Synchronizes prompt KV cache contents into GPU buffers (called once after prefill).
 */
int aljabr_metal_sync_kv_cache(int layer, int seqLen, const float *kData, const float *vData);

/**
 * Registers pre-loaded norm buffers (slot 0=postAttn, 1=ffnNorm, 2=postFfn, 3=attnNorm).
 */
int aljabr_metal_register_norm_buffer(int slot, const void *ptr, size_t bytes);

/**
 * Registers per-layer rope_freqs.weight divisor factors for proportional RoPE.
 */
int aljabr_metal_register_rope_freqs(int layer, const float *freqData, int freqLen);

/**
 * Registers static weights and metadata for one decoder layer.
 */
int aljabr_metal_register_decode_layer(
    int layer, int kvLayerIndex,
    const void *wQ, const void *wK, const void *wV, const void *wO,
    const void *gate, const void *up, const void *down,
    int typeQ, int typeK, int typeV, int typeWO, int typeGate, int typeUp, int typeDown,
    int n0, int n1, int n2, int wO_K, int wO_N, int ffnDim,
    int nH, int nKVH, int headDim, int stride,
    float freqBase, int isSwa, int kvLayerIndex,
    float outScale, float attnScale,
    int hasPostAttnNorm, int hasPostFfnNorm,
    int hasQNorm, int hasKNorm, int hasUnitVNorm
);

/**
 * Registers output norm and projection weights for logit generation.
 */
int aljabr_metal_register_output(
    const float *outputNormW,
    const void *outputW, int typeOutputW, int outputK, int outputN,
    float cap
);

/**
 * Executes ALL 48 transformer layers and logits projection in a SINGLE Metal command buffer.
 * Eliminates all CPU-GPU synchronization bubbles and inter-layer memcpy operations.
 */
int aljabr_metal_decode_step(
    int pos, int slidingWindow, float eps,
    const float *x_in, float *logits_out
);

#ifdef __cplusplus
}
#endif

#endif
