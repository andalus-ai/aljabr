/**
 * aljabr_metal_pipelines.h — runtime Metal pipeline compilation and
 * storage.
 */

#ifndef ALJABR_METAL_PIPELINES_H
#define ALJABR_METAL_PIPELINES_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
  __strong id<MTLComputePipelineState> matvec_half;
  __strong id<MTLComputePipelineState> matvec_t_half;
  __strong id<MTLComputePipelineState> matvec_tb_int4;
  __strong id<MTLComputePipelineState> matvec_tb_int4_128;
  __strong id<MTLComputePipelineState> matvec_tb_nf4;
  __strong id<MTLComputePipelineState> matvec_tb_nf4_128;
  // GGML block-quantized pipelines
  __strong id<MTLComputePipelineState> matvec_tb_q4_k;
  __strong id<MTLComputePipelineState> matvec_tb_q4_k_mr4; // 4-row multi-row: 4× A bandwidth reduction
  __strong id<MTLComputePipelineState> matvec_tb_q4_k_mr4_softcap; // 4-row multi-row Q4_K with in-kernel softcap
  __strong id<MTLComputePipelineState> matvec_tb_q4_k_b4_mr4; // 4-token batched 4-row multi-row Q4_K for prefill
  __strong id<MTLComputePipelineState> ffn_gate_up_act_q4_k_mr4; // Fused Gate + Up + Act (GeGLU/SwiGLU) Q4_K
  __strong id<MTLComputePipelineState> ffn_gate_up_act_q4_k_b4_mr4; // 4-token batched Fused Gate + Up + Act Q4_K
  __strong id<MTLComputePipelineState> ffn_gate_up_act_q6_k_mr4; // Fused Gate + Up + Act (GeGLU/SwiGLU) Q6_K
  __strong id<MTLComputePipelineState> matvec_tb_q6_k;
  __strong id<MTLComputePipelineState> matvec_tb_q6_k_mr4; // 4-row multi-row Q6_K
  __strong id<MTLComputePipelineState> matvec_tb_q6_k_mr4_softcap; // 4-row multi-row Q6_K with in-kernel softcap
  __strong id<MTLComputePipelineState> matvec_tb_q6_k_b4_mr4; // 4-token batched 4-row multi-row Q6_K for prefill
  __strong id<MTLComputePipelineState> matvec_tb_q8_0;
  __strong id<MTLComputePipelineState> matvec_half_pair;
  __strong id<MTLComputePipelineState> matvec_half_triple_mixed;
  __strong id<MTLComputePipelineState> matvec_bf16;
  __strong id<MTLComputePipelineState> matvec_bf16_pair;
  __strong id<MTLComputePipelineState> matvec_bf16_pair_simd;
  __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed;
  __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed_x4;
  __strong id<MTLComputePipelineState> matvec_bf16_x4;
  __strong id<MTLComputePipelineState> matvec_bf16_x8;
  __strong id<MTLComputePipelineState> matvec_bf16_pair_x4;
  __strong id<MTLComputePipelineState> matvec_bf16_x4_simd;
  __strong id<MTLComputePipelineState> matvec_bf16_pair_x4_simd;
  __strong id<MTLComputePipelineState> matvec_half_gated_pair;
  __strong id<MTLComputePipelineState> matvec_bf16_gated_pair;
  __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_x4;
  __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_simd;
  __strong id<MTLComputePipelineState> matvec_bf16_rows_gated_pair;
  __strong id<MTLComputePipelineState> matvec_bf16_rows;
  __strong id<MTLComputePipelineState> matvec_bf16_rows_gated_pair_x4;
  __strong id<MTLComputePipelineState> matvec_bf16_rows_x4;
  __strong id<MTLComputePipelineState> matvec_half_128;
  __strong id<MTLComputePipelineState> matvec_t_half_128;
  __strong id<MTLComputePipelineState> matvec_half_pair_128;
  __strong id<MTLComputePipelineState> matvec_half_triple_mixed_128;
  __strong id<MTLComputePipelineState> matvec_bf16_128;
  __strong id<MTLComputePipelineState> matvec_bf16_pair_128;
  __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed_128;
  __strong id<MTLComputePipelineState> matvec_half_gated_pair_128;
  __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_128;
  __strong id<MTLComputePipelineState> add;
  __strong id<MTLComputePipelineState> silu_ffn;
  __strong id<MTLComputePipelineState> gelu_ffn;
  __strong id<MTLComputePipelineState> rope;
  __strong id<MTLComputePipelineState> rope_float2;
  __strong id<MTLComputePipelineState> rmsnorm;
  __strong id<MTLComputePipelineState> rmsnorm_float4;
  __strong id<MTLComputePipelineState> rmsnorm_rows;
  __strong id<MTLComputePipelineState> layernorm;
  __strong id<MTLComputePipelineState> layernorm_float4;
  __strong id<MTLComputePipelineState> layernorm_rows;
  __strong id<MTLComputePipelineState> silu;
  __strong id<MTLComputePipelineState> gelu;
  __strong id<MTLComputePipelineState> softmax;
  __strong id<MTLComputePipelineState> softmax_rows;
  __strong id<MTLComputePipelineState> softcap; // Logit soft-capping: x[i] = cap * tanh(x[i] / cap)
  __strong id<MTLComputePipelineState> scale;   // Vector scalar scaling: x[i] *= scale
  __strong id<MTLComputePipelineState> decode_attention;
  __strong id<MTLComputePipelineState> flash_attention;
} AljabrMetalPipelines;

AljabrMetalPipelines *aljabr_metal_pipelines(void);
void aljabr_metal_compile_runtime_pipelines(
    AljabrMetalPipelines *pipelines);

#endif
