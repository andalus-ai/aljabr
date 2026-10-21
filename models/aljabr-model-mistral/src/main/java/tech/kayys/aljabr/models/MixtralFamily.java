/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MixtralFamily.java
 * ───────────────────
 * ModelArchitecture for Mixtral 8×7B and Mixtral 8×22B.
 *
 * Weight layout differs from LLaMA in the FFN layers:
 *   LLaMA:   model.layers.{i}.mlp.gate_proj.weight
 *   Mixtral: model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight  (j=0..7)
 *            model.layers.{i}.block_sparse_moe.gate.weight  (router)
 *
 * The layerFfn*Weight methods return null for the router weights
 * because MoeForwardPass handles expert weight lookup directly.
 * DirectForwardPass checks config.isMoeLayer() and routes appropriately.
 */
package tech.kayys.tafkir.models;

import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Architecture for Mistral MoE models (Mixtral 8×7B, 8×22B).
 */
public final class MixtralFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "mixtral";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("MixtralForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("mixtral");
    }

    // ── Shared with LLaMA-style (attention weights unchanged) ─────────────────

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public String lmHeadWeight() {
        return "lm_head.weight";
    }

    @Override
    public String layerQueryWeight(int i) {
        return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public String layerKeyWeight(int i) {
        return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    @Override
    public String layerOutputWeight(int i) {
        return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "model.layers.%d.input_layernorm.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    // ── MoE expert weights — null signals MoeForwardPass to handle directly ───

    @Override
    public String layerFfnGateWeight(int i) {
        return null;
    } // handled by MoeForwardPass

    @Override
    public String layerFfnUpWeight(int i) {
        return null;
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return null;
    }

    // ── MoE router gate ────────────────────────────────────────────────────────

    /**
     * Router gate weight for layer i.
     * Shape: [numExperts, hiddenSize]
     */
    @Override
    public String layerMoeGateWeight(int i) {
        return "model.layers.%d.block_sparse_moe.gate.weight".formatted(i);
    }

    /**
     * Expert FFN gate projection weight (SwiGLU w1).
     * Shape: [intermediateSize, hiddenSize]
     */
    @Override
    public String expertGateWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.block_sparse_moe.experts.%d.w1.weight"
                .formatted(layerIdx, expertIdx);
    }

    /**
     * Expert FFN up projection weight (SwiGLU w3).
     * Shape: [intermediateSize, hiddenSize]
     */
    @Override
    public String expertUpWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.block_sparse_moe.experts.%d.w3.weight"
                .formatted(layerIdx, expertIdx);
    }

    /**
     * Expert FFN down projection weight (SwiGLU w2).
     * Shape: [hiddenSize, intermediateSize]
     */
    @Override
    public String expertDownWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.block_sparse_moe.experts.%d.w2.weight"
                .formatted(layerIdx, expertIdx);
    }
}
