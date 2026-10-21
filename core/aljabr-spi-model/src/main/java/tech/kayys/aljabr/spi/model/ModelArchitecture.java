/*
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelArchitecture.java
 * ───────────────────────
 * SPI interface for model architecture definitions.
 */
package tech.kayys.aljabr.spi.model;

import java.util.List;

/**
 * Model architecture SPI — defines weight naming and architecture-specific
 * behavior.
 *
 * <p>
 * Implementations are auto-discovered via CDI
 * {@code Instance<ModelArchitecture>}.
 * </p>
 * @author bhangun
 */
public interface ModelArchitecture {

    /**
     * Canonical architecture ID: "llama", "mistral", "gemma", "phi", "qwen2", etc.
     */
    String id();

    /**
     * Supported architecture class names (from config.json "architectures" array).
     * Example: ["LlamaForCausalLM", "MistralForCausalLM"].
     */
    List<String> supportedArchClassNames();

    /**
     * Supported model_type values (from config.json "model_type").
     * Example: ["llama", "mistral"].
     */
    List<String> supportedModelTypes();

    // ── Weight name resolution ────────────────────────────────────────────────

    private static List<String> oneCandidate(String tensorName) {
        return tensorName == null ? List.of() : List.of(tensorName);
    }

    /** Full tensor name of the token embedding table. */
    String embedTokensWeight();

    /**
     * Candidate tensor names for the token embedding table, in preference order.
     */
    default List<String> embedTokensWeightCandidates() {
        return oneCandidate(embedTokensWeight());
    }

    /** Full tensor name of the packed per-layer embedding table (Gemma-4 PLE). */
    default String embedTokensPerLayerWeight() {
        return null;
    }

    /**
     * Candidate tensor names for the packed per-layer embedding table, in
     * preference order.
     */
    default List<String> embedTokensPerLayerWeightCandidates() {
        return oneCandidate(embedTokensPerLayerWeight());
    }

    /** Full tensor name of the model -> PLE projection weight (Gemma-4 PLE). */
    default String perLayerModelProjectionWeight() {
        return null;
    }

    /**
     * Candidate tensor names for the model -> PLE projection weight, in preference
     * order.
     */
    default List<String> perLayerModelProjectionWeightCandidates() {
        return oneCandidate(perLayerModelProjectionWeight());
    }

    /** Full tensor name of the PLE projection norm weight (Gemma-4 PLE). */
    default String perLayerProjectionNormWeight() {
        return null;
    }

    /**
     * Candidate tensor names for the PLE projection norm weight, in preference
     * order.
     */
    default List<String> perLayerProjectionNormWeightCandidates() {
        return oneCandidate(perLayerProjectionNormWeight());
    }

    /** Full tensor name of the final layer norm weight. */
    String finalNormWeight();

    /**
     * Candidate tensor names for the final layer norm weight, in preference order.
     */
    default List<String> finalNormWeightCandidates() {
        return oneCandidate(finalNormWeight());
    }

    /** Full tensor name of the final layer norm bias (or null if none). */
    default String finalNormBias() {
        return null;
    }

    /** Full tensor name of the LM head weight (null if tied to embed_tokens). */
    default String lmHeadWeight() {
        return null;
    }

    /** Candidate tensor names for the LM head weight, in preference order. */
    default List<String> lmHeadWeightCandidates() {
        return oneCandidate(lmHeadWeight());
    }

    /** Full tensor name of the query projection weight for layer {@code i}. */
    String layerQueryWeight(int i);

    /**
     * Candidate tensor names for the query projection weight, in preference order.
     */
    default List<String> layerQueryWeightCandidates(int i) {
        return hasFusedQKV() ? layerFusedQKVWeightCandidates(i) : oneCandidate(layerQueryWeight(i));
    }

    /** Full tensor name of the key projection weight for layer {@code i}. */
    String layerKeyWeight(int i);

    /**
     * Candidate tensor names for the key projection weight, in preference order.
     */
    default List<String> layerKeyWeightCandidates(int i) {
        return hasFusedQKV() ? layerFusedQKVWeightCandidates(i) : oneCandidate(layerKeyWeight(i));
    }

    /** Full tensor name of the value projection weight for layer {@code i}. */
    String layerValueWeight(int i);

    /**
     * Candidate tensor names for the value projection weight, in preference order.
     */
    default List<String> layerValueWeightCandidates(int i) {
        return hasFusedQKV() ? layerFusedQKVWeightCandidates(i) : oneCandidate(layerValueWeight(i));
    }

    /** Full tensor name of the query projection bias (null if none). */
    default String layerQueryBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the query projection bias, in preference order.
     */
    default List<String> layerQueryBiasCandidates(int i) {
        return oneCandidate(layerQueryBias(i));
    }

    /** Full tensor name of the key projection bias (null if none). */
    default String layerKeyBias(int i) {
        return null;
    }

    /** Candidate tensor names for the key projection bias, in preference order. */
    default List<String> layerKeyBiasCandidates(int i) {
        return oneCandidate(layerKeyBias(i));
    }

    /** Full tensor name of the value projection bias (null if none). */
    default String layerValueBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the value projection bias, in preference order.
     */
    default List<String> layerValueBiasCandidates(int i) {
        return oneCandidate(layerValueBias(i));
    }

    /**
     * Whether Q/K/V projections are fused into a single "qkv" tensor
     * (Falcon, old GPT-NeoX, some Phi variants).
     *
     * <p>
     * When {@code true}, use {@link #layerFusedQKVWeight(int)} instead of
     * the separate Q/K/V methods.
     */
    default boolean hasFusedQKV() {
        return false;
    }

    /**
     * Full tensor name of the fused QKV weight (only when {@link #hasFusedQKV()}).
     */
    default String layerFusedQKVWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the fused QKV weight, in preference order. */
    default List<String> layerFusedQKVWeightCandidates(int i) {
        return oneCandidate(layerFusedQKVWeight(i));
    }

    /** Full tensor name of the output projection weight for layer {@code i}. */
    String layerOutputWeight(int i);

    /**
     * Candidate tensor names for the output projection weight, in preference order.
     */
    default List<String> layerOutputWeightCandidates(int i) {
        return oneCandidate(layerOutputWeight(i));
    }

    /** Full tensor name of the output projection bias (null if none). */
    default String layerOutputBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the output projection bias, in preference order.
     */
    default List<String> layerOutputBiasCandidates(int i) {
        return oneCandidate(layerOutputBias(i));
    }

    /** Full tensor name of the attention layer-norm weight for layer {@code i}. */
    String layerAttentionNormWeight(int i);

    /**
     * Candidate tensor names for the attention layer-norm weight, in preference
     * order.
     */
    default List<String> layerAttentionNormWeightCandidates(int i) {
        return oneCandidate(layerAttentionNormWeight(i));
    }

    /** Full tensor name of the attention layer-norm bias (null if absent). */
    default String layerAttentionNormBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the attention layer-norm bias, in preference
     * order.
     */
    default List<String> layerAttentionNormBiasCandidates(int i) {
        return oneCandidate(layerAttentionNormBias(i));
    }

    /** Full tensor name of the FFN gate projection weight (SwiGLU / GeGLU). */
    String layerFfnGateWeight(int i);

    /**
     * Candidate tensor names for the FFN gate projection weight, in preference
     * order.
     */
    default List<String> layerFfnGateWeightCandidates(int i) {
        return oneCandidate(layerFfnGateWeight(i));
    }

    /** Full tensor name of the FFN up projection weight. */
    String layerFfnUpWeight(int i);

    /**
     * Candidate tensor names for the FFN up projection weight, in preference order.
     */
    default List<String> layerFfnUpWeightCandidates(int i) {
        return oneCandidate(layerFfnUpWeight(i));
    }

    /** Full tensor name of the FFN down projection weight. */
    String layerFfnDownWeight(int i);

    /**
     * Candidate tensor names for the FFN down projection weight, in preference
     * order.
     */
    default List<String> layerFfnDownWeightCandidates(int i) {
        return oneCandidate(layerFfnDownWeight(i));
    }

    /** Full tensor name of the FFN gate projection bias (null if none). */
    default String layerFfnGateBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the FFN gate projection bias, in preference order.
     */
    default List<String> layerFfnGateBiasCandidates(int i) {
        return oneCandidate(layerFfnGateBias(i));
    }

    /** Full tensor name of the FFN up projection bias (null if none). */
    default String layerFfnUpBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the FFN up projection bias, in preference order.
     */
    default List<String> layerFfnUpBiasCandidates(int i) {
        return oneCandidate(layerFfnUpBias(i));
    }

    /** Full tensor name of the FFN down projection bias (null if none). */
    default String layerFfnDownBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the FFN down projection bias, in preference order.
     */
    default List<String> layerFfnDownBiasCandidates(int i) {
        return oneCandidate(layerFfnDownBias(i));
    }

    /** Full tensor name of the post-attention/pre-FFN layer-norm weight. */
    String layerFfnNormWeight(int i);

    /**
     * Candidate tensor names for the post-attention/pre-FFN layer-norm weight, in
     * preference order.
     */
    default List<String> layerFfnNormWeightCandidates(int i) {
        return oneCandidate(layerFfnNormWeight(i));
    }

    /** Full tensor name of the post-attention layer-norm bias (null if absent). */
    default String layerFfnNormBias(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the post-attention layer-norm bias, in preference
     * order.
     */
    default List<String> layerFfnNormBiasCandidates(int i) {
        return oneCandidate(layerFfnNormBias(i));
    }

    /** Full tensor name of the Q-norm weight (null if none, e.g. Qwen 2.5). */
    default String layerQueryNormWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the Q-norm weight, in preference order. */
    default List<String> layerQueryNormWeightCandidates(int i) {
        return oneCandidate(layerQueryNormWeight(i));
    }

    /** Full tensor name of the K-norm weight (null if none, e.g. Qwen 2.5). */
    default String layerKeyNormWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the K-norm weight, in preference order. */
    default List<String> layerKeyNormWeightCandidates(int i) {
        return oneCandidate(layerKeyNormWeight(i));
    }

    /** Full tensor name of the post-attention norm weight (Gemma-2). */
    default String layerPostAttnNormWeight(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the post-attention norm weight, in preference
     * order.
     */
    default List<String> layerPostAttnNormWeightCandidates(int i) {
        return oneCandidate(layerPostAttnNormWeight(i));
    }

    /** Full tensor name of the pre-FFN norm weight (Gemma-2). */
    default String layerPreFfnNormWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the pre-FFN norm weight, in preference order. */
    default List<String> layerPreFfnNormWeightCandidates(int i) {
        return oneCandidate(layerPreFfnNormWeight(i));
    }

    /** Full tensor name of the post-FFN norm weight (Gemma-2/3/4 style blocks). */
    default String layerPostFfnNormWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the post-FFN norm weight, in preference order. */
    default List<String> layerPostFfnNormWeightCandidates(int i) {
        return oneCandidate(layerPostFfnNormWeight(i));
    }

    /** Full tensor name of the per-layer input gate weight (Gemma-4 PLE). */
    default String layerPerLayerInputGateWeight(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the per-layer input gate weight, in preference
     * order.
     */
    default List<String> layerPerLayerInputGateWeightCandidates(int i) {
        return oneCandidate(layerPerLayerInputGateWeight(i));
    }

    /** Full tensor name of the per-layer projection weight (Gemma-4 PLE). */
    default String layerPerLayerProjectionWeight(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the per-layer projection weight, in preference
     * order.
     */
    default List<String> layerPerLayerProjectionWeightCandidates(int i) {
        return oneCandidate(layerPerLayerProjectionWeight(i));
    }

    /** Full tensor name of the post per-layer input norm weight (Gemma-4 PLE). */
    default String layerPostPerLayerInputNormWeight(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the post per-layer input norm weight, in
     * preference order.
     */
    default List<String> layerPostPerLayerInputNormWeightCandidates(int i) {
        return oneCandidate(layerPostPerLayerInputNormWeight(i));
    }

    /** Full tensor name of the residual skip scale (Gemma-4 layer_scalar). */
    default String layerScalarWeight(int i) {
        return null;
    }

    /** Candidate tensor names for the residual skip scale, in preference order. */
    default List<String> layerScalarWeightCandidates(int i) {
        return oneCandidate(layerScalarWeight(i));
    }

    /**
     * Full tensor name of the MoE router gate weight (null for dense FFN layers).
     */
    default String layerMoeGateWeight(int i) {
        return null;
    }

    /**
     * Candidate tensor names for the MoE router gate weight, in preference order.
     */
    default List<String> layerMoeGateWeightCandidates(int i) {
        return oneCandidate(layerMoeGateWeight(i));
    }

    /**
     * Full tensor name of the expert FFN gate projection weight (null for dense FFN
     * layers).
     */
    default String expertGateWeight(int layerIdx, int expertIdx) {
        return null;
    }

    /**
     * Candidate tensor names for the expert FFN gate projection weight, in
     * preference order.
     */
    default List<String> expertGateWeightCandidates(int layerIdx, int expertIdx) {
        return oneCandidate(expertGateWeight(layerIdx, expertIdx));
    }

    /**
     * Full tensor name of the expert FFN up projection weight (null for dense FFN
     * layers).
     */
    default String expertUpWeight(int layerIdx, int expertIdx) {
        return null;
    }

    /**
     * Candidate tensor names for the expert FFN up projection weight, in preference
     * order.
     */
    default List<String> expertUpWeightCandidates(int layerIdx, int expertIdx) {
        return oneCandidate(expertUpWeight(layerIdx, expertIdx));
    }

    /**
     * Full tensor name of the expert FFN down projection weight (null for dense FFN
     * layers).
     */
    default String expertDownWeight(int layerIdx, int expertIdx) {
        return null;
    }

    /**
     * Candidate tensor names for the expert FFN down projection weight, in
     * preference order.
     */
    default List<String> expertDownWeightCandidates(int layerIdx, int expertIdx) {
        return oneCandidate(expertDownWeight(layerIdx, expertIdx));
    }

    // ── Architecture properties ───────────────────────────────────────────────

    /** The activation function used in the FFN (e.g. SiLU, GELU). */
    default FFNActivationType activationType() {
        return FFNActivationType.SILU;
    }

    /** Whether this architecture uses RMSNorm (true) or LayerNorm (false). */
    default boolean usesRmsNorm() {
        return true;
    }

    /**
     * Whether this architecture has a separate gate projection in the FFN
     * (i.e. SwiGLU / GeGLU gate×up structure).
     */
    default boolean hasSeparateGateProjection() {
        return true;
    }

    /**
     * Check if embeddings are tied with LM head.
     */
    default boolean hasTiedEmbeddings() {
        return false;
    }

    /**
     * RMSNorm epsilon from config.
     */
    default double rmsNormEps() {
        return 1e-6;
    }

    /**
     * Check if this architecture has sliding window attention.
     */
    default boolean hasSlidingWindow() {
        return false;
    }

    /**
     * Get sliding window size (if applicable).
     */
    default int slidingWindowSize() {
        return Integer.MAX_VALUE;
    }

    // ── Runtime inference behaviors ──────────────────────────────────────────

    /**
     * Embedding scale factor applied after token embedding lookup.
     * Gemma models multiply embeddings by sqrt(hidden_dim).
     * Most other architectures return 1.0 (no scaling).
     *
     * @param hiddenDim the model's hidden dimension
     * @return the scale factor to multiply embeddings by
     */
    default float embeddingScaleFactor(int hiddenDim) {
        return 1.0f;
    }

    /**
     * Whether this architecture uses Neox-style RoPE (split-half rotation)
     * or interleaved (LLaMA/GPT-J style, adjacent-pair rotation).
     *
     * @return true for Neox style, false for interleaved
     */
    default boolean usesNeoxRope() {
        return true;
    }

    /**
     * Default attention logit soft-capping value.
     * Gemma-2 uses 50.0. Most architectures return 0 (disabled).
     */
    default float defaultAttnSoftCap() {
        return 0.0f;
    }

    /**
     * Default final logit soft-capping value.
     * Gemma-2 uses 30.0. Most architectures return 0 (disabled).
     */
    default float defaultFinalSoftCap() {
        return 0.0f;
    }

    /**
     * Default RoPE frequency base for this architecture.
     */
    default float defaultRopeFreqBase() {
        return 10000.0f;
    }

    /**
     * Whether this architecture requires adding 1.0 to the RMSNorm weights.
     * Used by Gemma models which store w-1.
     */
    default boolean addOneToRmsNormWeight() {
        return false;
    }

    /**
     * Runtime-only family traits for inference policy decisions.
     *
     * <p>
     * Model farm implementations should override this when a family has
     * tokenizer, normalization, cache, or kernel policy details that are not
     * expressible as weight names.
     * </p>
     */
    default ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }

    /**
     * Whether the GGUF architecture string matches this family.
     * Used for non-CDI resolution in the native inference path.
     *
     * @param ggufArch the architecture string from GGUF metadata (e.g. "gemma",
     *                 "llama")
     * @return true if this family handles the given arch
     */
    default boolean matchesGgufArch(String ggufArch) {
        return id().equals(ggufArch) || supportedModelTypes().contains(ggufArch);
    }
}
