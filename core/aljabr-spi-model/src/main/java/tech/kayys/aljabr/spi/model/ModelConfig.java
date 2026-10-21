/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelConfig.java
 * ─────────────────
 * Model configuration from config.json.
 */
package tech.kayys.aljabr.spi.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import org.jboss.logging.Logger;

/**
 * Parsed HuggingFace {@code config.json} for a transformer model checkpoint.
 *
 * <p>
 * Load via {@link #load(Path, ObjectMapper)} or
 * {@link #fromDirectory(Path, ObjectMapper)}.
 *
 * <p>
 * All fields default to sensible values if absent from the JSON — this
 * handles both slim community configs and full Transformers library exports.
 * @author bhangun
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    private static final String DISABLE_GEMMA4_SHARED_KV_PROPERTY = "aljabr.safetensor.disable_gemma4_shared_kv";

    public ModelConfig() {
        // Defaults applied via field initializers
    }

    private static final Logger log = Logger.getLogger(ModelConfig.class);

    // ── Architecture identity ─────────────────────────────────────────────────

    /**
     * Architecture type string — maps to a registered {@link ModelArchitecture}.
     * Examples: "LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM",
     * "Qwen2ForCausalLM", "PhiForCausalLM", "GPTNeoXForCausalLM".
     */
    @JsonProperty("architectures")
    private List<String> architectures;

    /** Short model-type key: "llama", "mistral", "gemma", "phi", "qwen2", etc. */
    @JsonProperty("model_type")
    private String modelType;

    // ── Core dimensions ───────────────────────────────────────────────────────

    /** Embedding / residual stream dimension. */
    @JsonProperty("hidden_size")
    private int hiddenSize = 4096;

    /** Number of transformer layers. */
    @JsonProperty("num_hidden_layers")
    private int numHiddenLayers = 32;

    /** Number of attention heads (query heads). */
    @JsonProperty("num_attention_heads")
    private int numAttentionHeads = 32;

    /**
     * Number of key-value heads.
     * Equal to {@code numAttentionHeads} for MHA.
     * Less than {@code numAttentionHeads} for GQA (LLaMA-2-70B, Mistral, Gemma2).
     * 1 for MQA.
     */
    @JsonProperty("num_key_value_heads")
    private Integer numKeyValueHeads; // null → defaults to numAttentionHeads

    /** FFN intermediate dimension. Typically 4× or 2.67× hidden_size. */
    @JsonProperty("intermediate_size")
    private int intermediateSize = 11008;

    /** Vocabulary size. */
    @JsonProperty("vocab_size")
    private int vocabSize = 32000;

    /** Maximum sequence length the model was trained on. */
    @JsonProperty("max_position_embeddings")
    private int maxPositionEmbeddings = 4096;

    /** Hidden size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("hidden_size_per_layer_input")
    private int hiddenSizePerLayerInput = 0;

    /** Vocab size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("vocab_size_per_layer_input")
    private int vocabSizePerLayerInput = 0;

    // ── Normalization ─────────────────────────────────────────────────────────

    /** RMS norm epsilon (LLaMA, Gemma, Mistral). */
    @JsonProperty("rms_norm_eps")
    private double rmsNormEps = 1e-5;

    /** Layer norm epsilon (GPT-2, BERT-style). */
    @JsonProperty("layer_norm_eps")
    private double layerNormEps = 1e-5;

    // ── Rotary position embedding ─────────────────────────────────────────────

    /** RoPE base frequency (default 10000 for most models, 500000 for LLaMA-3). */
    @JsonProperty("rope_theta")
    private double ropeTheta = 10000.0;

    /** Per-dimension RoPE scaling factor (LLaMA-2 long-context). */
    @JsonProperty("rope_scaling")
    private RopeScaling ropeScaling;

    // ── Gemma-4 layer-specific RoPE ─────────────────────────────────────────
    @JsonProperty("layer_types")
    private List<String> layerTypes;

    @JsonProperty("rope_local_base_freq")
    private Double ropeLocalBaseFreq;

    private Double ropeThetaFull;
    private Double ropeThetaSliding;
    @JsonProperty("partial_rotary_factor")
    private Double partialRotaryFactor;
    private Double partialRotaryFactorFull;
    private Double partialRotaryFactorSliding;

    @JsonProperty("query_pre_attn_scalar")
    private Double queryPreAttnScalar;

    @JsonProperty("attn_logit_softcapping")
    private Double attnLogitSoftcapping;

    @JsonProperty("final_logit_softcapping")
    private Double finalLogitSoftcapping;

    // ── Activation function ───────────────────────────────────────────────────

    /** Activation in the FFN: "silu", "gelu", "relu", "gelu_new". */
    @JsonProperty("hidden_act")
    private String hiddenAct = "silu";

    // ── Special tokens ────────────────────────────────────────────────────────

    @JsonProperty("bos_token_id")
    private Integer bosTokenId;
    @JsonProperty("eos_token_id")
    private Integer eosTokenId;
    private List<Integer> eosTokenIds = new ArrayList<>();
    @JsonProperty("pad_token_id")
    private Integer padTokenId;

    // ── Dtype ─────────────────────────────────────────────────────────────────

    /** Storage dtype declared in config: "bfloat16", "float16", "float32". */
    @JsonProperty("torch_dtype")
    private String torchDtype = "bfloat16";

    // ── Tie embeddings ────────────────────────────────────────────────────────

    /**
     * Whether {@code lm_head.weight} is tied (shared) with
     * {@code model.embed_tokens.weight}.
     */
    @JsonProperty("tie_word_embeddings")
    private boolean tieWordEmbeddings = false;

    // ── Sliding window attention (Mistral) ────────────────────────────────────

    @JsonProperty("sliding_window")
    private Integer slidingWindow; // null = full attention

    @JsonProperty("use_sliding_window")
    private Boolean useSlidingWindow;

    // ── Head dimension override (Gemma2, some Phi variants) ──────────────────

    @JsonProperty("head_dim")
    private Integer headDim; // null = hiddenSize / numAttentionHeads

    @JsonProperty("global_head_dim")
    private Integer globalHeadDim;

    @JsonProperty("num_global_key_value_heads")
    private Integer numGlobalKeyValueHeads;

    @JsonProperty("attention_k_eq_v")
    private Boolean attentionKeyEqualsValue;

    @JsonProperty("num_kv_shared_layers")
    private Integer numKvSharedLayers;

    // ── MoE fields ──────────────────────────────────────────────────────────

    @JsonProperty("num_local_experts")
    private Integer numLocalExperts;

    @JsonProperty("num_experts_per_tok")
    private Integer numExpertsPerTok;

    @JsonProperty("decoder_sparse_step")
    private Integer decoderSparseStep;

    @JsonProperty("enable_moe_block")
    private Boolean enableMoeBlock;

    @JsonProperty("moe_intermediate_size")
    private Integer moeIntermediateSize;

    @JsonProperty("use_double_wide_mlp")
    private Boolean useDoubleWideMlp;

    // ─────────────────────────────────────────────────────────────────────────
    // Factory / loading
    // ─────────────────────────────────────────────────────────────────────────

    /** Parse {@code config.json} from an explicit file path. */
    public static ModelConfig load(Path configPath, ObjectMapper mapper) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("config.json not found: " + configPath);
        }
        ModelConfig cfg = mapper.readValue(configPath.toFile(), ModelConfig.class);
        // Fallback for multi-modal configs where root fields may be lost or malformed.
        if ((cfg.modelType == null || cfg.modelType.isBlank())
                && (cfg.architectures == null || cfg.architectures.isEmpty())) {
            try {
                JsonNode root = mapper.readTree(configPath.toFile());
                String inferredType = textValue(root, "model_type");
                List<String> inferredArch = listValue(root, "architectures");

                if (inferredType == null || inferredType.isBlank()) {
                    inferredType = textValue(root.path("text_config"), "model_type");
                }
                if (inferredArch == null || inferredArch.isEmpty()) {
                    inferredArch = listValue(root.path("text_config"), "architectures");
                }
                if ((inferredType != null && !inferredType.isBlank())
                        || (inferredArch != null && !inferredArch.isEmpty())) {
                    cfg.modelType = inferredType;
                    cfg.architectures = inferredArch;
                    log.infof("ModelConfig: inferred fields from fallback: type=%s arch=%s",
                            cfg.modelType, cfg.primaryArchitecture());
                }
            } catch (Exception e) {
                log.debugf("ModelConfig: fallback inference failed: %s", e.getMessage());
            }
        }
        // Merge text_config values when root fields are missing (Gemma4-style configs).
        try {
            JsonNode root = mapper.readTree(configPath.toFile());
            JsonNode textCfg = root.path("text_config");
            if (textCfg != null && !textCfg.isMissingNode()) {
                if (!root.has("hidden_size")) {
                    Integer v = intValue(textCfg, "hidden_size");
                    if (v != null)
                        cfg.hiddenSize = v;
                }
                if (!root.has("num_hidden_layers")) {
                    Integer v = intValue(textCfg, "num_hidden_layers");
                    if (v != null)
                        cfg.numHiddenLayers = v;
                }
                if (!root.has("num_attention_heads")) {
                    Integer v = intValue(textCfg, "num_attention_heads");
                    if (v != null)
                        cfg.numAttentionHeads = v;
                }
                if (!root.has("num_key_value_heads")) {
                    Integer v = intValue(textCfg, "num_key_value_heads");
                    if (v != null)
                        cfg.numKeyValueHeads = v;
                }
                if (!root.has("intermediate_size")) {
                    Integer v = intValue(textCfg, "intermediate_size");
                    if (v != null)
                        cfg.intermediateSize = v;
                }
                if (!root.has("vocab_size")) {
                    Integer v = intValue(textCfg, "vocab_size");
                    if (v != null)
                        cfg.vocabSize = v;
                }
                if (!root.has("rms_norm_eps")) {
                    Double v = doubleValue(textCfg, "rms_norm_eps");
                    if (v != null)
                        cfg.rmsNormEps = v;
                }
                if (!root.has("rope_theta")) {
                    Double v = doubleValue(textCfg, "rope_theta");
                    if (v != null)
                        cfg.ropeTheta = v;
                }
                if (!root.has("head_dim")) {
                    Integer v = intValue(textCfg, "head_dim");
                    if (v != null)
                        cfg.headDim = v;
                }
                if (!root.has("global_head_dim")) {
                    Integer v = intValue(textCfg, "global_head_dim");
                    if (v != null)
                        cfg.globalHeadDim = v;
                }
                if (!root.has("num_global_key_value_heads")) {
                    Integer v = intValue(textCfg, "num_global_key_value_heads");
                    if (v != null)
                        cfg.numGlobalKeyValueHeads = v;
                }
                if (!root.has("attention_k_eq_v")) {
                    Boolean v = boolValue(textCfg, "attention_k_eq_v");
                    if (v != null)
                        cfg.attentionKeyEqualsValue = v;
                }
                if (!root.has("num_kv_shared_layers")) {
                    Integer v = intValue(textCfg, "num_kv_shared_layers");
                    if (v != null)
                        cfg.numKvSharedLayers = v;
                }
                if (!root.has("enable_moe_block")) {
                    Boolean v = boolValue(textCfg, "enable_moe_block");
                    if (v != null)
                        cfg.enableMoeBlock = v;
                }
                if (!root.has("num_local_experts") && !root.has("num_experts")) {
                    Integer v = intValue(textCfg, "num_local_experts");
                    if (v == null) {
                        v = intValue(textCfg, "num_experts");
                    }
                    if (v != null)
                        cfg.numLocalExperts = v;
                }
                if (!root.has("num_experts_per_tok") && !root.has("top_k_experts")) {
                    Integer v = intValue(textCfg, "num_experts_per_tok");
                    if (v == null) {
                        v = intValue(textCfg, "top_k_experts");
                    }
                    if (v != null)
                        cfg.numExpertsPerTok = v;
                }
                if (!root.has("moe_intermediate_size")) {
                    Integer v = intValue(textCfg, "moe_intermediate_size");
                    if (v != null)
                        cfg.moeIntermediateSize = v;
                }
                if (!root.has("use_double_wide_mlp")) {
                    Boolean v = boolValue(textCfg, "use_double_wide_mlp");
                    if (v != null)
                        cfg.useDoubleWideMlp = v;
                }

                if (!root.has("query_pre_attn_scalar")) {
                    Double v = doubleValue(textCfg, "query_pre_attn_scalar");
                    if (v != null)
                        cfg.queryPreAttnScalar = v;
                }
                if (!root.has("attn_logit_softcapping")) {
                    Double v = doubleValue(textCfg, "attn_logit_softcapping");
                    if (v != null)
                        cfg.attnLogitSoftcapping = v;
                }
                if (!root.has("final_logit_softcapping")) {
                    Double v = doubleValue(textCfg, "final_logit_softcapping");
                    if (v != null)
                        cfg.finalLogitSoftcapping = v;
                }
                if (!root.has("layer_types")) {
                    JsonNode types = textCfg.get("layer_types");
                    if (types != null && types.isArray()) {
                        List<String> values = new ArrayList<>();
                        types.forEach(n -> values.add(n.asText()));
                        cfg.layerTypes = values;
                    }
                }
                if (!root.has("rope_parameters")) {
                    JsonNode ropeParams = textCfg.get("rope_parameters");
                    if (ropeParams != null && ropeParams.isObject()) {
                        mergeRopeParameters(cfg, ropeParams);
                    }
                }
                if (!root.has("partial_rotary_factor")) {
                    Double v = doubleValue(textCfg, "partial_rotary_factor");
                    if (v != null)
                        cfg.partialRotaryFactor = v;
                }
                if (!root.has("rope_local_base_freq")) {
                    Double v = doubleValue(textCfg, "rope_local_base_freq");
                    if (v != null)
                        cfg.ropeLocalBaseFreq = v;
                }
                if (!root.has("hidden_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "hidden_size_per_layer_input");
                    if (v != null)
                        cfg.hiddenSizePerLayerInput = v;
                }
                if (!root.has("vocab_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "vocab_size_per_layer_input");
                    if (v != null)
                        cfg.vocabSizePerLayerInput = v;
                }
                if (!root.has("bos_token_id")) {
                    Integer v = intValue(textCfg, "bos_token_id");
                    if (v != null)
                        cfg.bosTokenId = v;
                }
                if (!root.has("eos_token_id")) {
                    Integer v = intValue(textCfg, "eos_token_id");
                    if (v != null)
                        cfg.eosTokenId = v;
                }
                if (!root.has("pad_token_id")) {
                    Integer v = intValue(textCfg, "pad_token_id");
                    if (v != null)
                        cfg.padTokenId = v;
                }
                if (!root.has("sliding_window")) {
                    Integer v = intValue(textCfg, "sliding_window");
                    if (v != null) {
                        cfg.slidingWindow = v;
                    }
                }
                if (!root.has("use_sliding_window")) {
                    Boolean v = boolValue(textCfg, "use_sliding_window");
                    if (v != null) {
                        cfg.useSlidingWindow = v;
                    }
                }
                if (!root.has("max_position_embeddings")) {
                    Integer v = intValue(textCfg, "max_position_embeddings");
                    if (v != null) {
                        cfg.maxPositionEmbeddings = v;
                    }
                }
                if (!root.has("hidden_act")) {
                    String act = textValue(textCfg, "hidden_activation");
                    if (act == null || act.isBlank()) {
                        act = textValue(textCfg, "hidden_act");
                    }
                    if (act != null && !act.isBlank()) {
                        cfg.hiddenAct = act;
                    }
                }
            }
            JsonNode rootRopeParams = root.get("rope_parameters");
            if (rootRopeParams != null && rootRopeParams.isObject()) {
                mergeRopeParameters(cfg, rootRopeParams);
            }
            String rootActivation = textValue(root, "hidden_activation");
            if (rootActivation != null && !rootActivation.isBlank()) {
                cfg.hiddenAct = rootActivation;
            }
            String dtype = textValue(root, "dtype");
            if (dtype != null && !dtype.isBlank()) {
                cfg.torchDtype = dtype;
            }
            reconcileGemma3RopeDefaults(cfg);
        } catch (Exception e) {
            log.debugf("ModelConfig: text_config merge failed: %s", e.getMessage());
        }
        log.infof("Loaded model config: type=%s arch=%s layers=%d hidden=%d heads=%d kvHeads=%d",
                cfg.modelType, cfg.primaryArchitecture(),
                cfg.numHiddenLayers, cfg.hiddenSize,
                cfg.numAttentionHeads, cfg.resolvedNumKvHeads());
        return cfg;
    }

    /**
     * Load {@code config.json} from the model directory.
     *
     * @param modelDir directory containing config.json and the safetensors weights
     */
    public static ModelConfig fromDirectory(Path modelDir, ObjectMapper mapper) throws IOException {
        return load(modelDir.resolve("config.json"), mapper);
    }

    /**
     * Build a direct-engine config from GGUF metadata.
     *
     * <p>
     * GGUF exports store the same architectural facts as HuggingFace
     * {@code config.json}, but under architecture-prefixed keys such as
     * {@code llama.embedding_length} or {@code gemma4.attention.head_count}.
     * This mapper keeps the Java-native GGUF path from using constructor
     * defaults that accidentally describe a different model.
     * </p>
     */
    public static ModelConfig fromGgufMetadata(Map<String, Object> metadata) {
        Map<String, Object> meta = metadata != null ? metadata : Map.of();
        String arch = metadataString(meta, "general.architecture")
                .orElse("llama")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (arch.isBlank()) {
            arch = "llama";
        }

        ModelConfig cfg = new ModelConfig();
        cfg.modelType = arch;
        cfg.architectures = List.of(ggufArchitectureClassName(arch));
        cfg.maxPositionEmbeddings = metadataInt(meta, cfg.maxPositionEmbeddings,
                arch + ".context_length",
                "general.context_length");
        cfg.hiddenSize = metadataInt(meta, cfg.hiddenSize,
                arch + ".embedding_length",
                "general.embedding_length");
        cfg.numHiddenLayers = metadataInt(meta, cfg.numHiddenLayers,
                arch + ".block_count",
                "general.block_count");
        cfg.intermediateSize = metadataMaxInt(meta, cfg.intermediateSize,
                arch + ".feed_forward_length",
                "general.feed_forward_length");
        cfg.numAttentionHeads = metadataInt(meta, cfg.numAttentionHeads,
                arch + ".attention.head_count",
                "general.attention.head_count");
        cfg.numKeyValueHeads = metadataInt(meta, cfg.numAttentionHeads,
                arch + ".attention.head_count_kv",
                "general.attention.head_count_kv");
        cfg.vocabSize = metadataInt(meta,
                metadataTokenCount(meta).orElse(cfg.vocabSize),
                arch + ".vocab_size",
                "general.vocab_size");
        cfg.rmsNormEps = metadataDouble(meta, cfg.rmsNormEps,
                arch + ".attention.layer_norm_rms_epsilon",
                "general.attention.layer_norm_rms_epsilon");
        cfg.ropeTheta = metadataDouble(meta, cfg.ropeTheta,
                arch + ".rope.freq_base",
                "general.rope.freq_base");
        cfg.ropeThetaFull = cfg.ropeTheta;

        Integer localHeadDim = metadataIntOptional(meta,
                arch + ".attention.head_dim",
                arch + ".attention.key_length_swa",
                arch + ".rope.dimension_count_swa",
                "general.attention.head_dim").orElse(null);
        Integer globalHeadDim = metadataIntOptional(meta,
                arch + ".attention.key_length",
                arch + ".rope.dimension_count").orElse(null);
        if (localHeadDim == null) {
            localHeadDim = globalHeadDim != null
                    ? globalHeadDim
                    : (cfg.numAttentionHeads > 0 ? cfg.hiddenSize / cfg.numAttentionHeads : 0);
        }
        cfg.headDim = localHeadDim != null && localHeadDim > 0 ? localHeadDim : null;
        if (globalHeadDim != null && globalHeadDim > 0 && !globalHeadDim.equals(cfg.headDim)) {
            cfg.globalHeadDim = globalHeadDim;
        }

        metadataIntOptional(meta, arch + ".attention.sliding_window")
                .ifPresent(value -> {
                    cfg.slidingWindow = value;
                    cfg.useSlidingWindow = value > 0;
                });
        metadataIntOptional(meta, arch + ".attention.shared_kv_layers")
                .ifPresent(value -> cfg.numKvSharedLayers = value);
        metadataIntOptional(meta, arch + ".embedding_length_per_layer_input")
                .ifPresent(value -> cfg.hiddenSizePerLayerInput = value);
        metadataIntOptional(meta, arch + ".vocab_size_per_layer_input")
                .ifPresent(value -> cfg.vocabSizePerLayerInput = value);
        metadataDoubleOptional(meta, arch + ".rope.freq_base_swa")
                .ifPresent(value -> {
                    cfg.ropeLocalBaseFreq = value;
                    cfg.ropeThetaSliding = value;
                });
        metadataDoubleOptional(meta, arch + ".final_logit_softcapping")
                .ifPresent(value -> cfg.finalLogitSoftcapping = value);
        metadataIntOptional(meta, "tokenizer.ggml.bos_token_id")
                .ifPresent(value -> cfg.bosTokenId = value);
        metadataIntOptional(meta, "tokenizer.ggml.eos_token_id")
                .ifPresent(value -> {
                    cfg.eosTokenId = value;
                    cfg.eosTokenIds = List.of(value);
                });
        metadataIntOptional(meta, "tokenizer.ggml.padding_token_id")
                .or(() -> metadataIntOptional(meta, "tokenizer.ggml.pad_token_id"))
                .ifPresent(value -> cfg.padTokenId = value);

        List<String> layerTypes = metadataLayerTypes(meta, arch + ".attention.sliding_window_pattern");
        if (!layerTypes.isEmpty()) {
            cfg.layerTypes = layerTypes;
        }
        return cfg;
    }

    /**
     * Check whether a model directory has a config.json.
     */
    public static boolean exists(Path modelDir) {
        return Files.exists(modelDir.resolve("config.json"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Derived / computed accessors
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Primary architecture class name (first entry in the list, or model_type
     * fallback).
     */
    public String primaryArchitecture() {
        if (architectures != null && !architectures.isEmpty())
            return architectures.get(0);
        return modelType != null ? modelType : "unknown";
    }

    /**
     * Accepts either a JSON array or single string for "architectures".
     */
    @JsonSetter("architectures")
    public void setArchitectures(JsonNode node) {
        if (node == null || node.isNull()) {
            this.architectures = null;
            return;
        }
        if (node.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : node) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            this.architectures = out;
            return;
        }
        if (node.isTextual()) {
            this.architectures = List.of(node.asText());
            return;
        }
        // Unknown shape, keep null so we can fall back to model_type.
        this.architectures = null;
    }

    /**
     * Direct setter for runtime overrides.
     */
    public void setArchitectures(List<String> architectures) {
        this.architectures = architectures;
    }

    @JsonSetter("bos_token_id")
    public void setBosTokenId(JsonNode node) {
        this.bosTokenId = firstInt(node);
    }

    @JsonSetter("eos_token_id")
    public void setEosTokenId(JsonNode node) {
        this.eosTokenIds = allInts(node);
        this.eosTokenId = eosTokenIds.isEmpty() ? null : eosTokenIds.get(0);
    }

    @JsonSetter("pad_token_id")
    public void setPadTokenId(JsonNode node) {
        this.padTokenId = firstInt(node);
    }

    @JsonSetter("hidden_activation")
    public void setHiddenActivation(String hiddenActivation) {
        if (hiddenActivation != null && !hiddenActivation.isBlank()) {
            this.hiddenAct = hiddenActivation;
        }
    }

    @JsonSetter("activation")
    public void setActivation(String activation) {
        if (activation != null && !activation.isBlank()) {
            this.hiddenAct = activation;
        }
    }

    @JsonSetter("num_kv_heads")
    public void setNumKvHeads(Integer numKvHeads) {
        if (numKvHeads != null) {
            this.numKeyValueHeads = numKvHeads;
        }
    }

    @JsonSetter("ffn_hidden_size")
    public void setFfnHiddenSize(Integer ffnHiddenSize) {
        if (ffnHiddenSize != null) {
            this.intermediateSize = ffnHiddenSize;
        }
    }

    @JsonSetter("num_experts")
    public void setNumExperts(Integer numExperts) {
        if (numExperts != null) {
            this.numLocalExperts = numExperts;
        }
    }

    @JsonSetter("top_k_experts")
    public void setTopKExperts(Integer topKExperts) {
        if (topKExperts != null) {
            this.numExpertsPerTok = topKExperts;
        }
    }

    @JsonSetter("layer_norm_epsilon")
    public void setLayerNormEpsilon(Double layerNormEpsilon) {
        if (layerNormEpsilon != null) {
            this.layerNormEps = layerNormEpsilon;
        }
    }

    @JsonSetter("dtype")
    public void setDtype(String dtype) {
        if (dtype != null && !dtype.isBlank()) {
            this.torchDtype = dtype;
        }
    }

    private static void mergeRopeParameters(ModelConfig cfg, JsonNode ropeParams) {
        if (cfg == null || ropeParams == null || !ropeParams.isObject()) {
            return;
        }
        JsonNode full = ropeParams.get("full_attention");
        if (full != null && full.isObject()) {
            Double v = doubleValue(full, "rope_theta");
            if (v != null)
                cfg.ropeThetaFull = v;
            Double p = doubleValue(full, "partial_rotary_factor");
            if (p != null)
                cfg.partialRotaryFactorFull = p;
        }
        JsonNode sliding = ropeParams.get("sliding_attention");
        if (sliding != null && sliding.isObject()) {
            Double v = doubleValue(sliding, "rope_theta");
            if (v != null)
                cfg.ropeThetaSliding = v;
            Double p = doubleValue(sliding, "partial_rotary_factor");
            if (p != null)
                cfg.partialRotaryFactorSliding = p;
        }
    }

    private static void reconcileGemma3RopeDefaults(ModelConfig cfg) {
        if (cfg == null || cfg.modelType == null || !cfg.modelType.toLowerCase().startsWith("gemma3")) {
            return;
        }
        if (cfg.ropeThetaFull == null) {
            cfg.ropeThetaFull = cfg.ropeTheta;
        }
        if (cfg.ropeThetaSliding == null) {
            cfg.ropeThetaSliding = cfg.ropeLocalBaseFreq != null ? cfg.ropeLocalBaseFreq : 10000.0;
        }
    }

    private static String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }

    private static List<String> listValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        if (v.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : v) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            return out;
        }
        if (v.isTextual()) {
            return List.of(v.asText());
        }
        return null;
    }

    private static Integer firstInt(JsonNode node) {
        if (node == null || node.isNull()) {
            return null;
        }
        if (node.isNumber()) {
            return node.asInt();
        }
        if (node.isArray() && node.size() > 0) {
            JsonNode first = node.get(0);
            return first != null && first.isNumber() ? first.asInt() : null;
        }
        return null;
    }

    private static Integer intValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asInt() : null;
    }

    private static Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }

    private static Boolean boolValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isBoolean() ? v.asBoolean() : null;
    }

    private static List<Integer> allInts(JsonNode node) {
        if (node == null || node.isNull()) {
            return List.of();
        }
        List<Integer> values = new ArrayList<>();
        if (node.isArray()) {
            for (JsonNode item : node) {
                if (item != null && item.isNumber()) {
                    values.add(item.asInt());
                }
            }
            return values;
        }
        if (node.isNumber()) {
            values.add(node.asInt());
        }
        return values;
    }

    private static String ggufArchitectureClassName(String arch) {
        String normalized = arch == null ? "" : arch.toLowerCase(Locale.ROOT);
        if (normalized.contains("shieldgemma2") || normalized.contains("shield_gemma2")
                || normalized.contains("shield-gemma2")) {
            return "ShieldGemma2ForImageClassification";
        }
        if (normalized.contains("t5gemma2") || normalized.contains("t5_gemma2")
                || normalized.contains("t5-gemma2")) {
            return "T5Gemma2ForConditionalGeneration";
        }
        if (normalized.contains("t5gemma") || normalized.contains("t5_gemma")
                || normalized.contains("t5-gemma")) {
            return "T5GemmaForConditionalGeneration";
        }
        if (normalized.contains("vaultgemma") || normalized.contains("vault_gemma")
                || normalized.contains("vault-gemma")) {
            return "VaultGemmaForCausalLM";
        }
        if (normalized.contains("gemma4_unified") || normalized.contains("gemma4-unified")
                || normalized.contains("gemma-4-unified")) {
            return "Gemma4ForMultimodalLM";
        }
        if (normalized.contains("gemma4_audio") || normalized.contains("gemma4-audio")
                || normalized.contains("gemma-4-audio")) {
            return "Gemma4AudioModel";
        }
        if (normalized.contains("gemma4_vision") || normalized.contains("gemma4-vision")
                || normalized.contains("gemma-4-vision")) {
            return "Gemma4VisionModel";
        }
        if (normalized.contains("gemma4_text") || normalized.contains("gemma4-text")
                || normalized.contains("gemma-4-text")) {
            return "Gemma4ForCausalLM";
        }
        if (normalized.contains("gemma4") || normalized.contains("gemma-4")) {
            return "Gemma4ForConditionalGeneration";
        }
        if (normalized.contains("gemma3n")) {
            return "Gemma3nForConditionalGeneration";
        }
        if (normalized.contains("paligemma") || normalized.contains("pali_gemma")) {
            return "PaliGemmaForConditionalGeneration";
        }
        if (normalized.contains("recurrent_gemma") || normalized.contains("recurrent-gemma")) {
            return "RecurrentGemmaForCausalLM";
        }
        if (normalized.contains("gemma3")) {
            return "Gemma3ForCausalLM";
        }
        if (normalized.contains("gemma2")) {
            return "Gemma2ForCausalLM";
        }
        if (normalized.contains("gemma")) {
            return "GemmaForCausalLM";
        }
        if (normalized.contains("colqwen2")) {
            return "ColQwen2ForRetrieval";
        }
        if (normalized.contains("qwen3_omni_moe") || normalized.contains("qwen3-omni-moe")) {
            return "Qwen3OmniMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5_moe") || normalized.contains("qwen3.5_moe")
                || normalized.contains("qwen3-5-moe") || normalized.contains("qwen3.5-moe")) {
            return "Qwen3_5MoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5") || normalized.contains("qwen3.5")
                || normalized.contains("qwen3-5")) {
            return "Qwen3_5ForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl_moe") || normalized.contains("qwen3-vl-moe")) {
            return "Qwen3VLMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl") || normalized.contains("qwen3-vl")) {
            return "Qwen3VLForConditionalGeneration";
        }
        if (normalized.contains("qwen3_moe") || normalized.contains("qwen3-moe")) {
            return "Qwen3MoeForCausalLM";
        }
        if (normalized.contains("qwen3_next") || normalized.contains("qwen3-next")) {
            return "Qwen3NextForCausalLM";
        }
        if (normalized.contains("qwen3")) {
            return "Qwen3ForCausalLM";
        }
        if (normalized.contains("qwen2_5_omni") || normalized.contains("qwen2.5_omni")
                || normalized.contains("qwen2-5-omni") || normalized.contains("qwen2.5-omni")) {
            return "Qwen2_5OmniForConditionalGeneration";
        }
        if (normalized.contains("qwen2_5_vl") || normalized.contains("qwen2.5_vl")
                || normalized.contains("qwen2-5-vl") || normalized.contains("qwen2.5-vl")) {
            return "Qwen2_5_VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_vl") || normalized.contains("qwen2-vl")) {
            return "Qwen2VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_audio") || normalized.contains("qwen2-audio")) {
            return "Qwen2AudioForConditionalGeneration";
        }
        if (normalized.contains("qwen2_moe") || normalized.contains("qwen2-moe")) {
            return "Qwen2MoeForCausalLM";
        }
        if (normalized.contains("qwen")) {
            return "Qwen2ForCausalLM";
        }
        if (normalized.contains("ministral3") || normalized.contains("ministral_3")
                || normalized.contains("ministral-3")) {
            return "Ministral3ForCausalLM";
        }
        if (normalized.contains("ministral")) {
            return "MinistralForCausalLM";
        }
        if (normalized.contains("mixtral")) {
            return "MixtralForCausalLM";
        }
        if (normalized.contains("mistral4") || normalized.contains("mistral_4")
                || normalized.contains("mistral-4")) {
            return "Mistral4ForCausalLM";
        }
        if (normalized.contains("mistral3") || normalized.contains("mistral_3")
                || normalized.contains("mistral-3")) {
            return "Mistral3ForConditionalGeneration";
        }
        if (normalized.contains("mistral")) {
            return "MistralForCausalLM";
        }
        if (normalized.contains("phi3")) {
            return "Phi3ForCausalLM";
        }
        if (normalized.contains("phi4_multimodal") || normalized.contains("phi4-multimodal")
                || normalized.contains("phi_4_multimodal") || normalized.contains("phi-4-multimodal")) {
            return "Phi4MultimodalForCausalLM";
        }
        if (normalized.contains("phimoe") || normalized.contains("phi_moe")
                || normalized.contains("phi-moe")) {
            return "PhimoeForCausalLM";
        }
        if (normalized.contains("phi")) {
            return "PhiForCausalLM";
        }
        if (normalized.contains("falcon_h1") || normalized.contains("falcon-h1")) {
            return "FalconH1ForCausalLM";
        }
        if (normalized.contains("falcon_mamba") || normalized.contains("falcon-mamba")) {
            return "FalconMambaForCausalLM";
        }
        if (normalized.contains("falcon")) {
            return "FalconForCausalLM";
        }
        if (normalized.contains("bloom")) {
            return "BloomForCausalLM";
        }
        if (normalized.contains("bitnet")) {
            return "BitNetForCausalLM";
        }
        if (normalized.contains("dbrx")) {
            return "DbrxForCausalLM";
        }
        if (normalized.contains("exaone_moe") || normalized.contains("exaone-moe")) {
            return "ExaoneMoeForCausalLM";
        }
        if (normalized.contains("exaone4")) {
            return "Exaone4ForCausalLM";
        }
        if (normalized.contains("bamba")) {
            return "BambaForCausalLM";
        }
        if (normalized.contains("zamba2")) {
            return "Zamba2ForCausalLM";
        }
        if (normalized.contains("zamba")) {
            return "ZambaForCausalLM";
        }
        if (normalized.contains("arcee")) {
            return "ArceeForCausalLM";
        }
        if (normalized.contains("granitemoehybrid")) {
            return "GraniteMoeHybridForCausalLM";
        }
        if (normalized.contains("granitemoeshared")) {
            return "GraniteMoeSharedForCausalLM";
        }
        if (normalized.contains("granitemoe")) {
            return "GraniteMoeForCausalLM";
        }
        if (normalized.contains("granite")) {
            return "GraniteForCausalLM";
        }
        if (normalized.contains("nemotron")) {
            return "NemotronForCausalLM";
        }
        if (normalized.contains("stablelm") || normalized.contains("stable_lm")) {
            return "StableLmForCausalLM";
        }
        if (normalized.contains("persimmon")) {
            return "PersimmonForCausalLM";
        }
        if (normalized.contains("smollm3") || normalized.contains("smol_lm3")
                || normalized.contains("smol-lm3")) {
            return "SmolLM3ForCausalLM";
        }
        if (normalized.contains("xglm")) {
            return "XGLMForCausalLM";
        }
        if (normalized.contains("glm_ocr_vision") || normalized.contains("glm-ocr-vision")) {
            return "GlmOcrVisionModel";
        }
        if (normalized.contains("glm_ocr_text") || normalized.contains("glm-ocr-text")) {
            return "GlmOcrTextModel";
        }
        if (normalized.contains("glm_ocr") || normalized.contains("glm-ocr")) {
            return "GlmOcrForConditionalGeneration";
        }
        if (normalized.equals("glm") || normalized.contains("glm_") || normalized.contains("glm-")) {
            return "GlmForCausalLM";
        }
        if (normalized.contains("starcoder2")) {
            return "Starcoder2ForCausalLM";
        }
        if (normalized.contains("gpt_bigcode")) {
            return "GPTBigCodeForCausalLM";
        }
        if (normalized.contains("gpt_oss") || normalized.contains("gpt-oss")) {
            return "GptOssForCausalLM";
        }
        if (normalized.contains("codegen")) {
            return "CodeGenForCausalLM";
        }
        if (normalized.contains("biogpt") || normalized.contains("bio_gpt")) {
            return "BioGptForCausalLM";
        }
        if (normalized.equals("opt") || normalized.contains("facebook_opt")) {
            return "OPTForCausalLM";
        }
        if (normalized.contains("ctrl")) {
            return "CTRLLMHeadModel";
        }
        if (normalized.contains("cpmant") || normalized.contains("cpm-ant")) {
            return "CpmAntForCausalLM";
        }
        if (normalized.equals("cpm") || normalized.contains("cpm_") || normalized.contains("cpm-")) {
            return "CpmForCausalLM";
        }
        if (normalized.contains("gpt_neox_japanese") || normalized.contains("gpt-neox-japanese")
                || normalized.contains("gptneox_japanese")) {
            return "GPTNeoXJapaneseForCausalLM";
        }
        if (normalized.contains("gpt_neox") || normalized.contains("gptneox")) {
            return "GPTNeoXForCausalLM";
        }
        if (normalized.contains("gptj") || normalized.contains("gpt_j")) {
            return "GPTJForCausalLM";
        }
        if (normalized.contains("gpt_neo") || normalized.contains("gpt-neo")
                || normalized.contains("gptneo")) {
            return "GPTNeoForCausalLM";
        }
        if (normalized.contains("openai_gpt") || normalized.contains("openai-gpt")) {
            return "OpenAIGPTLMHeadModel";
        }
        if (normalized.contains("dialogpt")) {
            return "GPT2LMHeadModel";
        }
        if (normalized.contains("megatron_gpt2") || normalized.contains("megatron-gpt2")) {
            return "GPT2LMHeadModel";
        }
        if (normalized.contains("gpt2")) {
            return "GPT2LMHeadModel";
        }
        if (normalized.contains("mpt")) {
            return "MptForCausalLM";
        }
        if (normalized.contains("mamba2")) {
            return "Mamba2ForCausalLM";
        }
        if (normalized.contains("mamba")) {
            return "MambaForCausalLM";
        }
        if (normalized.contains("rwkv")) {
            return "RwkvForCausalLM";
        }
        if (normalized.contains("jamba")) {
            return "JambaForCausalLM";
        }
        if (normalized.contains("olmo_hybrid") || normalized.contains("olmo-hybrid")) {
            return "OlmoHybridForCausalLM";
        }
        if (normalized.contains("olmoe") || normalized.contains("olmo_moe")
                || normalized.contains("olmo-moe")) {
            return "OlmoeForCausalLM";
        }
        if (normalized.contains("olmo3")) {
            return "Olmo3ForCausalLM";
        }
        if (normalized.contains("olmo2")) {
            return "Olmo2ForCausalLM";
        }
        if (normalized.contains("olmo")) {
            return "OlmoForCausalLM";
        }
        if (normalized.contains("plbart")) {
            return "PLBartForConditionalGeneration";
        }
        if (normalized.contains("mbart")) {
            return "MBartForConditionalGeneration";
        }
        if (normalized.contains("barthez")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("bartpho")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("blenderbot_small") || normalized.contains("blenderbot-small")) {
            return "BlenderbotSmallForConditionalGeneration";
        }
        if (normalized.contains("blenderbot")) {
            return "BlenderbotForConditionalGeneration";
        }
        if (normalized.contains("bart")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("marian")) {
            return "MarianMTModel";
        }
        if (normalized.contains("fsmt")) {
            return "FSMTForConditionalGeneration";
        }
        if (normalized.contains("m2m_100") || normalized.contains("m2m100")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("nllb-moe") || normalized.contains("nllb_moe")) {
            return "NllbMoeForConditionalGeneration";
        }
        if (normalized.contains("nllb")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("bigbird_pegasus") || normalized.contains("bigbird-pegasus")) {
            return "BigBirdPegasusForConditionalGeneration";
        }
        if (normalized.contains("big_bird") || normalized.contains("bigbird")) {
            return "BigBirdModel";
        }
        if (normalized.contains("pegasus_x") || normalized.contains("pegasus-x")) {
            return "PegasusXForConditionalGeneration";
        }
        if (normalized.contains("pegasus")) {
            return "PegasusForConditionalGeneration";
        }
        if (normalized.contains("prophetnet") || normalized.contains("prophet_net")) {
            return "ProphetNetForConditionalGeneration";
        }
        if (normalized.contains("speecht5") || normalized.contains("speech_t5")) {
            return "SpeechT5Model";
        }
        if (normalized.contains("switch_transformers") || normalized.contains("switch-transformers")) {
            return "SwitchTransformersForConditionalGeneration";
        }
        if (normalized.contains("longt5")) {
            return "LongT5ForConditionalGeneration";
        }
        if (normalized.contains("umt5")) {
            return "UMT5ForConditionalGeneration";
        }
        if (normalized.contains("mt5")) {
            return "MT5ForConditionalGeneration";
        }
        if (normalized.contains("byt5")) {
            return "T5ForConditionalGeneration";
        }
        if (normalized.contains("t5")) {
            return "T5ForConditionalGeneration";
        }
        if (normalized.contains("bark")) {
            return "BarkModel";
        }
        if (normalized.contains("musicgen_decoder")) {
            return "MusicgenForCausalLM";
        }
        if (normalized.contains("musicgen")) {
            return "MusicgenForConditionalGeneration";
        }
        if (normalized.contains("seamless_m4t_v2") || normalized.contains("seamless-m4t-v2")) {
            return "SeamlessM4Tv2Model";
        }
        if (normalized.contains("seamless_m4t") || normalized.contains("seamless-m4t")) {
            return "SeamlessM4TModel";
        }
        if (normalized.contains("wav2vec2_conformer") || normalized.contains("wav2vec2-conformer")) {
            return "Wav2Vec2ConformerModel";
        }
        if (normalized.contains("wav2vec2")) {
            return "Wav2Vec2Model";
        }
        if (normalized.contains("hubert")) {
            return "HubertModel";
        }
        if (normalized.contains("wavlm")) {
            return "WavLMModel";
        }
        if (normalized.contains("encodec")) {
            return "EncodecModel";
        }
        if (normalized.contains("clap")) {
            return "ClapModel";
        }
        if (normalized.contains("whisper")) {
            return "WhisperForConditionalGeneration";
        }
        if (normalized.contains("rag")) {
            return "RagModel";
        }
        if (normalized.contains("dpr")) {
            return "DPRQuestionEncoder";
        }
        if (normalized.contains("lighton_ocr") || normalized.contains("lighton-ocr")) {
            return "LightOnOcrForConditionalGeneration";
        }
        if (normalized.contains("got_ocr2") || normalized.contains("got-ocr2")) {
            return "GotOcr2ForConditionalGeneration";
        }
        if (normalized.contains("video_llama_3") || normalized.contains("video-llama-3")) {
            return "VideoLlama3ForConditionalGeneration";
        }
        if (normalized.contains("llama4") || normalized.contains("llama-4")) {
            return "Llama4ForConditionalGeneration";
        }
        if (normalized.contains("mllama")) {
            return "MllamaForConditionalGeneration";
        }
        if (normalized.contains("diffllama") || normalized.contains("diff_llama")
                || normalized.contains("diff-llama")) {
            return "DiffLlamaForCausalLM";
        }
        if (normalized.contains("code_llama") || normalized.contains("code-llama")
                || normalized.contains("codellama")) {
            return "LlamaForCausalLM";
        }
        if (normalized.contains("pixtral")) {
            return "PixtralVisionModel";
        }
        if (normalized.contains("fuyu")) {
            return "FuyuForCausalLM";
        }
        if (normalized.contains("kosmos2_5") || normalized.contains("kosmos2.5")
                || normalized.contains("kosmos-2.5") || normalized.contains("kosmos_2_5")) {
            return "Kosmos2_5ForConditionalGeneration";
        }
        if (normalized.contains("kosmos-2") || normalized.contains("kosmos2")
                || normalized.contains("kosmos_2")) {
            return "Kosmos2ForConditionalGeneration";
        }
        if (normalized.contains("instructblipvideo")) {
            return "InstructBlipVideoForConditionalGeneration";
        }
        if (normalized.contains("instructblip")) {
            return "InstructBlipForConditionalGeneration";
        }
        if (normalized.contains("video_llava") || normalized.contains("video-llava")) {
            return "VideoLlavaForConditionalGeneration";
        }
        if (normalized.contains("vipllava") || normalized.contains("vip_llava")) {
            return "VipLlavaForConditionalGeneration";
        }
        if (normalized.contains("llava_next_video") || normalized.contains("llava-next-video")) {
            return "LlavaNextVideoForConditionalGeneration";
        }
        if (normalized.contains("llava_next") || normalized.contains("llava-next")) {
            return "LlavaNextForConditionalGeneration";
        }
        if (normalized.contains("llava_onevision") || normalized.contains("llava-onevision")) {
            return "LlavaOnevisionForConditionalGeneration";
        }
        if (normalized.contains("llava")) {
            return "LlavaForConditionalGeneration";
        }
        if (normalized.contains("florence2") || normalized.contains("florence_vision")) {
            return "Florence2ForConditionalGeneration";
        }
        if (normalized.contains("bridgetower")) {
            return "BridgeTowerModel";
        }
        if (normalized.contains("align_vision_model") || normalized.contains("align-vision-model")) {
            return "AlignVisionModel";
        }
        if (normalized.contains("align_text_model") || normalized.contains("align-text-model")) {
            return "AlignTextModel";
        }
        if (normalized.equals("align") || normalized.contains("align_")
                || normalized.contains("align-")) {
            return "AlignModel";
        }
        if (normalized.contains("visual_bert") || normalized.contains("visual-bert")) {
            return "VisualBertModel";
        }
        if (normalized.contains("lxmert")) {
            return "LxmertModel";
        }
        if (normalized.contains("vilt")) {
            return "ViltModel";
        }
        if (normalized.contains("layoutxlm")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlmv3")) {
            return "LayoutLMv3Model";
        }
        if (normalized.contains("layoutlmv2")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlm")) {
            return "LayoutLMModel";
        }
        if (normalized.contains("markuplm") || normalized.contains("markup_lm")) {
            return "MarkupLMModel";
        }
        if (normalized.equals("bros") || normalized.contains("bros_") || normalized.contains("bros-")) {
            return "BrosModel";
        }
        if (normalized.equals("lilt") || normalized.contains("lilt_") || normalized.contains("lilt-")) {
            return "LiltModel";
        }
        if (normalized.contains("mgp_str") || normalized.contains("mgp-str")) {
            return "MgpstrForSceneTextRecognition";
        }
        if (normalized.contains("trocr")) {
            return "TrOCRForCausalLM";
        }
        if (normalized.contains("pix2struct")) {
            return "Pix2StructForConditionalGeneration";
        }
        if (normalized.contains("nougat")) {
            return "VisionEncoderDecoderModel";
        }
        if (normalized.contains("internvl")) {
            return "InternVLForConditionalGeneration";
        }
        if (normalized.contains("smolvlm") || normalized.contains("smol_vlm")) {
            return "SmolVLMForConditionalGeneration";
        }
        if (normalized.contains("fast_vlm") || normalized.contains("fast-vlm")) {
            return "FastVlmForConditionalGeneration";
        }
        if (normalized.contains("idefics3")) {
            return "Idefics3ForConditionalGeneration";
        }
        if (normalized.contains("idefics2")) {
            return "Idefics2ForConditionalGeneration";
        }
        if (normalized.contains("idefics")) {
            return "IdeficsForVisionText2Text";
        }
        if (normalized.contains("chameleon")) {
            return "ChameleonForConditionalGeneration";
        }
        if (normalized.contains("colmodernvbert")) {
            return "ColModernVBertForRetrieval";
        }
        if (normalized.contains("colpali")) {
            return "ColPaliForRetrieval";
        }
        if (normalized.contains("blip_2") || normalized.contains("blip-2") || normalized.contains("blip2")) {
            return "Blip2ForConditionalGeneration";
        }
        if (normalized.contains("blip")) {
            return "BlipForConditionalGeneration";
        }
        if (normalized.contains("siglip2")) {
            return "Siglip2Model";
        }
        if (normalized.contains("siglip")) {
            return "SiglipModel";
        }
        if (normalized.contains("sam_hq") || normalized.contains("sam-hq")) {
            return "SamHQModel";
        }
        if (normalized.contains("sam2")) {
            return "Sam2Model";
        }
        if (normalized.equals("sam") || normalized.contains("segment_anything")) {
            return "SamModel";
        }
        if (normalized.contains("clipseg")) {
            return "CLIPSegForImageSegmentation";
        }
        if (normalized.contains("metaclip_2") || normalized.contains("metaclip-2")) {
            return "MetaClip2Model";
        }
        if (normalized.contains("vision_text_dual_encoder")
                || normalized.contains("vision-text-dual-encoder")) {
            return "VisionTextDualEncoderModel";
        }
        if (normalized.contains("x_clip") || normalized.contains("x-clip")
                || normalized.contains("xclip")) {
            return "XCLIPModel";
        }
        if (normalized.contains("altclip_vision_model") || normalized.contains("altclip-vision-model")) {
            return "AltCLIPVisionModel";
        }
        if (normalized.contains("altclip_text_model") || normalized.contains("altclip-text-model")) {
            return "AltCLIPTextModel";
        }
        if (normalized.contains("altclip")) {
            return "AltCLIPModel";
        }
        if (normalized.contains("chinese_clip_vision_model")
                || normalized.contains("chinese-clip-vision-model")) {
            return "ChineseCLIPVisionModel";
        }
        if (normalized.contains("chinese_clip_text_model")
                || normalized.contains("chinese-clip-text-model")) {
            return "ChineseCLIPTextModel";
        }
        if (normalized.contains("chinese_clip") || normalized.contains("chinese-clip")) {
            return "ChineseCLIPModel";
        }
        if (normalized.contains("clip")) {
            return "CLIPModel";
        }
        if (normalized.contains("groupvit_vision_model")
                || normalized.contains("groupvit-vision-model")) {
            return "GroupViTVisionModel";
        }
        if (normalized.contains("groupvit_text_model") || normalized.contains("groupvit-text-model")) {
            return "GroupViTTextModel";
        }
        if (normalized.contains("groupvit")) {
            return "GroupViTModel";
        }
        if (normalized.contains("owlv2") || normalized.contains("owl_v2")
                || normalized.contains("owl-v2")) {
            return "Owlv2ForObjectDetection";
        }
        if (normalized.contains("owlvit") || normalized.contains("owl_vit")
                || normalized.contains("owl-vit")) {
            return "OwlViTForObjectDetection";
        }
        if (normalized.contains("mm_grounding_dino") || normalized.contains("mm-grounding-dino")) {
            return "MMGroundingDinoForObjectDetection";
        }
        if (normalized.contains("grounding_dino") || normalized.contains("grounding-dino")) {
            return "GroundingDinoForObjectDetection";
        }
        if (normalized.contains("mask2former") || normalized.contains("mask2-former")) {
            return "Mask2FormerForUniversalSegmentation";
        }
        if (normalized.contains("maskformer")) {
            return "MaskFormerForInstanceSegmentation";
        }
        if (normalized.contains("oneformer")) {
            return "OneFormerForUniversalSegmentation";
        }
        if (normalized.contains("upernet")) {
            return "UperNetForSemanticSegmentation";
        }
        if (normalized.contains("yolos")) {
            return "YolosForObjectDetection";
        }
        if (normalized.contains("data2vec_audio") || normalized.contains("data2vec-audio")) {
            return "Data2VecAudioModel";
        }
        if (normalized.contains("data2vec_vision") || normalized.contains("data2vec-vision")) {
            return "Data2VecVisionModel";
        }
        if (normalized.contains("data2vec_text") || normalized.contains("data2vec-text")) {
            return "Data2VecTextModel";
        }
        if (normalized.contains("depth_anything") || normalized.contains("depth-anything")) {
            return "DepthAnythingForDepthEstimation";
        }
        if (normalized.contains("depth_pro") || normalized.contains("depth-pro")) {
            return "DepthProForDepthEstimation";
        }
        if (normalized.contains("zoedepth") || normalized.contains("zoe_depth")) {
            return "ZoeDepthForDepthEstimation";
        }
        if (normalized.equals("dpt") || normalized.contains("dpt_") || normalized.contains("dpt-")) {
            return "DPTForDepthEstimation";
        }
        if (normalized.contains("segformer")) {
            return "SegformerModel";
        }
        if (normalized.contains("beit")) {
            return "BeitModel";
        }
        if (normalized.contains("dinov2_with_registers")
                || normalized.contains("dinov2-with-registers")) {
            return "Dinov2WithRegistersModel";
        }
        if (normalized.contains("dinov2")) {
            return "Dinov2Model";
        }
        if (normalized.contains("deit")) {
            return "DeiTModel";
        }
        if (normalized.contains("focalnet")) {
            return "FocalNetModel";
        }
        if (normalized.equals("cvt") || normalized.contains("cvt_") || normalized.contains("cvt-")) {
            return "CvtModel";
        }
        if (normalized.contains("levit")) {
            return "LevitModel";
        }
        if (normalized.contains("mobilevitv2") || normalized.contains("mobilevit_v2")
                || normalized.contains("mobilevit-v2")) {
            return "MobileViTV2Model";
        }
        if (normalized.contains("mobilevit")) {
            return "MobileViTModel";
        }
        if (normalized.contains("poolformer")) {
            return "PoolFormerModel";
        }
        if (normalized.contains("pvt_v2") || normalized.contains("pvt-v2")) {
            return "PvtV2Model";
        }
        if (normalized.equals("pvt") || normalized.contains("pvt_") || normalized.contains("pvt-")) {
            return "PvtModel";
        }
        if (normalized.contains("regnet")) {
            return "RegNetModel";
        }
        if (normalized.contains("vits")) {
            return "VitsModel";
        }
        if (normalized.contains("vit")) {
            return "ViTModel";
        }
        if (normalized.contains("donut")) {
            return "DonutSwinModel";
        }
        if (normalized.contains("swinv2") || normalized.contains("swin_v2")) {
            return "Swinv2Model";
        }
        if (normalized.contains("swin")) {
            return "SwinModel";
        }
        if (normalized.contains("convnextv2") || normalized.contains("convnext_v2")) {
            return "ConvNextV2Model";
        }
        if (normalized.contains("convnext")) {
            return "ConvNextModel";
        }
        if (normalized.contains("efficientnet")) {
            return "EfficientNetModel";
        }
        if (normalized.contains("mobilenet_v2") || normalized.contains("mobilenet-v2")) {
            return "MobileNetV2Model";
        }
        if (normalized.contains("mobilenet_v1") || normalized.contains("mobilenet-v1")) {
            return "MobileNetV1Model";
        }
        if (normalized.contains("resnet")) {
            return "ResNetModel";
        }
        if (normalized.contains("deformable_detr") || normalized.contains("deformable-detr")) {
            return "DeformableDetrModel";
        }
        if (normalized.contains("conditional_detr") || normalized.contains("conditional-detr")) {
            return "ConditionalDetrModel";
        }
        if (normalized.contains("rt_detr") || normalized.contains("rt-detr")) {
            return "RTDetrModel";
        }
        if (normalized.contains("detr")) {
            return "DetrModel";
        }
        if (normalized.equals("led") || normalized.contains("led_") || normalized.contains("led-")) {
            return "LEDForConditionalGeneration";
        }
        if (normalized.contains("longformer")) {
            return "LongformerModel";
        }
        if (normalized.contains("reformer")) {
            return "ReformerModelWithLMHead";
        }
        if (normalized.contains("luke")) {
            return "LukeModel";
        }
        if (normalized.contains("funnel")) {
            return "FunnelModel";
        }
        if (normalized.contains("fnet")) {
            return "FNetModel";
        }
        if (normalized.contains("esmfold")) {
            return "EsmForProteinFolding";
        }
        if (normalized.equals("esm") || normalized.contains("esm_") || normalized.contains("esm-")) {
            return "EsmModel";
        }
        if (normalized.contains("deberta_v2") || normalized.contains("deberta-v2")) {
            return "DebertaV2Model";
        }
        if (normalized.contains("deberta")) {
            return "DebertaModel";
        }
        if (normalized.contains("xlm_roberta_xl") || normalized.contains("xlm-roberta-xl")) {
            return "XLMRobertaXLModel";
        }
        if (normalized.contains("xlm_roberta") || normalized.contains("xlm-roberta")) {
            return "XLMRobertaModel";
        }
        if (normalized.contains("jina_embeddings_v3") || normalized.contains("jina-embeddings-v3")) {
            return "JinaEmbeddingsV3Model";
        }
        if (normalized.contains("camembert")) {
            return "CamembertModel";
        }
        if (normalized.contains("flaubert")) {
            return "FlaubertModel";
        }
        if (normalized.contains("mpnet")) {
            return "MPNetModel";
        }
        if (normalized.contains("xlnet")) {
            return "XLNetModel";
        }
        if (normalized.equals("xmod") || normalized.contains("xmod_") || normalized.contains("xmod-")) {
            return "XmodModel";
        }
        if (normalized.equals("xlm") || normalized.contains("xlm_") || normalized.contains("xlm-")) {
            return "XLMModel";
        }
        if (normalized.contains("electra")) {
            return "ElectraModel";
        }
        if (normalized.contains("albert")) {
            return "AlbertModel";
        }
        if (normalized.contains("roberta_prelayernorm")
                || normalized.contains("roberta-prelayernorm")
                || normalized.contains("roberta_pre_layer_norm")
                || normalized.contains("roberta-pre-layer-norm")) {
            return "RobertaPreLayerNormModel";
        }
        if (normalized.contains("roberta")) {
            return "RobertaModel";
        }
        if (normalized.contains("roformer")) {
            return "RoFormerModel";
        }
        if (normalized.contains("phobert") || normalized.contains("pho_bert")
                || normalized.contains("pho-bert")) {
            return "RobertaModel";
        }
        if (normalized.contains("deepseek_vl_hybrid") || normalized.contains("deepseek-vl-hybrid")) {
            return "DeepseekVLHybridForConditionalGeneration";
        }
        if (normalized.contains("deepseek_vl") || normalized.contains("deepseek-vl")) {
            return "DeepseekVLForConditionalGeneration";
        }
        if (normalized.contains("deepseek_v2") || normalized.contains("deepseek-v2")) {
            return "DeepseekV2ForCausalLM";
        }
        if (normalized.contains("deepseek_v3") || normalized.contains("deepseek-v3")
                || normalized.contains("deepseek_moe") || normalized.contains("deepseek-moe")
                || normalized.contains("deepseek_r1") || normalized.contains("deepseek-r1")
                || normalized.contains("deepseek")) {
            return "DeepseekV3ForCausalLM";
        }
        if (normalized.contains("cohere")) {
            return "Cohere2ForCausalLM";
        }
        if (normalized.contains("kimi")) {
            return "KimiForCausalLM";
        }
        if (normalized.equals("yi") || normalized.contains("yi_")) {
            return "YiForCausalLM";
        }
        if (normalized.contains("modernbert_decoder")
                || normalized.contains("modernbert-decoder")
                || normalized.contains("modern_bert_decoder")) {
            return "ModernBertDecoderForCausalLM";
        }
        if (normalized.contains("modernbert") || normalized.contains("modern_bert")) {
            return "ModernBertModel";
        }
        if (normalized.contains("modernvbert") || normalized.contains("modern_vbert")
                || normalized.contains("modern-vbert")) {
            return "ModernVBertForMaskedLM";
        }
        if (normalized.contains("nomic_bert") || normalized.contains("nomic-bert")) {
            return "NomicBertModel";
        }
        if (normalized.contains("eurobert") || normalized.contains("euro_bert")) {
            return "EuroBertModel";
        }
        if (normalized.contains("distilbert") || normalized.contains("distil_bert")) {
            return "DistilBertModel";
        }
        if (normalized.contains("rembert") || normalized.contains("rem_bert")) {
            return "RemBertModel";
        }
        if (normalized.contains("mobilebert") || normalized.contains("mobile_bert")) {
            return "MobileBertModel";
        }
        if (normalized.contains("megatron_bert") || normalized.contains("megatron-bert")) {
            return "MegatronBertModel";
        }
        if (normalized.contains("squeezebert") || normalized.contains("squeeze_bert")) {
            return "SqueezeBertModel";
        }
        if (normalized.contains("bert_japanese") || normalized.contains("bert-japanese")) {
            return "BertModel";
        }
        if (normalized.contains("bert_generation") || normalized.contains("bert-generation")) {
            return "BertGenerationDecoder";
        }
        if (normalized.contains("bertweet") || normalized.contains("bert_tweet")) {
            return "RobertaModel";
        }
        if (normalized.contains("herbert") || normalized.contains("her_bert")) {
            return "RobertaModel";
        }
        if (normalized.contains("ibert") || normalized.contains("i_bert")) {
            return "IBertModel";
        }
        if (normalized.contains("convbert") || normalized.contains("conv_bert")) {
            return "ConvBertModel";
        }
        if (normalized.contains("roc_bert") || normalized.contains("roc-bert")) {
            return "RoCBertForMaskedLM";
        }
        if (normalized.contains("wav2vec2_bert") || normalized.contains("wav2vec2-bert")) {
            return "Wav2Vec2BertForCTC";
        }
        if (normalized.contains("ernie")) {
            return "ErnieModel";
        }
        if (normalized.contains("canine")) {
            return "CanineModel";
        }
        if (normalized.contains("bert")) {
            return "BertModel";
        }
        if (normalized.contains("flava")) {
            return "FlavaModel";
        }
        return "LlamaForCausalLM";
    }

    private static Optional<String> metadataString(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return Optional.empty();
        }
        String text = value.toString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }

    private static int metadataInt(Map<String, Object> metadata, int fallback, String... keys) {
        return metadataIntOptional(metadata, keys).orElse(fallback);
    }

    private static Optional<Integer> metadataIntOptional(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = firstNumericInt(metadata.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private static int metadataMaxInt(Map<String, Object> metadata, int fallback, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = maxNumericInt(metadata.get(key));
            if (value.isPresent()) {
                return value.get();
            }
        }
        return fallback;
    }

    private static double metadataDouble(Map<String, Object> metadata, double fallback, String... keys) {
        return metadataDoubleOptional(metadata, keys).orElse(fallback);
    }

    private static Optional<Double> metadataDoubleOptional(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Optional<Double> value = firstNumericDouble(metadata.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private static Optional<Integer> metadataTokenCount(Map<String, Object> metadata) {
        Object tokens = metadata.get("tokenizer.ggml.tokens");
        if (tokens instanceof List<?> list) {
            return list.isEmpty() ? Optional.empty() : Optional.of(list.size());
        }
        if (tokens != null && tokens.getClass().isArray()) {
            return Optional.of(java.lang.reflect.Array.getLength(tokens));
        }
        return Optional.empty();
    }

    private static Optional<Integer> firstNumericInt(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        if (value instanceof List<?> list) {
            for (Object item : list) {
                Optional<Integer> numeric = firstNumericInt(item);
                if (numeric.isPresent()) {
                    return numeric;
                }
            }
        }
        return Optional.empty();
    }

    private static Optional<Integer> maxNumericInt(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        if (value instanceof List<?> list) {
            Integer max = null;
            for (Object item : list) {
                Optional<Integer> numeric = firstNumericInt(item);
                if (numeric.isPresent()) {
                    max = max == null ? numeric.get() : Math.max(max, numeric.get());
                }
            }
            return Optional.ofNullable(max);
        }
        return Optional.empty();
    }

    private static Optional<Double> firstNumericDouble(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.doubleValue());
        }
        if (value instanceof List<?> list) {
            for (Object item : list) {
                Optional<Double> numeric = firstNumericDouble(item);
                if (numeric.isPresent()) {
                    return numeric;
                }
            }
        }
        return Optional.empty();
    }

    private static List<String> metadataLayerTypes(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return List.of();
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object item : list) {
            if (item instanceof Boolean sliding) {
                out.add(sliding ? "sliding_attention" : "full_attention");
            }
        }
        return out.size() == list.size() ? List.copyOf(out) : List.of();
    }

    /** Resolved number of KV heads — defaults to numAttentionHeads if not set. */
    public int resolvedNumKvHeads() {
        return numKeyValueHeads != null ? numKeyValueHeads : numAttentionHeads;
    }

    public int resolvedNumGlobalKvHeads() {
        return numGlobalKeyValueHeads != null ? numGlobalKeyValueHeads : resolvedNumKvHeads();
    }

    public int resolvedNumKvHeadsForLayer(int layerIdx) {
        return usesAlternativeAttentionForLayer(layerIdx) ? resolvedNumGlobalKvHeads() : resolvedNumKvHeads();
    }

    public int resolvedMaxKvHeads() {
        return Math.max(resolvedNumKvHeads(), resolvedNumGlobalKvHeads());
    }

    /** Per-head dimension: {@code hiddenSize / numAttentionHeads}. */
    public int resolvedHeadDim() {
        return headDim != null ? headDim : (numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0);
    }

    /**
     * Per-head dimension for a specific layer, including Gemma-4 global-attention
     * overrides.
     */
    public int resolvedHeadDimForLayer(int layerIdx) {
        return usesAlternativeAttentionForLayer(layerIdx) && globalHeadDim != null
                ? globalHeadDim
                : resolvedHeadDim();
    }

    /**
     * Maximum head dimension across all layers (for heterogeneous architectures
     * like Gemma 4).
     */
    public int resolvedMaxHeadDim() {
        int base = resolvedHeadDim();
        return globalHeadDim != null ? Math.max(base, globalHeadDim) : base;
    }

    public int resolvedNumKvSharedLayers() {
        return numKvSharedLayers != null ? Math.max(0, numKvSharedLayers) : 0;
    }

    public boolean usesSharedKvCache(int layerIdx) {
        String modelTypeLower = modelType != null ? modelType.toLowerCase() : "";
        if (modelTypeLower.startsWith("gemma4")
                && Boolean.getBoolean(DISABLE_GEMMA4_SHARED_KV_PROPERTY)) {
            return false;
        }
        int sharedLayers = resolvedNumKvSharedLayers();
        if (sharedLayers <= 0 || layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        return layerIdx >= Math.max(0, numHiddenLayers - sharedLayers);
    }

    public int sharedKvSourceLayer(int layerIdx) {
        if (!usesSharedKvCache(layerIdx)) {
            return layerIdx;
        }
        int sharedStart = Math.max(0, numHiddenLayers - resolvedNumKvSharedLayers());
        String type = layerType(layerIdx);
        for (int prev = sharedStart - 1; prev >= 0; prev--) {
            if (type == null || type.equals(layerType(prev))) {
                return prev;
            }
        }
        return Math.max(0, sharedStart - 1);
    }

    public boolean isSharedKvSourceLayer(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        int sharedStart = Math.max(0, numHiddenLayers - resolvedNumKvSharedLayers());
        if (sharedStart <= 0 || layerIdx >= sharedStart) {
            return false;
        }
        String type = layerType(layerIdx);
        for (int sharedLayer = sharedStart; sharedLayer < numHiddenLayers; sharedLayer++) {
            if ((type == null || type.equals(layerType(sharedLayer)))
                    && sharedKvSourceLayer(sharedLayer) == layerIdx) {
                return true;
            }
        }
        return false;
    }

    /**
     * GQA group size: number of query heads sharing one KV head.
     * 1 = MHA, N = GQA/MQA.
     */
    public int kvGroupSize() {
        return numAttentionHeads / resolvedNumKvHeads();
    }

    /** Whether this model uses Grouped-Query Attention (GQA or MQA). */
    public boolean isGroupedQueryAttention() {
        return resolvedNumKvHeads() < numAttentionHeads;
    }

    public boolean attentionKeyEqualsValue() {
        return Boolean.TRUE.equals(attentionKeyEqualsValue);
    }

    public boolean usesAlternativeAttentionForLayer(int layerIdx) {
        return attentionKeyEqualsValue() && !isSlidingAttentionLayer(layerIdx);
    }

    public Optional<Integer> bosTokenId() {
        return Optional.ofNullable(bosTokenId);
    }

    public Optional<Integer> eosTokenId() {
        return Optional.ofNullable(eosTokenId);
    }

    public List<Integer> eosTokenIds() {
        if (eosTokenIds == null || eosTokenIds.isEmpty()) {
            return eosTokenId != null ? List.of(eosTokenId) : List.of();
        }
        return List.copyOf(eosTokenIds);
    }

    public Optional<Integer> padTokenId() {
        return Optional.ofNullable(padTokenId);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Overrides (runtime reconciliation)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Override the number of attention heads when config.json is inconsistent with
     * weights.
     */
    public void overrideNumAttentionHeads(int heads) {
        if (heads > 0) {
            this.numAttentionHeads = heads;
        }
    }

    /**
     * Override hidden size when config.json is unavailable or reconciled from
     * weights.
     */
    public void overrideHiddenSize(int size) {
        if (size > 0) {
            this.hiddenSize = size;
        }
    }

    /**
     * Override layer count when config.json is unavailable or reconciled from
     * weights.
     */
    public void overrideNumHiddenLayers(int layers) {
        if (layers > 0) {
            this.numHiddenLayers = layers;
        }
    }

    /**
     * Override FFN intermediate size when config.json is unavailable or reconciled
     * from weights.
     */
    public void overrideIntermediateSize(int size) {
        if (size > 0) {
            this.intermediateSize = size;
        }
    }

    /**
     * Override the number of key-value heads when config.json is inconsistent with
     * weights.
     */
    public void overrideNumKeyValueHeads(Integer heads) {
        if (heads != null && heads > 0) {
            this.numKeyValueHeads = heads;
        }
    }

    /** Override head dimension when config.json is inconsistent with weights. */
    public void overrideHeadDim(Integer dim) {
        if (dim != null && dim > 0) {
            this.headDim = dim;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plain getters (used by JSON deserialisation and compute kernels)
    // ─────────────────────────────────────────────────────────────────────────

    public String modelType() {
        return modelType;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int numHiddenLayers() {
        return numHiddenLayers;
    }

    public int numAttentionHeads() {
        return numAttentionHeads;
    }

    public int intermediateSize() {
        return intermediateSize;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int maxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public int hiddenSizePerLayerInput() {
        return hiddenSizePerLayerInput;
    }

    public int vocabSizePerLayerInput() {
        return vocabSizePerLayerInput;
    }

    public String layerType(int layerIdx) {
        if (layerTypes == null || layerIdx < 0 || layerIdx >= layerTypes.size()) {
            return null;
        }
        return layerTypes.get(layerIdx);
    }

    public double ropeThetaForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && ropeThetaFull != null) {
            return ropeThetaFull;
        }
        if ("sliding_attention".equals(type) && ropeThetaSliding != null) {
            return ropeThetaSliding;
        }
        return ropeTheta;
    }

    public double partialRotaryFactorForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && partialRotaryFactorFull != null) {
            return partialRotaryFactorFull;
        }
        if ("sliding_attention".equals(type) && partialRotaryFactorSliding != null) {
            return partialRotaryFactorSliding;
        }
        if (partialRotaryFactor != null && partialRotaryFactor > 0.0) {
            return partialRotaryFactor;
        }
        return 1.0;
    }

    public boolean isSlidingAttentionLayer(int layerIdx) {
        return "sliding_attention".equals(layerType(layerIdx));
    }

    public Double finalLogitSoftcapping() {
        return finalLogitSoftcapping;
    }

    public Double attnLogitSoftcapping() {
        return attnLogitSoftcapping;
    }

    public double queryPreAttnScalar() {
        return queryPreAttnScalar != null && queryPreAttnScalar > 0
                ? queryPreAttnScalar
                : Math.max(1, resolvedHeadDim());
    }

    public double rmsNormEps() {
        return rmsNormEps;
    }

    public double layerNormEps() {
        return layerNormEps;
    }

    public double ropeTheta() {
        return ropeTheta;
    }

    public RopeScaling ropeScaling() {
        return ropeScaling;
    }

    public String hiddenAct() {
        return hiddenAct;
    }

    public String torchDtype() {
        return torchDtype;
    }

    public boolean tieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    public List<String> architectures() {
        return architectures;
    }

    // JavaBean-style compatibility getters
    public String getModelType() { return modelType(); }
    public String getPrimaryArchitecture() { return primaryArchitecture(); }
    public int getHiddenSize() { return hiddenSize(); }
    public int getNumHiddenLayers() { return numHiddenLayers(); }
    public int getNumAttentionHeads() { return numAttentionHeads(); }
    public int getIntermediateSize() { return intermediateSize(); }
    public int getVocabSize() { return vocabSize(); }
    public int getMaxPositionEmbeddings() { return maxPositionEmbeddings(); }
    public int getHiddenSizePerLayerInput() { return hiddenSizePerLayerInput(); }
    public int getVocabSizePerLayerInput() { return vocabSizePerLayerInput(); }
    public String getLayerType(int layerIdx) { return layerType(layerIdx); }
    public Double getFinalLogitSoftcapping() { return finalLogitSoftcapping(); }
    public Double getAttnLogitSoftcapping() { return attnLogitSoftcapping(); }
    public int getSlidingWindowSize() { return slidingWindowSize(); }
    public int getResolvedHeadDim() { return resolvedHeadDim(); }
    public int getResolvedMaxHeadDim() { return resolvedMaxHeadDim(); }
    public int getResolvedNumKvSharedLayers() { return resolvedNumKvSharedLayers(); }
    public double getRmsNormEps() { return rmsNormEps(); }
    public double getRopeTheta() { return ropeTheta(); }
    public List<String> getArchitectures() { return architectures(); }
    public Optional<Integer> getBosTokenId() { return bosTokenId(); }
    public Optional<Integer> getEosTokenId() { return eosTokenId(); }

    // ─────────────────────────────────────────────────────────────────────────
    // Nested types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * RoPE scaling configuration (used for long-context variants).
     * <p>
     * Types: "linear", "dynamic", "yarn".
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class RopeScaling {
        @JsonProperty("type")
        public String type;
        @JsonProperty("factor")
        public double factor = 1.0;
        @JsonProperty("low_freq_factor")
        public double lowFreqFactor = 1.0;
        @JsonProperty("high_freq_factor")
        public double highFreqFactor = 4.0;
        @JsonProperty("original_max_position_embeddings")
        public int originalMaxPositionEmbeddings = 8192;
    }

    @Override
    public String toString() {
        return "ModelConfig{type=" + modelType
                + ", arch=" + primaryArchitecture()
                + ", layers=" + numHiddenLayers
                + ", hidden=" + hiddenSize
                + ", heads=" + numAttentionHeads + "/" + resolvedNumKvHeads()
                + ", vocab=" + vocabSize
                + '}';
    }

    /**
     * Check if model has sliding window attention.
     */
    public boolean hasSlidingWindow() {
        return !Boolean.FALSE.equals(useSlidingWindow)
                && slidingWindow != null
                && slidingWindow > 0;
    }

    /**
     * Get sliding window size.
     */
    public int slidingWindowSize() {
        return slidingWindow != null ? slidingWindow : Integer.MAX_VALUE;
    }

    /**
     * Check if model is MoE (Mixture of Experts).
     */
    public boolean isMoe() {
        return numLocalExperts != null && numLocalExperts > 1;
    }

    /**
     * Get number of local experts (for MoE models).
     */
    public int numLocalExperts() {
        return numLocalExperts != null ? numLocalExperts : 0;
    }

    public int getNumLocalExperts() {
        return numLocalExperts();
    }

    /**
     * Get number of experts per token (for MoE models).
     */
    public int numExpertsPerTok() {
        return numExpertsPerTok != null ? numExpertsPerTok : 0;
    }

    public int getNumExpertsPerTok() {
        return numExpertsPerTok();
    }

    public boolean enableMoeBlock() {
        return Boolean.TRUE.equals(enableMoeBlock);
    }

    public int moeIntermediateSize() {
        return moeIntermediateSize != null ? moeIntermediateSize : 0;
    }

    public boolean usesDoubleWideMlp() {
        return Boolean.TRUE.equals(useDoubleWideMlp);
    }

    public boolean isGemma4PackedMoe() {
        String type = modelType == null ? "" : modelType.toLowerCase(Locale.ROOT);
        String arch = primaryArchitecture() == null ? "" : primaryArchitecture().toLowerCase(Locale.ROOT);
        boolean gemma4 = type.startsWith("gemma4") || arch.contains("gemma4");
        return gemma4
                && (enableMoeBlock()
                        || moeIntermediateSize() > 0
                        || (numLocalExperts() > 1 && numExpertsPerTok() > 0));
    }

    public boolean requiresGemma4PackedMoeRuntime() {
        return isGemma4PackedMoe();
    }

    /**
     * Check if layer is MoE layer (for interleaved architectures).
     */
    public boolean isMoeLayer(int layerIdx) {
        if (!isMoe())
            return false;
        int step = decoderSparseStep != null ? decoderSparseStep : 1;
        return (layerIdx + 1) % step == 0;
    }

    /**
     * Get number of key-value heads (GQA).
     */
    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }
}
