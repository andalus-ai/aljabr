package tech.kayys.aljabr.plugin.runner.gguf;

import tech.kayys.aljabr.gguf.loader.GGUFLoader;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorLoadResult;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Bridges GGUF tensors into the AccelTensor naming/layout expected by the Java direct engine.
 *
 * <p>For Gemma 4-style models, a converted GGUF may omit the per-layer input
 * pathway weights. When the original local safetensor checkpoint is available,
 * we supplement those tensors directly from that checkpoint instead of waiting
 * for a fully Gemma-4-complete converter.
 * @author bhangun
 */
public final class GgufWeightAdapter {

    private GgufWeightAdapter() {
    }

    public static Map<String, AccelTensor> adaptWeights(
            GGUFModel model,
            ModelConfig config,
            ModelArchitecture arch,
            Path safetensorOriginDir,
            SafetensorFFMLoader safetensorLoader) {

        Map<String, AccelTensor> weights = new HashMap<>();

        putIfPresent(weights, arch.embedTokensWeight(), GGUFLoader.loadTensorF32Optional(model, "token_embd.weight"));
        putIfPresent(weights, arch.finalNormWeight(), GGUFLoader.loadTensorF32Optional(model, "output_norm.weight"));

        AccelTensor lmHead = GGUFLoader.loadTensorF32Optional(model, "output.weight");
        if (lmHead != null) {
            putIfPresent(weights, arch.lmHeadWeight(), lmHead);
        }

        for (int i = 0; i < config.getNumHiddenLayers(); i++) {
            String prefix = "blk." + i + ".";

            putIfPresent(weights, arch.layerAttentionNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_norm.weight"));
            putIfPresent(weights, arch.layerOutputWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_output.weight"));

            putIfPresent(weights, arch.layerQueryWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_q.weight"));
            putIfPresent(weights, arch.layerKeyWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_k.weight"));
            putIfPresent(weights, arch.layerValueWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_v.weight"));

            putIfPresent(weights, arch.layerQueryNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_q_norm.weight"));
            putIfPresent(weights, arch.layerKeyNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "attn_k_norm.weight"));

            putIfPresent(weights, arch.layerFfnNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_norm.weight"));
            putIfPresent(weights, arch.layerPostAttnNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_norm.weight"));
            putIfPresent(weights, arch.layerPreFfnNormWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_pre_norm.weight"));

            putIfPresent(weights, arch.layerFfnGateWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_gate.weight"));
            putIfPresent(weights, arch.layerFfnUpWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_up.weight"));
            putIfPresent(weights, arch.layerFfnDownWeight(i),
                    GGUFLoader.loadTensorF32Optional(model, prefix + "ffn_down.weight"));
        }

        if (safetensorOriginDir != null && safetensorLoader != null) {
            supplementMissingSafetensorWeights(weights, config, arch, safetensorOriginDir, safetensorLoader);
        }

        return weights;
    }

    private static void supplementMissingSafetensorWeights(
            Map<String, AccelTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            Path safetensorOriginDir,
            SafetensorFFMLoader safetensorLoader) {

        Path modelFile = safetensorOriginDir.resolve("model.safetensors");
        if (!Files.isRegularFile(modelFile)) {
            return;
        }

        try (SafetensorLoadResult result = safetensorLoader.load(modelFile)) {
            maybeLoad(result, weights, arch.embedTokensPerLayerWeight());
            maybeLoad(result, weights, arch.perLayerModelProjectionWeight());
            maybeLoad(result, weights, arch.perLayerProjectionNormWeight());

            for (int i = 0; i < config.getNumHiddenLayers(); i++) {
                maybeLoad(result, weights, arch.layerPerLayerInputGateWeight(i));
                maybeLoad(result, weights, arch.layerPerLayerProjectionWeight(i));
                maybeLoad(result, weights, arch.layerPostPerLayerInputNormWeight(i));
                maybeLoad(result, weights, arch.layerScalarWeight(i));
            }
        }
    }

    private static void maybeLoad(SafetensorLoadResult result, Map<String, AccelTensor> weights, String tensorName) {
        if (tensorName == null || weights.containsKey(tensorName)) {
            return;
        }

        result.findTensor(tensorName).ifPresent(tensor -> weights.put(tensorName, wrapSafetensor(tensor)));
    }

    private static AccelTensor wrapSafetensor(SafetensorTensor tensor) {
        long[] shape = tensor.shape();
        AccelTensor base = AccelTensor.wrapSegment(tensor.segment(), shape);
        return switch (tensor.dtype()) {
            case F32 -> base;
            case F16 -> base.withQuantization(AccelTensor.QuantType.F16, null, null, -1);
            case BF16 -> base.withQuantization(AccelTensor.QuantType.BF16, null, null, -1);
            default -> AccelTensor.fromFloatArray(tensor.toFloatArray(), shape);
        };
    }

    private static void putIfPresent(Map<String, AccelTensor> weights, String key, AccelTensor tensor) {
        if (key != null && tensor != null) {
            weights.put(key, tensor);
        }
    }
}
