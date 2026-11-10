package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Resolves the per-layer tensors and vector parameters used by the GGUF Java
 * transformer execution path.
 *
 * <p>This component owns GGUF tensor-name mapping and optional-weight
 * interpretation. It deliberately does not execute operations, choose a
 * compute backend, or register native buffers.</p>
 */
final class GgufLayerWeightResolver {
    private GgufLayerWeightResolver() {
    }

    static GgufLayerWeights resolve(
            GgufTensorResolver tensors,
            GgufModelConfig config,
            int embeddingDimension,
            int embeddingDimensionPerLayer,
            GGUFTensorInfo perLayerTokenEmbedding) {
        int layers = config.nLayers();
        GgufLayerWeights weights = new GgufLayerWeights(layers);

        for (int layer = 0; layer < layers; layer++) {
            String prefix = "blk." + layer + ".";
            int headDimension = config.headDimForLayer(layer);

            weights.attnNorm[layer] = tensors.dequantizeVector(
                    tensors.requireTensor(prefix + "attn_norm.weight"), embeddingDimension);
            weights.attnQNorm[layer] = tensors.findTensor(prefix + "attn_q_norm.weight")
                    .map(tensor -> tensors.dequantizeVector(tensor, headDimension))
                    .orElse(null);
            weights.attnKNorm[layer] = tensors.findTensor(prefix + "attn_k_norm.weight")
                    .map(tensor -> tensors.dequantizeVector(tensor, headDimension))
                    .orElse(null);
            weights.attnVNorm[layer] = tensors.findTensor(prefix + "attn_v_norm.weight")
                    .map(tensor -> tensors.dequantizeVector(tensor, headDimension))
                    .orElse(null);
            weights.postAttnNorm[layer] = tensors.findTensor(prefix + "post_attention_norm.weight")
                    .map(tensor -> tensors.dequantizeVector(tensor, embeddingDimension))
                    .orElse(null);

            // Prefer a layer-local proportional RoPE tensor; otherwise use a
            // global tensor only when its width matches this layer's rotary half.
            GGUFTensorInfo layerRope = tensors.findTensor(prefix + "rope_freqs.weight").orElse(null);
            if (layerRope != null) {
                weights.ropeFreqs[layer] = tensors.dequantizeVector(layerRope, headDimension / 2);
            } else {
                weights.ropeFreqs[layer] = tensors.findTensor("rope_freqs.weight")
                        .filter(tensor -> tensor.shape().length > 0
                                && tensor.shape()[0] == headDimension / 2)
                        .map(tensor -> tensors.dequantizeVector(tensor, headDimension / 2))
                        .orElse(null);
            }

            weights.outScale[layer] = tensors.findTensor(prefix + "layer_output_scale.weight")
                    .map(tensor -> {
                        float[] scale = tensors.dequantizeVector(tensor, 1);
                        return scale.length > 0 ? scale[0] : 1.0f;
                    })
                    .orElse(1.0f);

            weights.wQ[layer] = tensors.requireTensor(prefix + "attn_q.weight");
            weights.wK[layer] = tensors.findTensor(prefix + "attn_k.weight").orElse(null);
            weights.wV[layer] = tensors.findTensor(prefix + "attn_v.weight").orElse(null);
            weights.wO[layer] = tensors.requireTensor(prefix + "attn_output.weight");

            weights.ffnNorm[layer] = tensors.dequantizeVector(
                    tensors.requireTensor(prefix + "ffn_norm.weight"), embeddingDimension);
            weights.postFfnNorm[layer] = tensors.findTensor(prefix + "post_ffw_norm.weight")
                    .map(tensor -> tensors.dequantizeVector(tensor, embeddingDimension))
                    .orElse(null);
            weights.ffnGate[layer] = tensors.findTensor(prefix + "ffn_gate.weight").orElse(null);
            weights.ffnUp[layer] = tensors.requireTensor(prefix + "ffn_up.weight");
            weights.ffnDown[layer] = tensors.requireTensor(prefix + "ffn_down.weight");

            if (perLayerTokenEmbedding != null) {
                weights.perLayerInpGate[layer] = tensors.findTensor(prefix + "inp_gate.weight").orElse(null);
                weights.perLayerProj[layer] = tensors.findTensor(prefix + "proj.weight").orElse(null);
                weights.perLayerPostNorm[layer] = tensors.findTensor(prefix + "post_norm.weight")
                        .map(tensor -> tensors.dequantizeVector(tensor, embeddingDimension))
                        .orElse(null);
            }
        }

        return weights;
    }
}
