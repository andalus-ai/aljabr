package tech.kayys.alkhawarizm.gguf.javabackend;

import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;
import tech.kayys.alkhawarizm.gguf.loader.GGUFTensorInfo;
import tech.kayys.alkhawarizm.gguf.loader.inference.KVCache;
import tech.kayys.alkhawarizm.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.alkhawarizm.gguf.runtime.GgufTensorOps;

import java.util.Optional;

/**
 * One decoder forward pass: {@code token, position -> logits}, given a
 * running {@link KVCache}.
 *
 * <p>Supports standard LLaMA/Mistral/Qwen models as well as Gemma family models
 * (Gemma 2, Gemma 3, Gemma 4) with embedding scaling, per-head Q/K/V RMSNorm,
 * post-attention and post-FFN RMSNorm, GeGLU activations, and final logit
 * softcapping.</p>
 */
final class GgufDecodeStep {
    private final GGUFModel model;
    private final GgufModelConfig cfg;

    private final GGUFTensorInfo tokenEmbed;
    private final float[] outputNorm;
    private final GGUFTensorInfo outputWeight;

    private final float[][] attnNorm;
    private final float[][] attnQNorm;
    private final float[][] attnKNorm;
    private final float[][] postAttnNorm;
    private final GGUFTensorInfo[] wQ, wK, wV, wO;

    private final float[][] ffnNorm;
    private final float[][] postFfnNorm;
    private final GGUFTensorInfo[] ffnGate, ffnUp, ffnDown;

    // Scratch buffers reused across calls to avoid per-token allocation
    private final float[] x, xb, q, k, v, attnFull, attnOut, gate, up, hb, scores;

    GgufDecodeStep(GGUFModel model, GgufModelConfig cfg) {
        this.model = model;
        this.cfg = cfg;

        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int nKVH = cfg.nKVHeads();
        int hDim = cfg.headDim();
        int layers = cfg.nLayers();

        this.tokenEmbed = requireTensor("token_embd.weight");
        this.outputNorm = dequantizeVector(requireTensor("output_norm.weight"), dim);
        this.outputWeight = findTensor("output.weight").orElse(tokenEmbed); // tied embeddings

        this.attnNorm = new float[layers][];
        this.attnQNorm = new float[layers][];
        this.attnKNorm = new float[layers][];
        this.postAttnNorm = new float[layers][];
        this.wQ = new GGUFTensorInfo[layers];
        this.wK = new GGUFTensorInfo[layers];
        this.wV = new GGUFTensorInfo[layers];
        this.wO = new GGUFTensorInfo[layers];

        this.ffnNorm = new float[layers][];
        this.postFfnNorm = new float[layers][];
        this.ffnGate = new GGUFTensorInfo[layers];
        this.ffnUp = new GGUFTensorInfo[layers];
        this.ffnDown = new GGUFTensorInfo[layers];

        for (int i = 0; i < layers; i++) {
            String p = "blk." + i + ".";
            attnNorm[i] = dequantizeVector(requireTensor(p + "attn_norm.weight"), dim);
            attnQNorm[i] = findTensor(p + "attn_q_norm.weight").map(t -> dequantizeVector(t, hDim)).orElse(null);
            attnKNorm[i] = findTensor(p + "attn_k_norm.weight").map(t -> dequantizeVector(t, hDim)).orElse(null);
            postAttnNorm[i] = findTensor(p + "post_attention_norm.weight").map(t -> dequantizeVector(t, dim)).orElse(null);

            wQ[i] = requireTensor(p + "attn_q.weight");
            wK[i] = requireTensor(p + "attn_k.weight");
            wV[i] = findTensor(p + "attn_v.weight").orElse(wK[i]);
            wO[i] = requireTensor(p + "attn_output.weight");

            ffnNorm[i] = dequantizeVector(requireTensor(p + "ffn_norm.weight"), dim);
            postFfnNorm[i] = findTensor(p + "post_ffw_norm.weight").map(t -> dequantizeVector(t, dim)).orElse(null);
            ffnGate[i] = findTensor(p + "ffn_gate.weight").orElse(null); // absent on some archs (plain MLP, no gate)
            ffnUp[i] = requireTensor(p + "ffn_up.weight");
            ffnDown[i] = requireTensor(p + "ffn_down.weight");
        }

        this.x = new float[dim];
        this.xb = new float[dim];
        this.q = new float[nH * hDim];
        this.k = new float[nKVH * hDim];
        this.v = new float[nKVH * hDim];
        this.attnFull = new float[nH * hDim];
        this.attnOut = new float[hDim];
        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];
        this.hb = new float[ffnDim];
        this.scores = new float[cfg.contextLength()];
    }

    /** One token's forward pass with optional logit computation. */
    float[] forward(int tokenId, int pos, KVCache cache, boolean computeLogits) {
        int dim = cfg.embeddingDim();
        int nH = cfg.nHeads();
        int nKVH = cfg.nKVHeads();
        int hDim = cfg.headDim();
        int ffnDim = cfg.ffnDim();
        float eps = cfg.rmsNormEps();

        GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);

        // Gemma scales token embeddings by sqrt(dim)
        if (cfg.isGemma4() || cfg.architecture().startsWith("gemma")) {
            GGUFVectorOps.scale(x, (float) Math.sqrt(dim), dim);
        }

        for (int layer = 0; layer < cfg.nLayers(); layer++) {
            GGUFVectorOps.rmsNorm(x, attnNorm[layer], xb, dim, eps);

            dispatchMatVec(wQ[layer], xb, q);
            dispatchMatVec(wK[layer], xb, k);
            dispatchMatVec(wV[layer], xb, v);

            // Gemma Q and K head norms before RoPE
            if (attnQNorm[layer] != null) {
                int qNormLen = attnQNorm[layer].length;
                for (int h = 0; h < nH; h++) {
                    GGUFVectorOps.rmsNorm(q, h * hDim, attnQNorm[layer], q, h * hDim, qNormLen, eps);
                }
            }

            if (attnKNorm[layer] != null) {
                int kNormLen = attnKNorm[layer].length;
                for (int kvH = 0; kvH < nKVH; kvH++) {
                    GGUFVectorOps.rmsNorm(k, kvH * hDim, attnKNorm[layer], k, kvH * hDim, kNormLen, eps);
                }
            }

            // Gemma 4 normalizes V head vectors with unit RMSNorm before cache store
            if (cfg.isGemma4()) {
                for (int kvH = 0; kvH < nKVH; kvH++) {
                    GGUFVectorOps.rmsNormUnit(v, kvH * hDim, v, kvH * hDim, hDim, eps);
                }
            }

            GGUFVectorOps.rope(q, nH, hDim, pos, cfg.ropeFreqBase());
            GGUFVectorOps.rope(k, nKVH, hDim, pos, cfg.ropeFreqBase());

            cache.store(layer, k, v);

            // Gemma 4 uses attention scale 1.0 (since Q and K are normalized)
            float scale = cfg.isGemma4() ? 1.0f : (float) (1.0 / Math.sqrt(hDim));
            int seqLen = pos + 1;

            GGUFVectorOps.zero(attnFull, nH * hDim);

            for (int h = 0; h < nH; h++) {
                int kvH = h / (nH / nKVH);
                int qOff = h * hDim;

                float[] kCache = cache.kLayer(layer);
                for (int t = 0; t < seqLen; t++) {
                    int kBase = t * cache.headStride() + kvH * hDim;
                    scores[t] = GGUFVectorOps.dot(q, qOff, kCache, kBase, hDim) * scale;
                }

                GGUFVectorOps.softmax(scores, seqLen);

                GGUFVectorOps.zero(attnOut, hDim);
                float[] vCache = cache.vLayer(layer);
                for (int t = 0; t < seqLen; t++) {
                    int vBase = t * cache.headStride() + kvH * hDim;
                    GGUFVectorOps.scaledAdd(attnOut, attnOut, scores[t], vCache, vBase, hDim);
                }

                System.arraycopy(attnOut, 0, attnFull, h * hDim, hDim);
            }

            dispatchMatVec(wO[layer], attnFull, xb);

            // Gemma post-attention norm
            if (postAttnNorm[layer] != null) {
                GGUFVectorOps.rmsNorm(xb, postAttnNorm[layer], xb, dim, eps);
            }
            GGUFVectorOps.addInPlace(x, xb, dim);

            GGUFVectorOps.rmsNorm(x, ffnNorm[layer], xb, dim, eps);

            if (ffnGate[layer] != null) {
                dispatchMatVec(ffnGate[layer], xb, gate);
                dispatchMatVec(ffnUp[layer], xb, up);
                if (cfg.isGemma4() || cfg.architecture().startsWith("gemma")) {
                    GGUFVectorOps.geGLU(gate, up, hb, ffnDim);
                } else {
                    GGUFVectorOps.swiGLU(gate, up, hb, ffnDim);
                }
            } else {
                dispatchMatVec(ffnUp[layer], xb, up);
                GGUFVectorOps.siluInPlace(up, ffnDim);
                System.arraycopy(up, 0, hb, 0, ffnDim);
            }

            dispatchMatVec(ffnDown[layer], hb, xb);

            // Gemma post-ffw norm
            if (postFfnNorm[layer] != null) {
                GGUFVectorOps.rmsNorm(xb, postFfnNorm[layer], xb, dim, eps);
            }
            GGUFVectorOps.addInPlace(x, xb, dim);
        }

        cache.advance();

        if (!computeLogits) {
            return null;
        }

        GGUFVectorOps.rmsNorm(x, outputNorm, xb, dim, eps);

        float[] logits = new float[cfg.vocabSize()];
        dispatchMatVec(outputWeight, xb, logits);

        float cap = cfg.finalLogitSoftcap();
        if (cap > 0.0f) {
            float invCap = 1.0f / cap;
            for (int i = 0; i < logits.length; i++) {
                logits[i] = cap * (float) Math.tanh(logits[i] * invCap);
            }
        }

        return logits;
    }

    /** Overload for backward compatibility computing logits by default. */
    float[] forward(int tokenId, int pos, KVCache cache) {
        return forward(tokenId, pos, cache, true);
    }

    private void dispatchMatVec(GGUFTensorInfo tensor, float[] input, float[] output) {
        if (!GgufMetalBridge.matVec(model, tensor, input, output)) {
            GgufTensorOps.matVecParallel(model, tensor, input, output);
        }
    }

    private GGUFTensorInfo requireTensor(String name) {
        return findTensor(name)
                .orElseThrow(() -> new IllegalStateException(
                        "GGUF model is missing required tensor '" + name + "' for architecture '"
                                + cfg.architecture() + "'"));
    }

    private Optional<GGUFTensorInfo> findTensor(String name) {
        try {
            return Optional.ofNullable(GgufTensorOps.findTensor(model, name));
        } catch (IllegalArgumentException e) {
            return Optional.empty();
        }
    }

    private float[] dequantizeVector(GGUFTensorInfo tensor, int fallbackDim) {
        int dim = (tensor.shape() != null && tensor.shape().length > 0 && tensor.shape()[0] > 0)
                ? (int) tensor.shape()[0]
                : fallbackDim;
        float[] out = new float[dim];
        GgufTensorOps.dequantizeRow(model, tensor, 0L, out);
        return out;
    }
}
