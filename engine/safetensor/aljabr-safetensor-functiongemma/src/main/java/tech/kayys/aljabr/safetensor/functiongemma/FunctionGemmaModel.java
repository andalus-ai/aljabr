package tech.kayys.aljabr.safetensor.functiongemma;

import java.util.Objects;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaTokenizer;
import tech.kayys.aljabr.models.gemma3.Gemma3TextBackbone;
import tech.kayys.aljabr.spi.model.ModelConfig;

/**
 * Executable FunctionGemma checkpoint backed by Aljabr's shared Gemma 3
 * text decoder reference implementation.
 *
 * <p>Weights are detached from the SafeTensors mapping during load, so this
 * model does not retain a file mapping and requires no close operation.
 * Runtime/tool dispatch is intentionally outside this model adapter.</p>
 */
public final class FunctionGemmaModel {
    private final ModelConfig config;
    private final FunctionGemmaTokenizer tokenizer;
    private final Gemma3TextBackbone backbone;

    FunctionGemmaModel(
            ModelConfig config,
            FunctionGemmaTokenizer tokenizer,
            Gemma3TextBackbone backbone) {
        this.config = Objects.requireNonNull(config, "config");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.backbone = Objects.requireNonNull(backbone, "backbone");
    }

    /** Returns logits for each token in one unpadded input sequence. */
    public float[][] forward(int[] inputIds) {
        return backbone.forward(inputIds);
    }

    /**
     * Accepts tokenizer-native token IDs and rejects values that cannot be
     * represented by the reference backbone's integer vocabulary indices.
     */
    public float[][] forward(long[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds");
        int[] converted = new int[inputIds.length];
        for (int index = 0; index < inputIds.length; index++) {
            converted[index] = Math.toIntExact(inputIds[index]);
        }
        return backbone.forward(converted);
    }

    public ModelConfig config() {
        return config;
    }

    public FunctionGemmaTokenizer tokenizer() {
        return tokenizer;
    }

    public int vocabularySize() {
        return backbone.vocabularySize();
    }

    public int hiddenSize() {
        return backbone.hiddenSize();
    }

    public int layerCount() {
        return backbone.layerCount();
    }
}
