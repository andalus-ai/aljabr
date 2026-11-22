package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * Non-trainable sinusoidal relative positions used by Gemma 4 chunked audio attention.
 *
 * <p>The output has shape [contextSize / 2 + 1][hiddenSize], with the
 * concatenated [sin, cos] convention and descending positions used by the
 * upstream relative-shift attention implementation.</p>
 */
public final class EmbeddingGemma2AudioRelativePositionEncoding {
    private final int hiddenSize;
    private final int contextSize;
    private final float[] inverseTimescales;

    public EmbeddingGemma2AudioRelativePositionEncoding(EmbeddingGemma2AudioConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        this.hiddenSize = config.hiddenSize();
        this.contextSize = Math.addExact(
                Math.addExact(config.attentionChunkSize(), config.attentionContextLeft() - 1),
                config.attentionContextRight());
        if ((hiddenSize & 1) != 0) {
            throw new IllegalArgumentException("relative audio position encoding requires an even hidden size");
        }
        int count = hiddenSize / 2;
        this.inverseTimescales = new float[count];
        double denominator = Math.max(count - 1, 1);
        double increment = Math.log(10_000.0) / denominator;
        for (int i = 0; i < count; i++) {
            inverseTimescales[i] = (float) Math.exp(-i * increment);
        }
    }

    /** Returns relative positions shaped [contextSize / 2 + 1][hiddenSize]. */
    public float[][] compute() {
        int positions = contextSize / 2 + 1;
        float[][] result = new float[positions][hiddenSize];
        for (int p = 0; p < positions; p++) {
            int positionId = contextSize / 2 - p;
            for (int i = 0; i < hiddenSize / 2; i++) {
                float angle = positionId * inverseTimescales[i];
                result[p][i] = (float) Math.sin(angle);
                result[p][i + hiddenSize / 2] = (float) Math.cos(angle);
            }
        }
        return result;
    }

    public int contextSize() {
        return contextSize;
    }
}
