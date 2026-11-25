package tech.kayys.aljabr.models.embeddinggemma2.output;

import tech.kayys.aljabr.models.embedding.common.MaskedMeanPooler;

/**
 * EmbeddingGemma 2 compatibility facade for the shared mask-aware mean pooler.
 *
 * @deprecated Use {@link MaskedMeanPooler} from the embedding-common module.
 */
@Deprecated(forRemoval = false)
public final class EmbeddingGemma2MeanPooler {
    private EmbeddingGemma2MeanPooler() {
    }

    public static float[] meanPool(float[][] tokenStates, boolean[] attentionMask) {
        return MaskedMeanPooler.meanPool(tokenStates, attentionMask);
    }
}
