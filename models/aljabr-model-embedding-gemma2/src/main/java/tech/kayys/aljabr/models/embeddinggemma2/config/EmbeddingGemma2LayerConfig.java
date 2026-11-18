package tech.kayys.aljabr.models.embeddinggemma2.config;

/**
 * Effective attention geometry and positional-encoding settings for one text
 * transformer layer.
 *
 * <p>Projection widths are derived from the effective head counts and head
 * dimension. They are deliberately not inferred from the model hidden size:
 * EmbeddingGemma 2 has per-layer attention overrides.</p>
 */
public record EmbeddingGemma2LayerConfig(
        int layerIndex,
        AttentionType attentionType,
        int numAttentionHeads,
        int numKeyValueHeads,
        int headDim,
        double ropeTheta,
        String ropeType,
        Double partialRotaryFactor,
        Integer slidingWindow) {

    public enum AttentionType {
        FULL,
        SLIDING
    }

    public EmbeddingGemma2LayerConfig {
        if (layerIndex < 0) {
            throw new IllegalArgumentException("layerIndex must not be negative");
        }
        if (attentionType == null) {
            throw new IllegalArgumentException("attentionType must not be null");
        }
        if (numAttentionHeads <= 0 || numKeyValueHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("attention head counts and headDim must be positive");
        }
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                    "numAttentionHeads must be divisible by numKeyValueHeads");
        }
        if (!Double.isFinite(ropeTheta) || ropeTheta <= 0.0d) {
            throw new IllegalArgumentException("ropeTheta must be finite and positive");
        }
        if (ropeType == null || ropeType.isBlank()) {
            throw new IllegalArgumentException("ropeType must not be blank");
        }
        if (partialRotaryFactor != null
                && (!Double.isFinite(partialRotaryFactor)
                || partialRotaryFactor <= 0.0d || partialRotaryFactor > 1.0d)) {
            throw new IllegalArgumentException("partialRotaryFactor must be in (0, 1]");
        }
        if (attentionType == AttentionType.SLIDING
                && (slidingWindow == null || slidingWindow <= 0)) {
            throw new IllegalArgumentException("sliding attention requires a positive slidingWindow");
        }
        if (attentionType == AttentionType.FULL && slidingWindow != null) {
            throw new IllegalArgumentException("full attention must not have a slidingWindow");
        }
    }

    public int queryProjectionSize() {
        return Math.multiplyExact(numAttentionHeads, headDim);
    }

    public int keyValueProjectionSize() {
        return Math.multiplyExact(numKeyValueHeads, headDim);
    }

    public int queriesPerKeyValueHead() {
        return numAttentionHeads / numKeyValueHeads;
    }
}
