package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Immutable logical geometry and location for one KV-cache compression block.
 *
 * <p>The values describe a contiguous, position-major block for one layer and
 * one tensor kind. The codec may use the location for deterministic transforms
 * or diagnostics, but decoding must not depend on mutable session state.</p>
 *
 * @param layer transformer layer index
 * @param firstPosition first sequence position represented by the block
 * @param tokenCount number of contiguous positions represented by the block
 * @param kvHeads number of KV heads in this layer
 * @param headDimension dimension of each KV head
 * @param tensorKind whether the block contains keys or values
 */
public record KVCacheCompressionContext(
        int layer,
        int firstPosition,
        int tokenCount,
        int kvHeads,
        int headDimension,
        KVCacheTensorKind tensorKind) {

    public KVCacheCompressionContext {
        if (layer < 0) {
            throw new IllegalArgumentException("layer must not be negative");
        }
        if (firstPosition < 0) {
            throw new IllegalArgumentException("firstPosition must not be negative");
        }
        if (tokenCount <= 0) {
            throw new IllegalArgumentException("tokenCount must be positive");
        }
        if (kvHeads <= 0) {
            throw new IllegalArgumentException("kvHeads must be positive");
        }
        if (headDimension <= 0) {
            throw new IllegalArgumentException("headDimension must be positive");
        }
        Objects.requireNonNull(tensorKind, "tensorKind");
        try {
            Math.multiplyExact(tokenCount, Math.multiplyExact(kvHeads, headDimension));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("KV compression block geometry exceeds supported indexing", overflow);
        }
    }

    /** Number of float elements represented by this block. */
    public int elementCount() {
        return Math.multiplyExact(tokenCount, Math.multiplyExact(kvHeads, headDimension));
    }
}
