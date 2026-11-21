package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/** Executes causal single-token attention over the active KV-cache window. */
final class GgufDecodeAttentionExecution {
    private GgufDecodeAttentionExecution() {
    }

    /**
     * Computes decode attention from a sequence-aware cache session.
     *
     * <p>The session may expose the committed prefix plus the current
     * position, but only after this layer's current K/V write has completed.
     * The active window is copied into compact host buffers before invoking
     * the shared attention kernel; no raw cache backing arrays are exposed.</p>
     */
    static void execute(
            float[] query,
            float[] output,
            float[] scores,
            KVCacheSession session,
            int cacheLayer,
            int kvHeads,
            int queryHeads,
            int headDimension,
            int sequenceLength,
            int startPosition,
            float scale) {
        if (session == null) {
            throw new IllegalArgumentException("KV cache session must not be null");
        }
        if (cacheLayer < 0 || cacheLayer >= session.layout().layerCount()) {
            throw new IndexOutOfBoundsException("KV cache layer is outside the session layout");
        }
        if (kvHeads <= 0 || headDimension <= 0
                || session.layout().kvHeads(cacheLayer) != kvHeads
                || session.layout().headDimension(cacheLayer) != headDimension) {
            throw new IllegalArgumentException("KV cache session geometry does not match decode attention");
        }
        if (sequenceLength != session.committedLength() + 1) {
            throw new IllegalStateException(
                    "Decode sequence length must include exactly the session's current uncommitted position");
        }
        if (startPosition < 0 || startPosition >= sequenceLength) {
            throw new IndexOutOfBoundsException("Attention start position is outside the decode sequence");
        }
        if (queryHeads <= 0 || queryHeads % kvHeads != 0) {
            throw new IllegalArgumentException("Query heads must be a positive multiple of KV heads");
        }
        int activeLength = sequenceLength - startPosition;
        int stride = session.layout().layerStride(cacheLayer);
        float[] activeKeys = new float[Math.multiplyExact(activeLength, stride)];
        float[] activeValues = new float[Math.multiplyExact(activeLength, stride)];
        session.readKeyRangeForAttention(cacheLayer, startPosition, activeLength, activeKeys, 0);
        session.readValueRangeForAttention(cacheLayer, startPosition, activeLength, activeValues, 0);
        execute(query, output, scores, activeKeys, activeValues, stride,
                kvHeads, queryHeads, headDimension, activeLength, 0, scale);
    }

    /**
     * Computes grouped-query attention for one decode position.
     *
     * <p>The caller selects the KV-cache layer and active sequence window. This
     * keeps cache ownership, SWA layer selection, and KV writes in the decode
     * orchestrator while this class owns only score, softmax, and value
     * accumulation.</p>
     */
    static void execute(
            float[] query,
            float[] output,
            float[] scores,
            float[] keyCache,
            float[] valueCache,
            int cacheHeadStride,
            int kvHeads,
            int queryHeads,
            int headDimension,
            int sequenceLength,
            int startPosition,
            float scale) {
        GGUFVectorOps.zero(output, queryHeads * headDimension);

        int activeLength = sequenceLength - startPosition;
        for (int head = 0; head < queryHeads; head++) {
            int kvHead = head / (queryHeads / kvHeads);
            int queryOffset = head * headDimension;

            for (int position = startPosition; position < sequenceLength; position++) {
                int keyOffset = position * cacheHeadStride + kvHead * headDimension;
                scores[position - startPosition] = GGUFVectorOps.dot(
                        query, queryOffset, keyCache, keyOffset, headDimension) * scale;
            }

            GGUFVectorOps.softmax(scores, activeLength);

            int outputOffset = head * headDimension;
            for (int position = startPosition; position < sequenceLength; position++) {
                int valueOffset = position * cacheHeadStride + kvHead * headDimension;
                GGUFVectorOps.scaledAddInPlace(
                        output, outputOffset, scores[position - startPosition],
                        valueCache, valueOffset, headDimension);
            }
        }
    }
}
