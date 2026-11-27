package tech.kayys.aljabr.gguf.javabackend;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Owns the off-heap normalization weights shared by the GGUF Metal decode path.
 *
 * <p>The native Metal runtime retains registered buffer addresses, so replaced
 * generations are deliberately not closed here. A published generation keeps
 * its shared arena reachable for the lifetime of the store.</p>
 */
final class GgufMetalNormBufferStore {
    private static final int SLOT_COUNT = 6;
    private static final int MODEL_DIMENSION_SLOTS = 4;

    private volatile Generation current;

    synchronized MemorySegment[] register(
            int layers, int dimension, int headDimension,
            float[][] postAttention, float[][] ffn, float[][] postFfn, float[][] attention,
            float[][] attentionQuery, float[][] attentionKey) {
        int headDim = headDimension > 0 ? headDimension : 256;
        Arena arena = Arena.ofShared();
        MemorySegment[] segments = new MemorySegment[SLOT_COUNT];
        int[] widths = {dimension, dimension, dimension, dimension, headDim, headDim};
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            segments[slot] = arena.allocate((long) layers * widths[slot] * Float.BYTES);
        }

        float[][][] sources = {
                postAttention, ffn, postFfn, attention, attentionQuery, attentionKey
        };
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            float[][] source = sources[slot];
            int width = widths[slot];
            for (int layer = 0; layer < layers; layer++) {
                float[] values = source != null && layer < source.length ? source[layer] : null;
                if (values != null) {
                    int copyLength = Math.min(values.length, width);
                    MemorySegment.copy(values, 0, segments[slot], ValueLayout.JAVA_FLOAT,
                            (long) layer * width * Float.BYTES, copyLength);
                }
            }
        }

        // Publish the arena, dimensions and segments as one immutable generation.
        current = new Generation(arena, segments, widths, layers);
        return segments.clone();
    }

    long[] slotByteSizes(int layers, int dimension, int headDimension) {
        int headDim = headDimension > 0 ? headDimension : 256;
        long modelBytes = (long) layers * dimension * Float.BYTES;
        long headBytes = (long) layers * headDim * Float.BYTES;
        return new long[] {modelBytes, modelBytes, modelBytes, modelBytes, headBytes, headBytes};
    }

    MemorySegment slice(int slot, int layer) {
        Generation generation = current;
        if (generation == null || slot < 0 || slot >= MODEL_DIMENSION_SLOTS
                || layer < 0 || layer >= generation.layers()) {
            return MemorySegment.NULL;
        }
        long offset = (long) layer * generation.widths()[slot] * Float.BYTES;
        return generation.segments()[slot].asSlice(
                offset, (long) generation.widths()[slot] * Float.BYTES);
    }

    private record Generation(
            Arena arena, MemorySegment[] segments, int[] widths, int layers) {}
}
