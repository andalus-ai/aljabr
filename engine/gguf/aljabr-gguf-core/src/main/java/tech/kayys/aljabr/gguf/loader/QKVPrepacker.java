package tech.kayys.aljabr.gguf.loader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Simple QKV prepacker: converts a weight matrix [hidden, 3*hidden]
 * into packed layout [3][numHeads][headDim][hidden] for efficient per-head dot products.
 * <p>
 * This is a prototype helper to be used by the native runner to prepack model weights
 * after GGUF / SafeTensor loading. It operates on raw MemorySegment handles (FFM).
 * @author bhangun
 */
public final class QKVPrepacker {

    private QKVPrepacker() {}

    public static MemorySegment prepack(
            MemorySegment wqkv, // Fused [Q, K, V] row-major
            int hidden,
            int numHeads,
            int numHeadsKv,
            int headDim,
            Arena arena
    ) {
        int qLen = numHeads * headDim;
        int kLen = numHeadsKv * headDim;
        
        long packedElements = (long) (numHeads + 2 * numHeadsKv) * headDim * hidden;
        long bytes = packedElements * Float.BYTES;
        MemorySegment packed = arena.allocate(bytes, 64);

        for (int h = 0; h < numHeads; h++) {
            for (int d = 0; d < headDim; d++) {
                int row = h * headDim + d;
                for (int c = 0; c < hidden; c++) {
                    float val = wqkv.get(ValueLayout.JAVA_FLOAT, ((long) row * hidden + c) * Float.BYTES);
                    long dstQ = (((long) 0 * numHeads + h) * headDim + d) * hidden + c;
                    packed.set(ValueLayout.JAVA_FLOAT, dstQ * Float.BYTES, val);
                }
            }
        }

        for (int h = 0; h < numHeadsKv; h++) {
            for (int d = 0; d < headDim; d++) {
                int rowK = (numHeads * headDim) + (h * headDim + d);
                int rowV = ((numHeads + numHeadsKv) * headDim) + (h * headDim + d);
                for (int c = 0; c < hidden; c++) {
                    float valK = wqkv.get(ValueLayout.JAVA_FLOAT, ((long) rowK * hidden + c) * Float.BYTES);
                    float valV = wqkv.get(ValueLayout.JAVA_FLOAT, ((long) rowV * hidden + c) * Float.BYTES);
                    
                    long dstK = (((long) numHeads + h) * headDim + d) * hidden + c;
                    long dstV = (((long) numHeads + numHeadsKv + h) * headDim + d) * hidden + c;
                    
                    packed.set(ValueLayout.JAVA_FLOAT, dstK * Float.BYTES, valK);
                    packed.set(ValueLayout.JAVA_FLOAT, dstV * Float.BYTES, valV);
                }
            }
        }

        return packed;
    }
}
