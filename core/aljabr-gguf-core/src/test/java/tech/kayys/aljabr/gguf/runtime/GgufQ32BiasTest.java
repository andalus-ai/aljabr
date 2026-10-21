package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;

class GgufQ32BiasTest {
    @Test
    void rawSingleRowDispatchClassifiesQ32BiasStateAtRowOffset() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4 = arena.allocate(2L * Q4_1_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(2L * Q5_1_BLOCK_BYTES);
            writeQ4_1Block(q4.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ4_1Block(
                    q4.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    (byte) 0x21);
            writeQ5_1Block(q5.asSlice(0, Q5_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, -1, (byte) 0x21);
            writeQ5_1Block(
                    q5.asSlice(Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    -1,
                    (byte) 0x21);

            float[] vector = ramp(32);
            float[] vectorBlockSums = GgufSum.vector32GroupSums(vector, 32, new GgufTensorOps.Q4KWorkBuffer());

            assertEquals(GgufQ32RawDot.dotRowQ4_1NoBias(q4, 0, 32, vector, 0),
                    GgufRawDot.q4_1(q4, 0, 32, vector, null), 0.0f);
            assertEquals(GgufQ32RawDot.dotRowQ4_1(q4, Q4_1_BLOCK_BYTES, 32, vector, vectorBlockSums),
                    GgufRawDot.q4_1(q4, Q4_1_BLOCK_BYTES, 32, vector, null), 0.0f);
            assertEquals(GgufQ32RawDot.dotRowQ5_1NoBias(q5, 0, 32, vector, 0),
                    GgufRawDot.q5_1(q5, 0, 32, vector, null), 0.0f);
            assertEquals(GgufQ32RawDot.dotRowQ5_1(q5, Q5_1_BLOCK_BYTES, 32, vector, vectorBlockSums),
                    GgufRawDot.q5_1(q5, Q5_1_BLOCK_BYTES, 32, vector, null), 0.0f);
        }
    }

    @Test
    void directQ32BiasRawDotsMatchPrecomputedVectorSumsAcrossBlocks() {
        int blocks = 5;
        int columns = Q4_0_BLOCK_SIZE * blocks;
        short[] halfScales = {(short) 0x3c00, (short) 0x4000, (short) 0xbc00, (short) 0x3800, (short) 0x4200};
        short[] halfMins = {(short) 0x3800, (short) 0x3400, (short) 0xb800, (short) 0x3c00, (short) 0x3000};
        byte[] packedQuants = {0x21, 0x43, 0x65, (byte) 0x87, (byte) 0xa9};
        int[] highBits = {0x5a5a_a5a5, 0xa5a5_5a5a, 0, -1, 0x3333_cccc};
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4 = arena.allocate(blocks * (long) Q4_1_BLOCK_BYTES);
            MemorySegment q5 = arena.allocate(blocks * (long) Q5_1_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ4_1Block(
                        q4.asSlice(block * (long) Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                        halfScales[block],
                        halfMins[block],
                        packedQuants[block]);
                writeQ5_1Block(
                        q5.asSlice(block * (long) Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                        halfScales[block],
                        halfMins[block],
                        highBits[block],
                        packedQuants[block]);
            }

            float[] vector = fractionalRamp(columns);
            float[] vectorBlockSums = GgufSum.vector32GroupSums(
                    vector,
                    columns,
                    new GgufTensorOps.Q4KWorkBuffer());

            assertEquals(
                    GgufQ32RawDot.dotRowQ4_1(q4, 0, columns, vector, vectorBlockSums),
                    GgufQ32RawDot.dotRowQ4_1(q4, 0, columns, vector, 0),
                    1.0e-4f);
            assertEquals(
                    GgufQ32RawDot.dotRowQ5_1(q5, 0, columns, vector, vectorBlockSums),
                    GgufQ32RawDot.dotRowQ5_1(q5, 0, columns, vector, 0),
                    1.0e-4f);
        }
    }

    @Test
    void rawQ4_1AndQ5_1MatVecRememberBiasHintsWhenEstimateHintIsMissing() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "32");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(20 + 24);
            writeQ4_1Block(segment.asSlice(0, 20), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ5_1Block(segment.asSlice(20, 24), (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4_1.raw.unknown_bias", new long[]{32, 1}, 3, 0, 20);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5_1.raw.unknown_bias", new long[]{32, 1}, 7, 20, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(32);
            float[] q4Output = new float[1];
            float[] q5Output = new float[1];
            GgufTensorOps.matVecRows(model, q4Tensor, vector, q4Output, 1, true);
            GgufTensorOps.matVecRows(model, q5Tensor, vector, q5Output, 1, true);

            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), q4Output[0], 0.0f);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), q5Output[0], 0.0f);
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
        }
    }

    @Test
    void rawKnownBiasQ32MatVecUsesDirectRowsForSingleDecodeRow() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(20 + 24);
            writeQ4_1Block(segment.asSlice(0, 20), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ5_1Block(segment.asSlice(20, 24), (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo q4Tensor = new GGUFTensorInfo("q4_1.raw.single.known_bias", new long[]{32, 1}, 3, 0, 20);
            GGUFTensorInfo q5Tensor = new GGUFTensorInfo("q5_1.raw.single.known_bias", new long[]{32, 1}, 7, 20, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q4Tensor, q5Tensor), 0, segment, null);

            float[] vector = ramp(32);
            float[] q4Output = new float[1];
            float[] q5Output = new float[1];
            GgufTensorOps.matVecRows(model, q4Tensor, vector, q4Output, 1, true);
            GgufTensorOps.matVecRows(model, q5Tensor, vector, q5Output, 1, true);

            assertEquals(GgufTensorOps.dotRow(model, q4Tensor, 0, vector), q4Output[0], 0.0f);
            assertEquals(GgufTensorOps.dotRow(model, q5Tensor, 0, vector), q5Output[0], 0.0f);
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
        }
    }

    private static float[] fractionalRamp(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i + 1) * 0.125f + ((i & 1) == 0 ? 0.5f : -0.25f);
        }
        return values;
    }
}
