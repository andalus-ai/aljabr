package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.expectedQ4KLaneOrderDot;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KNoMinLaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_K_BLOCK_BYTES;

class GgufRawHintTest {
    @Test
    void rawSingleRowRemembersKNoMinHintForOneRowMatrix() {
        String previousMinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "999");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q4_K_BLOCK_BYTES);
            writeQ4KNoMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.raw.single.no_mins",
                    new long[]{256, 1},
                    GgmlType.Q4_K.id,
                    0,
                    Q4_K_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, false);

            assertEquals(expectedQ4KLaneOrderDot(vector, false), output[0], 0.0f);
            assertEquals(Boolean.FALSE, GgufPreparedMatrixEstimator.cachedKHasMins(
                    model,
                    GgufKey.from(tensor),
                    1,
                    8));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousMinRows);
        }
    }

    @Test
    void rawSingleRowRemembersPositiveQ32BiasHintForLargerMatrix() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "999");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q4_1_BLOCK_BYTES * 2L);
            writeQ4_1Block(segment.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            writeQ4_1Block(
                    segment.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0,
                    (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4_1.raw.single.positive_bias",
                    new long[]{32, 2},
                    GgmlType.Q4_1.id,
                    0,
                    Q4_1_BLOCK_BYTES * 2L);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(32);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, false);

            assertEquals(GgufQ32RawDot.dotRowQ4_1(segment, 0, 32, vector, 0), output[0], 0.0f);
            assertEquals(Boolean.TRUE, GgufPreparedMatrixEstimator.cachedQ32HasBlockBiases(
                    model,
                    GgufKey.from(tensor),
                    2));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
        }
    }

    @Test
    void rawKHintRecentCacheStoresDecodedEstimateDecision() {
        GgufRawPathHints.clearRecentHintCache();
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q4_K_BLOCK_BYTES);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.raw.recent_hint.no_mins",
                    new long[]{256, 1},
                    GgmlType.Q4_K.id,
                    0,
                    Q4_K_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufKey key = GgufKey.from(tensor);

            GgufPreparedMatrixEstimator.rememberKHasMinsHint(model, key, 1, 8, false);
            assertEquals(0, GgufRawPathHints.recentHintCacheSize());
            assertEquals(0, GgufRawPathHints.recentHintFastCacheSize());

            assertEquals(Boolean.FALSE, GgufRawPathHints.q4KHasMins(
                    model, key, 1, 1, segment, Q4_K_BLOCK_BYTES, 256, 1));
            assertEquals(1, GgufRawPathHints.recentHintCacheSize());
            assertEquals(1, GgufRawPathHints.recentHintFastCacheSize());

            assertEquals(Boolean.FALSE, GgufRawPathHints.q4KHasMins(
                    model, key, 1, 1, segment, Q4_K_BLOCK_BYTES, 256, 1));
            assertEquals(1, GgufRawPathHints.recentHintCacheSize());
            assertEquals(1, GgufRawPathHints.recentHintFastCacheSize());
        } finally {
            GgufRawPathHints.clearRecentHintCache();
        }
    }

    @Test
    void rawFullRowsRememberKNoMinHintWhenPreparedCacheSkipped() {
        String previousMinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "999");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q4_K_BLOCK_BYTES * 2L);
            writeQ4KNoMinLaneOrderBlock(segment.asSlice(0, Q4_K_BLOCK_BYTES));
            writeQ4KNoMinLaneOrderBlock(segment.asSlice(Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.raw.full.no_mins",
                    new long[]{256, 2},
                    GgmlType.Q4_K.id,
                    0,
                    Q4_K_BLOCK_BYTES * 2L);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[2];
            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, false);

            float expected = expectedQ4KLaneOrderDot(vector, false);
            assertEquals(expected, output[0], 0.0f);
            assertEquals(expected, output[1], 0.0f);
            assertEquals(Boolean.FALSE, GgufPreparedMatrixEstimator.cachedKHasMins(
                    model,
                    GgufKey.from(tensor),
                    2,
                    8));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousMinRows);
        }
    }

    @Test
    void rawFullRowsRememberQ32NoBiasHintWhenPreparedCacheSkipped() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "999");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(Q4_1_BLOCK_BYTES * 2L);
            writeQ4_1Block(segment.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ4_1Block(
                    segment.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0,
                    (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4_1.raw.full.no_bias",
                    new long[]{32, 2},
                    GgmlType.Q4_1.id,
                    0,
                    Q4_1_BLOCK_BYTES * 2L);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(32);
            float[] output = new float[2];
            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, false);

            assertEquals(GgufQ32RawDot.dotRowQ4_1NoBias(segment, 0, 32, vector, 0), output[0], 0.0f);
            assertEquals(
                    GgufQ32RawDot.dotRowQ4_1NoBias(segment, Q4_1_BLOCK_BYTES, 32, vector, 0),
                    output[1],
                    0.0f);
            assertEquals(Boolean.FALSE, GgufPreparedMatrixEstimator.cachedQ32HasBlockBiases(
                    model,
                    GgufKey.from(tensor),
                    2));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
        }
    }
}
