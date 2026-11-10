package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ2KRawTest {
    @Test
    void rawQ2KMatVecReusesVectorGroupSumsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlockWithMin(segment.asSlice(0, 84), (byte) 0x12, (byte) 0x55);
            writeQ2KBlockWithMin(segment.asSlice(84, 84), (byte) 0x13, (byte) 0x55);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.raw.mins", new long[]{256, 2}, 10, 0, 2L * 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(768L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ2KMatVecRemembersHintWhenEstimateHintIsMissing() {
        String previous = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "32");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84);
            writeQ2KBlockWithMin(segment, (byte) 0x12, (byte) 0x55);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.raw.unknown_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 1, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previous);
        }
    }

    @Test
    void rawQ2KRowWalkersHandleUnrolledRowsAndBlockTail() {
        assertRawQ2KRows(false);
        assertRawQ2KRows(true);
    }

    private static void assertRawQ2KRows(boolean hasMins) {
        int rows = 5;
        int blocks = 5;
        int columns = blocks * QK_K;
        long rowBytes = blocks * (long) Q2_K_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    MemorySegment blockSegment =
                            segment.asSlice(rowOffset + block * (long) Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES);
                    if (hasMins) {
                        writeQ2KMinLaneOrderBlock(blockSegment);
                    } else {
                        writeQ2KNoMinLaneOrderBlock(blockSegment);
                    }
                }
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 37 - 18) * 0.021484375f;
            }
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            if (hasMins) {
                float[] groupSums = GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ2RawDot.dotRowQ2KWithGroupSums(
                            segment, row * rowBytes, columns, vector, groupSums);
                }
                GgufKRawRows.fillMatVecRowsQ2K(segment, columns, rowBytes, vector, groupSums, actual, 0, rows);
            } else {
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ2RawDot.dotRowQ2KNoMins(segment, row * rowBytes, columns, vector, 0);
                }
                GgufKRawRows.fillMatVecRowsQ2KNoMins(segment, columns, rowBytes, vector, actual, 0, rows);
            }

            assertArrayEquals(expected, actual, 1.0e-3f);
        }
    }
}
