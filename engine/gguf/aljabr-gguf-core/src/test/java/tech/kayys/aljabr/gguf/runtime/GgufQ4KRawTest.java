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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ4KRawTest {
    @Test
    void rawQ4KNoMinMatVecPreservesPackedNibbleLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KNoMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.raw.no_min.lanes", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            assertEquals(expectedQ4KLaneOrderDot(vector, false), output[0], 0.0f);
            assertArrayEquals(expectedQ4KLaneOrderRow(false), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ4KMinMatVecPreservesPackedNibbleLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KMinLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.raw.min.lanes", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            assertEquals(expectedQ4KLaneOrderDot(vector, true), output[0], 0.0f);
            assertArrayEquals(expectedQ4KLaneOrderRow(true), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ4KRowWalkersHandleUnrolledRowsAndBlockTail() {
        assertRawQ4KRows(false);
        assertRawQ4KRows(true);
    }

    private static void assertRawQ4KRows(boolean hasMins) {
        int rows = 5;
        int blocks = 5;
        int columns = blocks * QK_K;
        long rowBytes = blocks * (long) Q4_K_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    MemorySegment blockSegment =
                            segment.asSlice(rowOffset + block * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES);
                    if (hasMins) {
                        writeQ4KMinLaneOrderBlock(blockSegment);
                    } else {
                        writeQ4KNoMinLaneOrderBlock(blockSegment);
                    }
                }
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 41 - 20) * 0.017578125f;
            }
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            if (hasMins) {
                float[] groupSums = GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ4RawDot.dotRowQ4KWithGroupSums(
                            segment, row * rowBytes, columns, vector, groupSums);
                }
                GgufKRawRows.fillMatVecRowsQ4K(segment, columns, rowBytes, vector, groupSums, actual, 0, rows);
            } else {
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ4RawDot.dotRowQ4KNoMins(segment, row * rowBytes, columns, vector, 0);
                }
                GgufKRawRows.fillMatVecRowsQ4KNoMins(segment, columns, rowBytes, vector, actual, 0, rows);
            }

            assertArrayEquals(expected, actual, 1.0e-3f);
        }
    }
}
