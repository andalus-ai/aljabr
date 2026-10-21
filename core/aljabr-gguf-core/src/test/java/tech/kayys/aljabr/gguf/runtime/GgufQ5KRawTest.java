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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ5KRawTest {
    @Test
    void rawQ5KMatVecReusesVectorGroupSumsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "288");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlockWithMin(segment.asSlice(0, 176), (byte) 2, (byte) 1, (byte) 0, (byte) 0x11);
            writeQ5KBlockWithMin(segment.asSlice(176, 176), (byte) 3, (byte) 1, (byte) 0, (byte) 0x11);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5.raw.mins", new long[]{256, 2}, 13, 0, 2L * 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ5KRowWalkersHandleUnrolledRowsAndBlockTail() {
        assertRawQ5KRows(false);
        assertRawQ5KRows(true);
    }

    private static void assertRawQ5KRows(boolean hasMins) {
        int rows = 5;
        int blocks = 5;
        int columns = blocks * QK_K;
        long rowBytes = blocks * (long) Q5_K_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    MemorySegment blockSegment =
                            segment.asSlice(rowOffset + block * (long) Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES);
                    if (hasMins) {
                        writeQ5KMinLaneOrderBlock(blockSegment);
                    } else {
                        writeQ5KNoMinLaneOrderBlock(blockSegment);
                    }
                }
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 43 - 21) * 0.01953125f;
            }
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            if (hasMins) {
                float[] groupSums = GgufSum.q4KVectorGroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ5RawDot.dotRowQ5KWithGroupSums(
                            segment, row * rowBytes, columns, vector, groupSums);
                }
                GgufKRawRows.fillMatVecRowsQ5K(segment, columns, rowBytes, vector, groupSums, actual, 0, rows);
            } else {
                for (int row = 0; row < rows; row++) {
                    expected[row] = GgufQ5RawDot.dotRowQ5KNoMins(segment, row * rowBytes, columns, vector, 0);
                }
                GgufKRawRows.fillMatVecRowsQ5KNoMins(segment, columns, rowBytes, vector, actual, 0, rows);
            }

            assertArrayEquals(expected, actual, 1.0e-3f);
        }
    }
}
