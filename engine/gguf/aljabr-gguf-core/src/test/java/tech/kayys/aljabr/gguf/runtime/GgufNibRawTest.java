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
import static tech.kayys.aljabr.gguf.runtime.GgufFx.dequantizedRow;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4NLBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4XSBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeNVFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_XS_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;

class GgufNibRawTest {
    private static final int BLOCKS_WITH_TAIL = 5;
    private static final int VECTOR_OFFSET = 3;

    @Test
    void mxfp4RawDotHandlesUnrolledBlocksAndTail() {
        int columns = BLOCKS_WITH_TAIL * MXFP4_BLOCK_SIZE;
        long rowBytes = (long) BLOCKS_WITH_TAIL * MXFP4_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rowBytes);
            byte[] exponents = {(byte) 128, (byte) 129, (byte) 127, (byte) 128, (byte) 129};
            byte[] quants = {(byte) 0xA5, (byte) 0x3C, (byte) 0x87, (byte) 0x1E, (byte) 0xF0};
            for (int block = 0; block < BLOCKS_WITH_TAIL; block++) {
                long offset = (long) block * MXFP4_BLOCK_BYTES;
                writeMXFP4Block(segment.asSlice(offset, MXFP4_BLOCK_BYTES), exponents[block], quants[block]);
            }
            GGUFTensorInfo tensor = tensor("mxfp4_raw", GgmlType.MXFP4, columns, rowBytes);
            GGUFModel model = model(tensor, segment);
            float[] vector = wave(columns + VECTOR_OFFSET);
            float expected = dotAt(dequantizedRow(model, tensor), vector, VECTOR_OFFSET);

            assertClose(expected, GgufNibRawDot.dotRowMXFP4(segment, 0, columns, vector, VECTOR_OFFSET));
        }
    }

    @Test
    void nvfp4RawDotHandlesUnrolledBlocksAndTail() {
        int columns = BLOCKS_WITH_TAIL * NVFP4_BLOCK_SIZE;
        long rowBytes = (long) BLOCKS_WITH_TAIL * NVFP4_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rowBytes);
            byte[] quants = {(byte) 0xA5, (byte) 0x3C, (byte) 0x87, (byte) 0x1E, (byte) 0xF0};
            for (int block = 0; block < BLOCKS_WITH_TAIL; block++) {
                long offset = (long) block * NVFP4_BLOCK_BYTES;
                writeNVFP4Block(
                        segment.asSlice(offset, NVFP4_BLOCK_BYTES),
                        (byte) 0x40,
                        (byte) 0x48,
                        (byte) 0x38,
                        (byte) 0x30,
                        quants[block]);
            }
            GGUFTensorInfo tensor = tensor("nvfp4_raw", GgmlType.NVFP4, columns, rowBytes);
            GGUFModel model = model(tensor, segment);
            float[] vector = wave(columns + VECTOR_OFFSET);
            float expected = dotAt(dequantizedRow(model, tensor), vector, VECTOR_OFFSET);

            assertClose(expected, GgufNibRawDot.dotRowNVFP4(segment, 0, columns, vector, VECTOR_OFFSET));
        }
    }

    @Test
    void iq4NlRawDotHandlesUnrolledBlocksAndTail() {
        int columns = BLOCKS_WITH_TAIL * IQ4_NL_BLOCK_SIZE;
        long rowBytes = (long) BLOCKS_WITH_TAIL * IQ4_NL_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rowBytes);
            short[] scales = {
                    (short) 0x3c00,
                    (short) 0x4000,
                    (short) 0x3800,
                    (short) 0x3c00,
                    (short) 0x4000};
            byte[] quants = {(byte) 0x88, (byte) 0x99, (byte) 0x7A, (byte) 0x21, (byte) 0xFE};
            for (int block = 0; block < BLOCKS_WITH_TAIL; block++) {
                long offset = (long) block * IQ4_NL_BLOCK_BYTES;
                writeIQ4NLBlock(segment.asSlice(offset, IQ4_NL_BLOCK_BYTES), scales[block], quants[block]);
            }
            GGUFTensorInfo tensor = tensor("iq4_nl_raw", GgmlType.IQ4_NL, columns, rowBytes);
            GGUFModel model = model(tensor, segment);
            float[] vector = wave(columns + VECTOR_OFFSET);
            float expected = dotAt(dequantizedRow(model, tensor), vector, VECTOR_OFFSET);

            assertClose(expected, GgufNibRawDot.dotRowIQ4NL(segment, 0, columns, vector, VECTOR_OFFSET));
        }
    }

    @Test
    void iq4XsRawDotHandlesUnrolledBlocksAndTail() {
        int columns = BLOCKS_WITH_TAIL * QK_K;
        long rowBytes = (long) BLOCKS_WITH_TAIL * IQ4_XS_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rowBytes);
            short[] scales = {
                    (short) 0x3c00,
                    (short) 0x4000,
                    (short) 0x3800,
                    (short) 0x3c00,
                    (short) 0x4000};
            byte[] quants = {(byte) 0x88, (byte) 0x99, (byte) 0x7A, (byte) 0x21, (byte) 0xFE};
            for (int block = 0; block < BLOCKS_WITH_TAIL; block++) {
                long offset = (long) block * IQ4_XS_BLOCK_BYTES;
                writeIQ4XSBlock(segment.asSlice(offset, IQ4_XS_BLOCK_BYTES), scales[block], quants[block]);
            }
            GGUFTensorInfo tensor = tensor("iq4_xs_raw", GgmlType.IQ4_XS, columns, rowBytes);
            GGUFModel model = model(tensor, segment);
            float[] vector = wave(columns + VECTOR_OFFSET);
            float expected = dotAt(dequantizedRow(model, tensor), vector, VECTOR_OFFSET);

            assertClose(expected, GgufNibRawDot.dotRowIQ4XS(segment, 0, columns, vector, VECTOR_OFFSET));
        }
    }

    private static GGUFModel model(GGUFTensorInfo tensor, MemorySegment segment) {
        return new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
    }

    private static GGUFTensorInfo tensor(String name, GgmlType type, int columns, long rowBytes) {
        return new GGUFTensorInfo(name, new long[]{columns, 1}, type.id, 0, rowBytes);
    }

    private static float[] wave(int length) {
        float[] values = new float[length];
        for (int index = 0; index < values.length; index++) {
            float sign = (index & 1) == 0 ? 1.0f : -1.0f;
            values[index] = sign * (0.125f + (index % 11) * 0.03125f);
        }
        return values;
    }

    private static float dotAt(float[] row, float[] vector, int offset) {
        float total = 0.0f;
        for (int index = 0; index < row.length; index++) {
            total += row[index] * vector[offset + index];
        }
        return total;
    }

    private static void assertClose(float expected, float actual) {
        float tolerance = Math.max(1.0e-3f, Math.abs(expected) * 1.0e-5f);
        assertEquals(expected, actual, tolerance);
    }
}
