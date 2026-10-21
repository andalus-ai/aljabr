package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufHalfFx.LE_SHORT;
import static tech.kayys.aljabr.gguf.runtime.GgufHalfFx.writeShorts;

class GgufBf16Test {
    @Test
    void rawDenseMatVecSupportsBf16Rows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Short.BYTES);
            short[] values = {
                    (short) 0x3f80, (short) 0x4000, (short) 0x4040,
                    (short) 0x4080, (short) 0x40a0, (short) 0x40c0
            };
            writeShorts(segment, values);
            GGUFTensorInfo tensor = new GGUFTensorInfo("bf16", new long[]{3, 2}, 30, 0, 6L * Short.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(3), output, 2, true);

            assertEquals(6.0f, output[0], 0.0f);
            assertEquals(15.0f, output[1], 0.0f);
        }
    }

    @Test
    void rawDenseMatVecHandlesUnrolledBf16RowsWithTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = GgufTensorOps.preferredFloatVectorLanes() * 2 + 3;
            int rows = 2;
            MemorySegment segment = arena.allocate((long) columns * rows * Short.BYTES);
            for (int i = 0; i < columns; i++) {
                segment.set(LE_SHORT, i * (long) Short.BYTES, (short) 0x3f80);
                segment.set(LE_SHORT, (columns + i) * (long) Short.BYTES, (short) 0x4000);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("bf16.unrolled", new long[]{columns, rows}, 30, 0,
                    (long) columns * rows * Short.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[rows];
            GgufTensorOps.matVecRows(model, tensor, ramp(columns), output, rows, true);

            assertEquals(columns * (columns + 1) / 2.0f, output[0], 0.0f);
            assertEquals(columns * (columns + 1), output[1], 0.0f);
        }
    }

    @Test
    void rawDenseBf16ScalarDotHandlesWideUnrollAndTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 19;
            MemorySegment segment = arena.allocate((long) columns * Short.BYTES);
            short[] values = new short[columns];
            float[] vector = new float[columns];
            float expected = 0.0f;
            for (int i = 0; i < columns; i++) {
                float value = (i % 9 - 4) * 0.75f;
                values[i] = f32ToBf16(value);
                vector[i] = (i % 7 - 3) * 0.125f;
                expected += bf16ToF32(values[i]) * vector[i];
            }
            writeShorts(segment, values);

            assertEquals(expected, GgufDenseDot.dotRowBF16Scalar(segment, 0, columns, vector, 0), 0.0f);
        }
    }

    private static short f32ToBf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    private static float bf16ToF32(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }
}
