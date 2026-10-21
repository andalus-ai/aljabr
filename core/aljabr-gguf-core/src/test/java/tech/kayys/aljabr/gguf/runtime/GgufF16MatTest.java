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

class GgufF16MatTest {
    @Test
    void rawDenseMatVecSupportsF16Rows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Short.BYTES);
            short[] values = {
                    (short) 0x3c00, (short) 0x4000, (short) 0x4200,
                    (short) 0x4400, (short) 0x4500, (short) 0x4600
            };
            writeShorts(segment, values);
            GGUFTensorInfo tensor = new GGUFTensorInfo("f16", new long[]{3, 2}, 1, 0, 6L * Short.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(3), output, 2, true);

            assertEquals(6.0f, output[0], 0.0f);
            assertEquals(15.0f, output[1], 0.0f);
        }
    }

    @Test
    void rawDenseMatVecHandlesUnrolledF16RowsWithTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 10;
            int rows = 2;
            MemorySegment segment = arena.allocate((long) columns * rows * Short.BYTES);
            for (int i = 0; i < columns; i++) {
                segment.set(LE_SHORT, i * (long) Short.BYTES, (short) 0x3c00);
                segment.set(LE_SHORT, (columns + i) * (long) Short.BYTES, (short) 0x4000);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("f16.unrolled", new long[]{columns, rows}, 1, 0,
                    (long) columns * rows * Short.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[rows];
            GgufTensorOps.matVecRows(model, tensor, ramp(columns), output, rows, true);

            assertEquals(55.0f, output[0], 0.0f);
            assertEquals(110.0f, output[1], 0.0f);
        }
    }

    @Test
    void rawDenseF16DotHandlesWideUnrollAndTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 19;
            MemorySegment segment = arena.allocate((long) columns * Short.BYTES);
            short[] values = new short[columns];
            float[] vector = new float[columns];
            float expected = 0.0f;
            for (int i = 0; i < columns; i++) {
                float value = (i % 7 - 3) * 0.5f;
                values[i] = Float.floatToFloat16(value);
                vector[i] = (i % 5 - 2) * 0.25f;
                expected += Float.float16ToFloat(values[i]) * vector[i];
            }
            writeShorts(segment, values);

            assertEquals(expected, GgufDenseDot.dotRowF16(segment, 0, columns, vector, 0), 0.0f);
        }
    }
}
