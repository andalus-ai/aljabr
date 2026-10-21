package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufDenseTest {
    static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void usesGgufShapeZeroAsMatrixColumnsForF32MatVec() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Float.BYTES);
            float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            for (int i = 0; i < values.length; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, values[i]);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("f32", new long[]{3, 2}, 0, 0, 6L * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVec(model, tensor, new float[]{1.0f, 1.0f, 1.0f}, output);

            assertEquals(6.0f, output[0], 0.0f);
            assertEquals(15.0f, output[1], 0.0f);

            float[] parallelOutput = new float[2];
            GgufTensorOps.matVecParallel(model, tensor, new float[]{1.0f, 1.0f, 1.0f}, parallelOutput);
            assertEquals(6.0f, parallelOutput[0], 0.0f);
            assertEquals(15.0f, parallelOutput[1], 0.0f);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GgufTensorOps.matVec(model, tensor, new float[]{1.0f, 1.0f, 1.0f}, new float[1]));
        }
    }

    @Test
    void rowOpsReuseCheckedLayoutForOffsetsAndBounds() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Float.BYTES);
            float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            for (int i = 0; i < values.length; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, values[i]);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("f32.row.ops", new long[]{3, 2}, 0, 0, 6L * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] row = new float[3];
            GgufTensorOps.dequantizeRow(model, tensor, 1, row);

            assertEquals(4.0f, row[0], 0.0f);
            assertEquals(5.0f, row[1], 0.0f);
            assertEquals(6.0f, row[2], 0.0f);
            assertEquals(15.0f, GgufTensorOps.dotRow(model, tensor, 1, ones(3)), 0.0f);
            assertThrows(IllegalArgumentException.class, () -> GgufTensorOps.dequantizeRow(model, tensor, 2, row));
            assertThrows(IllegalArgumentException.class, () -> GgufTensorOps.dotRow(model, tensor, -1, ones(3)));
        }
    }

    @Test
    void rawDenseMatVecHandlesUnrolledF32RowsWithTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = GgufTensorOps.preferredFloatVectorLanes() * 2 + 3;
            int rows = 2;
            MemorySegment segment = arena.allocate((long) columns * rows * Float.BYTES);
            for (int i = 0; i < columns * rows; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, i + 1.0f);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("f32.unrolled", new long[]{columns, rows}, 0, 0,
                    (long) columns * rows * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[rows];
            GgufTensorOps.matVecRows(model, tensor, ones(columns), output, rows, true);

            assertEquals(columns * (columns + 1) / 2.0f, output[0], 0.0f);
            assertEquals(columns * (3L * columns + 1) / 2.0f, output[1], 0.0f);
        }
    }

    @Test
    void rawDenseF32ScalarDotHandlesWideUnrollAndTail() {
        try (Arena arena = Arena.ofConfined()) {
            int columns = 19;
            MemorySegment segment = arena.allocate((long) columns * Float.BYTES);
            float[] vector = new float[columns];
            float expected = 0.0f;
            for (int i = 0; i < columns; i++) {
                float value = (i % 9 - 4) * 0.5f;
                float input = (i % 7 - 3) * 0.25f;
                segment.set(LE_FLOAT, i * (long) Float.BYTES, value);
                vector[i] = input;
                expected += value * input;
            }

            assertEquals(expected, GgufDenseDot.dotRowF32Scalar(segment, 0, columns, vector, 0), 0.0f);
        }
    }
}
