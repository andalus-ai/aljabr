package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufHalfFx.writeShorts;

class GgufF16SpecialTest {
    @Test
    void rawF16DequantizePreservesSpecialValues() {
        try (Arena arena = Arena.ofConfined()) {
            short[] values = {
                    (short) 0x0000,
                    (short) 0x8000,
                    (short) 0x7c00,
                    (short) 0xfc00,
                    (short) 0x0001,
                    (short) 0x7e00
            };
            MemorySegment segment = arena.allocate((long) values.length * Short.BYTES);
            writeShorts(segment, values);
            GGUFTensorInfo tensor = new GGUFTensorInfo("f16.special", new long[]{values.length, 1}, 1, 0,
                    (long) values.length * Short.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] row = new float[values.length];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);

            assertEquals(Float.floatToRawIntBits(0.0f), Float.floatToRawIntBits(row[0]));
            assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(row[1]));
            assertEquals(Float.POSITIVE_INFINITY, row[2], 0.0f);
            assertEquals(Float.NEGATIVE_INFINITY, row[3], 0.0f);
            assertEquals((float) Math.scalb(1.0f / 1024.0f, -14), row[4], 0.0f);
            assertTrue(Float.isNaN(row[5]));
        }
    }
}
