package tech.kayys.aljabr.gguf.loader;

import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GGUFParserTest {
    @Test
    void parsesModernKnownQuantTensorSizes() {
        GGUFModel model = new GGUFParser().parse(minimalGgufWithSingleTensor(34, 256), null);

        assertEquals(3, model.version());
        assertEquals(1, model.tensors().size());
        GGUFTensorInfo tensor = model.tensors().get(0);
        assertEquals("w", tensor.name());
        assertEquals(34, tensor.typeId());
        assertEquals(54L, tensor.sizeInBytes());
    }

    @Test
    void rejectsRemovedRuntimeRepackedTensorLayouts() {
        MemorySegment segment = minimalGgufWithSingleTensor(31, 32);

        assertThrows(UnsupportedOperationException.class, () -> new GGUFParser().parse(segment, null));
    }

    private static MemorySegment minimalGgufWithSingleTensor(int typeId, long elements) {
        ByteBuffer buffer = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("GGUF".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(3);
        buffer.putLong(1);
        buffer.putLong(0);

        putString(buffer, "w");
        buffer.putInt(1);
        buffer.putLong(elements);
        buffer.putInt(typeId);
        buffer.putLong(0);

        int dataStart = (buffer.position() + 31) & ~31;
        while (buffer.position() < dataStart) {
            buffer.put((byte) 0);
        }
        buffer.flip();
        return MemorySegment.ofBuffer(buffer);
    }

    private static void putString(ByteBuffer buffer, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        buffer.putLong(bytes.length);
        buffer.put(bytes);
    }
}
