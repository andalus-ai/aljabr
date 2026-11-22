package tech.kayys.aljabr.gguf.loader;

import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void preservesOriginalScalarAndArrayMetadataTypes() {
        ByteBuffer buffer = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("GGUF".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(3);
        buffer.putLong(0);
        buffer.putLong(4);

        putString(buffer, "unsigned-byte");
        buffer.putInt(0);
        buffer.put((byte) 0xfe);
        putString(buffer, "signed-byte");
        buffer.putInt(1);
        buffer.put((byte) -7);
        putString(buffer, "labels");
        buffer.putInt(9);
        buffer.putInt(8); // STRING array
        buffer.putLong(2);
        putString(buffer, "alpha");
        putString(buffer, "beta");
        putString(buffer, "dimensions");
        buffer.putInt(9);
        buffer.putInt(2); // UINT16 array
        buffer.putLong(2);
        buffer.putShort((short) 128);
        buffer.putShort((short) 256);

        int dataStart = (buffer.position() + 31) & ~31;
        while (buffer.position() < dataStart) buffer.put((byte) 0);
        buffer.flip();

        GGUFModel model = new GGUFParser().parse(MemorySegment.ofBuffer(buffer), null);

        assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt8Val.class,
                model.typedMetadata().get("unsigned-byte"));
        assertEquals(254, ((tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt8Val)
                model.typedMetadata().get("unsigned-byte")).value());
        assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.Int8Val.class,
                model.typedMetadata().get("signed-byte"));
        var labels = assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.ArrayVal.class,
                model.typedMetadata().get("labels"));
        assertEquals(tech.kayys.aljabr.gguf.core.GgufMetaType.STRING, labels.elementType());
        assertEquals("beta", ((tech.kayys.aljabr.gguf.core.GgufMetaValue.StringVal)
                labels.elements().get(1)).value());
        var dimensions = assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.ArrayVal.class,
                model.typedMetadata().get("dimensions"));
        assertEquals(tech.kayys.aljabr.gguf.core.GgufMetaType.UINT16, dimensions.elementType());
        assertEquals(256, ((tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt16Val)
                dimensions.elements().get(1)).value());
        assertTrue(model.metadata().get("labels") instanceof java.util.List<?>);
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
