package tech.kayys.aljabr.converter.ir;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufModelDocumentAdapterTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void inspectsTensorInventoryWithoutCopyingPayloadIntoAmir() throws Exception {
        Path model = temporaryDirectory.resolve("tiny.gguf");
        ByteBuffer bytes = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);

        bytes.put("GGUF".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(3);
        bytes.putLong(1); // tensor count
        bytes.putLong(1); // metadata count

        putString(bytes, "general.architecture");
        bytes.putInt(8); // GGUF string metadata type
        putString(bytes, "llama");

        putString(bytes, "weight");
        bytes.putInt(2);
        bytes.putLong(2);
        bytes.putLong(2);
        bytes.putInt(0); // F32
        bytes.putLong(0); // tensor data-section offset

        int dataStart = (bytes.position() + 31) & ~31;
        while (bytes.position() < dataStart) {
            bytes.put((byte) 0);
        }
        bytes.putFloat(1.0f);
        bytes.putFloat(2.0f);
        bytes.putFloat(3.0f);
        bytes.putFloat(4.0f);
        Files.write(model, java.util.Arrays.copyOf(bytes.array(), bytes.position()));

        var document = new GgufModelDocumentAdapter(new ObjectMapper()).inspect(model);

        assertEquals("tiny", document.id());
        assertEquals("llama", document.architecture());
        assertEquals("gguf", document.sourceFormat());
        assertTrue(document.extensions().containsKey("gguf.metadata.json"));
        assertEquals(1, document.tensors().size());

        var tensor = document.tensors().getFirst();
        assertEquals("weight", tensor.name());
        assertEquals("F32", tensor.dataType());
        assertEquals(java.util.List.of(2L, 2L), tensor.shape());
        assertEquals(dataStart, tensor.payload().offset());
        assertEquals(16L, tensor.payload().length());
        assertEquals(PayloadReference.OffsetBasis.CONTAINER, tensor.payload().offsetBasis());
    }

    private static void putString(ByteBuffer buffer, String value) {
        byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
        buffer.putLong(encoded.length);
        buffer.put(encoded);
    }
}
