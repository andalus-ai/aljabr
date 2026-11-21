package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptorReaders;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AmirModelDescriptorReaderTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void discoversSafetensorsAndCreatesLazyPayloadReferences() throws IOException {
        Path source = temporaryDirectory.resolve("tiny.safetensors");
        byte[] payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.25f).putFloat(-2.5f).array();
        String header = """
                {"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
                 "__metadata__":{"general.architecture":"llama"}}
                """.replaceAll("\\s+", "");
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + headerBytes.length + payload.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(headerBytes.length).put(headerBytes).put(payload);
        Files.write(source, file.array());

        ModelDescriptorReader reader = ModelDescriptorReaders.find(
                ModelDescriptorReaders.discover(), "safetensors").orElseThrow();
        ModelDescriptor descriptor = reader.read(source);

        assertEquals("SAFETENSORS", descriptor.sourceFormat());
        assertEquals("llama", descriptor.architecture());
        assertEquals(1, descriptor.tensors().size());
        var tensor = descriptor.tensors().getFirst();
        assertEquals("weight", tensor.name());
        assertEquals("F32", tensor.dataType());
        assertEquals(List.of(2L), tensor.shape());
        assertEquals(8L, tensor.payload().length());
        assertEquals(0L, tensor.payload().offset());
        assertEquals(PayloadReference.OffsetBasis.DATA_SECTION, tensor.payload().offsetBasis());
        assertEquals(source.toAbsolutePath().normalize().toUri(), tensor.payload().source());
    }

    @Test
    void rejectsSafetensorsPayloadOutsideFile() throws IOException {
        Path source = temporaryDirectory.resolve("invalid.safetensors");
        byte[] headerBytes = """
                {"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + headerBytes.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(headerBytes.length).put(headerBytes);
        Files.write(source, file.array());

        assertThrows(IOException.class, () -> new AmirModelDescriptorReader().read(source));
    }
}
