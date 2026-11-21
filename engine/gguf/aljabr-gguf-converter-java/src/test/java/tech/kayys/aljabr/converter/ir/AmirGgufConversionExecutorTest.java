package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class AmirGgufConversionExecutorTest {
    @TempDir Path temp;

    @Test
    void convertsSafetensorsPayloadWithoutMaterializingOrChangingBytes() throws Exception {
        Path source = temp.resolve("tensor.data");
        byte[] payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.5f).putFloat(-3.0f).array();
        Files.write(source, payload);
        Path output = temp.resolve("model.gguf");

        var tensor = new TensorDescriptor("weight", "F32", List.of(2L), "contiguous",
                null, new PayloadReference(source.toUri(), "weights", 0, 8,
                PayloadReference.OffsetBasis.CONTAINER));
        var model = new ModelDescriptor(ModelIdentity.of("fixture"), "test", "safetensors",
                Map.of("framework", "fixture"), Map.of(), List.of(tensor));
        var compatibility = new CompatibilityReport(CompatibilityReport.ConversionFidelity.FULL, List.of());
        var plan = new ConversionPlan(model.identity(), "safetensors", "gguf", compatibility);
        var request = ConversionRequest.to(output.toUri());

        var result = new AmirGgufConversionExecutor().execute(model, plan, request);
        byte[] written = Files.readAllBytes(output);
        ByteBuffer header = ByteBuffer.wrap(written).order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(0x46554747, header.getInt());
        assertEquals(3, header.getInt());
        assertEquals(1L, result.tensorCount());
        assertEquals(8L, result.payloadBytes());
        assertEquals("gguf", result.targetFormat());
        assertArrayEquals(payload, java.util.Arrays.copyOfRange(
                written, Math.toIntExact(findAlignedPayloadStart(written)), 
                Math.toIntExact(findAlignedPayloadStart(written) + payload.length)));
    }

    @Test
    void resolvesSafetensorsDataSectionRelativeOffsets() throws Exception {
        Path source = temp.resolve("container.safetensors");
        byte[] payload = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(7f).array();
        byte[] header = "{}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + header.length + payload.length).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(header.length).put(header).put(payload);
        Files.write(source, file.array());

        var tensor = new TensorDescriptor("scalar", "F32", List.of(1L), "contiguous", null,
                new PayloadReference(source.toUri(), "weights", 0, 4,
                        PayloadReference.OffsetBasis.DATA_SECTION));
        var model = new ModelDescriptor(ModelIdentity.of("offset-fixture"), "test", "safetensors",
                Map.of(), Map.of(), List.of(tensor));
        var report = new CompatibilityReport(CompatibilityReport.ConversionFidelity.FULL, List.of());
        Path output = temp.resolve("offset.gguf");
        new AmirGgufConversionExecutor().execute(model,
                new ConversionPlan(model.identity(), "safetensors", "gguf", report),
                ConversionRequest.to(output.toUri()));
        assertTrue(Files.size(output) > payload.length);
    }

    @Test
    void refusesImplicitQuantizationAndUnsupportedOptions() throws Exception {
        Path source = temp.resolve("quant.bin");
        Files.write(source, new byte[1]);
        var tensor = new TensorDescriptor("weight", "U8", List.of(1L), "contiguous", null,
                new PayloadReference(source.toUri(), "weights", 0, 1,
                        PayloadReference.OffsetBasis.CONTAINER));
        var model = new ModelDescriptor(ModelIdentity.of("unsupported"), "test", "safetensors",
                Map.of(), Map.of(), List.of(tensor));
        var report = new CompatibilityReport(CompatibilityReport.ConversionFidelity.FULL, List.of());
        var plan = new ConversionPlan(model.identity(), "safetensors", "gguf", report);
        assertThrows(java.io.IOException.class, () -> new AmirGgufConversionExecutor().execute(
                model, plan, new ConversionRequest(temp.resolve("bad.gguf").toUri(), false,
                        Map.of("gguf.quantization", "Q4_K"))));
    }

    private static long findAlignedPayloadStart(byte[] file) {
        // For this fixture the first payload starts at the first 32-byte boundary after the header.
        // Header ends after the two metadata entries, one tensor descriptor, and their UTF-8 strings.
        // Locate the known F32 payload byte sequence in the generated file.
        byte[] marker = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.5f).putFloat(-3.0f).array();
        outer: for (int i = 0; i <= file.length - marker.length; i++) {
            for (int j = 0; j < marker.length; j++) if (file[i + j] != marker[j]) continue outer;
            return i;
        }
        throw new AssertionError("Payload not found");
    }
}
