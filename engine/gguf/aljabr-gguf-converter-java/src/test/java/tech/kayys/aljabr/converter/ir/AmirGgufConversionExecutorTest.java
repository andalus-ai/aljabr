package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.*;
import tech.kayys.aljabr.converter.java.gguf.GgufReader;
import tech.kayys.aljabr.gguf.core.GgufModel;

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

        // Validate the complete output with Aljabr's existing GGUF reader,
        // not only by scanning for the source byte sequence.
        try (GgufReader reader = new GgufReader(output)) {
            GgufModel parsed = reader.read();
            var parsedTensor = parsed.findTensor("weight").orElseThrow();
            assertEquals(tech.kayys.aljabr.gguf.core.GgmlType.F32, parsedTensor.type());
            assertArrayEquals(new long[] {2L}, parsedTensor.shape());
            assertEquals(payload.length, parsedTensor.dataSize());
            assertArrayEquals(payload, java.util.Arrays.copyOfRange(
                    parsed.tensorData(), Math.toIntExact(parsedTensor.dataOffset()),
                    Math.toIntExact(parsedTensor.dataOffset() + parsedTensor.dataSize())));
            assertEquals("test", parsed.architecture());
        }
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
    void convertsAmirScalarShapeToExplicitSingleElementGgufDimension() throws Exception {
        Path source = temp.resolve("scalar.bin");
        byte[] payload = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(2.5f).array();
        Files.write(source, payload);
        var tensor = new TensorDescriptor("scalar", "F32", List.of(), "contiguous", null,
                new PayloadReference(source.toUri(), "weights", 0, payload.length,
                        PayloadReference.OffsetBasis.CONTAINER));
        var model = new ModelDescriptor(ModelIdentity.of("scalar-fixture"), "test", "safetensors",
                Map.of(), Map.of(), List.of(tensor));
        var report = new CompatibilityReport(CompatibilityReport.ConversionFidelity.FULL, List.of());
        Path output = temp.resolve("scalar.gguf");

        var result = new AmirGgufConversionExecutor().execute(model,
                new ConversionPlan(model.identity(), "safetensors", "gguf", report),
                ConversionRequest.to(output.toUri()));

        assertEquals(1L, result.tensorCount());
        assertEquals(4L, result.payloadBytes());
        try (GgufReader reader = new GgufReader(output)) {
            GgufModel parsed = reader.read();
            var parsedTensor = parsed.findTensor("scalar").orElseThrow();
            assertArrayEquals(new long[] {1L}, parsedTensor.shape());
            assertEquals(4L, parsedTensor.dataSize());
            assertArrayEquals(payload, java.util.Arrays.copyOfRange(
                    parsed.tensorData(), Math.toIntExact(parsedTensor.dataOffset()),
                    Math.toIntExact(parsedTensor.dataOffset() + parsedTensor.dataSize())));
        }
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

}
