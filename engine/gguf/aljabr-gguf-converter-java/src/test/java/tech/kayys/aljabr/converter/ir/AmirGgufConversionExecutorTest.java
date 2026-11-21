package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.converter.java.gguf.GgufReader;
import tech.kayys.aljabr.gguf.core.GgufModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutors;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class AmirGgufConversionExecutorTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void discoversAsTargetFormatExecutor() {
        var executor = new AmirGgufConversionExecutor();
        assertTrue(executor.supports("safetensors", "gguf"));
        assertTrue(executor.supports("gguf", "GGUF"));
        assertFalse(executor.supports("safetensors", "safetensors"));
    }

    @Test
    void discoversProviderThroughModelConversionExecutorService() {
        assertInstanceOf(AmirGgufConversionExecutor.class,
                ModelConversionExecutors.find("safetensors", "gguf").orElseThrow());
    }

    @Test
    void streamsSafetensorsPayloadIntoGgufAndCarriesMetadata() throws Exception {
        Path source = writeSafetensors();
        var payload = new PayloadReference(source.toUri(), source.getFileName().toString(),
                0L, 8L, PayloadReference.OffsetBasis.DATA_SECTION);
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L), "unknown", null, payload,
                Map.of("vendor.layout", "row-major"));
        var model = new ModelDescriptor(ModelIdentity.of("tiny"), "decoder", "safetensors",
                Map.of("model_type", "decoder"), Map.of("vendor.source", "checkpoint-v1"), List.of(tensor));
        var plan = plan(model, CompatibilityReport.ConversionFidelity.FULL);
        Path output = temporaryDirectory.resolve("out/model.gguf");

        var result = new AmirGgufConversionExecutor().execute(
                model, plan, ConversionRequest.to(output.toUri()));

        assertEquals("safetensors", result.sourceFormat());
        assertEquals("gguf", result.targetFormat());
        assertEquals(output.toAbsolutePath().normalize().toUri(), result.output());
        assertEquals(1, result.tensorCount());
        assertEquals(8L, result.payloadBytes());
        assertTrue(result.untransferredMetadataKeys().isEmpty());

        byte[] written = Files.readAllBytes(output);
        ByteBuffer header = ByteBuffer.wrap(written).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x46554747, header.getInt());
        assertEquals(3, header.getInt());
        assertEquals(1L, header.getLong());
        assertTrue(new String(written, java.nio.charset.StandardCharsets.ISO_8859_1)
                .contains("aljabr.source.metadata.model_type"));
        long dataStart = dataStart(written);
        assertArrayEquals(new byte[] {1, 2, 3, 4, 5, 6, 7, 8},
                java.util.Arrays.copyOfRange(written, Math.toIntExact(dataStart),
                        Math.toIntExact(dataStart + 8)));
    }

    @Test
    void partialPlanRequiresAcknowledgementAndRetainsOpaqueExtensions() throws Exception {
        Path source = writeSafetensors();
        var payload = new PayloadReference(source.toUri(), source.getFileName().toString(),
                0L, 8L, PayloadReference.OffsetBasis.DATA_SECTION);
        var model = new ModelDescriptor(ModelIdentity.of("partial"), "unknown", "safetensors",
                Map.of(), Map.of("vendor.note", "keep-as-metadata"),
                List.of(new TensorDescriptor("weight", "F32", List.of(2L), "unknown", null, payload)));
        var executor = new AmirGgufConversionExecutor();
        Path output = temporaryDirectory.resolve("partial.gguf");
        var plan = plan(model, CompatibilityReport.ConversionFidelity.PARTIAL);

        assertThrows(java.io.IOException.class, () -> executor.execute(
                model, plan, ConversionRequest.to(output.toUri())));
        var request = new ConversionRequest(output.toUri(), true, Map.of());
        var result = executor.execute(model, plan, request);

        assertTrue(Files.exists(output));
        assertTrue(result.untransferredMetadataKeys().isEmpty(),
                "opaque string extensions are retained as namespaced GGUF metadata");
    }

    @Test
    void rejectsPayloadByteLengthMismatchBeforeCreatingOutput() throws Exception {
        Path source = writeSafetensors();
        var payload = new PayloadReference(source.toUri(), source.getFileName().toString(),
                0L, 4L, PayloadReference.OffsetBasis.DATA_SECTION);
        var model = new ModelDescriptor(ModelIdentity.of("bad"), "decoder", "safetensors",
                Map.of(), Map.of(), List.of(
                        new TensorDescriptor("weight", "F32", List.of(2L), "unknown", null, payload)));
        Path output = temporaryDirectory.resolve("invalid.gguf");

        assertThrows(java.io.IOException.class, () -> new AmirGgufConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                ConversionRequest.to(output.toUri())));
        assertFalse(Files.exists(output));
    }

    private Path writeSafetensors() throws Exception {
        byte[] header = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}"
                .getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int padding = (int) ((8 - (Long.BYTES + header.length) % 8) % 8);
        byte[] padded = java.util.Arrays.copyOf(header, header.length + padding);
        java.util.Arrays.fill(padded, header.length, padded.length, (byte) ' ');
        ByteBuffer file = ByteBuffer.allocate(Long.BYTES + padded.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(padded.length).put(padded).put(new byte[] {1, 2, 3, 4, 5, 6, 7, 8});
        Path path = temporaryDirectory.resolve("source.safetensors");
        Files.write(path, file.array());
        return path;
    }

    private static long dataStart(byte[] gguf) {
        ByteBuffer buffer = ByteBuffer.wrap(gguf).order(ByteOrder.LITTLE_ENDIAN);
        buffer.position(16);
        long metadataCount = buffer.getLong();
        for (long i = 0; i < metadataCount; i++) {
            skipString(buffer);
            int type = buffer.getInt();
            switch (type) {
                case 0, 1, 7 -> buffer.position(buffer.position() + 1);
                case 2, 3 -> buffer.position(buffer.position() + 2);
                case 4, 5, 6 -> buffer.position(buffer.position() + 4);
                case 8 -> skipString(buffer);
                case 9 -> throw new AssertionError("unexpected metadata array");
                case 10, 11, 12 -> buffer.position(buffer.position() + 8);
                default -> throw new AssertionError("unknown metadata type " + type);
            }
        }
        long tensorCount = ByteBuffer.wrap(gguf).order(ByteOrder.LITTLE_ENDIAN).getLong(8);
        for (long i = 0; i < tensorCount; i++) {
            skipString(buffer);
            int dimensions = buffer.getInt();
            buffer.position(buffer.position() + dimensions * Long.BYTES + Integer.BYTES + Long.BYTES);
        }
        return (buffer.position() + 31L) & ~31L;
    }

    private static void skipString(ByteBuffer buffer) {
        long length = buffer.getLong();
        buffer.position(Math.toIntExact(buffer.position() + length));
    }

    private static ConversionPlan plan(ModelDescriptor model, CompatibilityReport.ConversionFidelity fidelity) {
        var diagnostics = fidelity == CompatibilityReport.ConversionFidelity.PARTIAL
                ? List.of(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.WARNING, "OPAQUE_METADATA",
                        "Opaque extensions are carried in namespaced GGUF metadata"))
                : List.<CompatibilityReport.Diagnostic>of();
        return new ConversionPlan(model.identity(), model.sourceFormat(), "gguf",
                new CompatibilityReport(fidelity, diagnostics));
    }
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
