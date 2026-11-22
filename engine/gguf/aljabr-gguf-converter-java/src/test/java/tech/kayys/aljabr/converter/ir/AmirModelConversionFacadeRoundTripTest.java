package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.ArchitectureConfiguration;
import tech.kayys.aljabr.spi.model.ir.ArchitectureDescriptor;
import tech.kayys.aljabr.spi.model.ir.ArchitectureFamily;
import tech.kayys.aljabr.spi.model.ir.ArchitectureKind;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionFacade;
import tech.kayys.aljabr.spi.model.ir.ModelConversionPlanner;
import tech.kayys.aljabr.spi.model.ir.TensorDataType;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorLayout;
import tech.kayys.aljabr.spi.model.ir.TensorShape;
import tech.kayys.aljabr.spi.model.ir.TensorType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AmirModelConversionFacadeRoundTripTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void convertsSafetensorsToGgufAndBackThroughFacadeWithoutChangingPayload() throws Exception {
        byte[] expectedPayload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.25f).putFloat(-2.5f).array();
        Path source = writeSafetensors("source.safetensors", expectedPayload);
        Path gguf = temporaryDirectory.resolve("round-trip.gguf");
        Path restored = temporaryDirectory.resolve("restored.safetensors");
        ModelConversionFacade facade = facade();

        var forward = facade.prepare(source, "safetensors", "gguf");
        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                forward.plan().compatibility().fidelity());
        var forwardResult = facade.execute(forward, ConversionRequest.to(gguf.toUri()));
        assertEquals(1, forwardResult.tensorCount());
        assertEquals(expectedPayload.length, forwardResult.payloadBytes());
        assertTrue(Files.isRegularFile(gguf));

        var reverse = facade.prepare(gguf, "gguf", "safetensors");
        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                reverse.plan().compatibility().fidelity());
        assertEquals("original-value", reverse.model().metadata().get("vendor.metadata/key"));
        assertEquals("extension-value",
                reverse.model().tensors().get(0).extensions().get("vendor.key/雪"));
        assertEquals(expectedTensorType(),
                reverse.model().tensors().get(0).tensorType().orElseThrow());
        assertTrue(reverse.model().metadata().keySet().stream()
                .noneMatch(AmirTensorTypeCodec::isEncodedKey));
        var reverseResult = facade.execute(reverse, ConversionRequest.to(restored.toUri()));
        assertEquals(1, reverseResult.tensorCount());
        assertEquals(expectedPayload.length, reverseResult.payloadBytes());

        var restoredDescriptor = new AmirModelDescriptorReader().read(restored);
        assertEquals(expectedTensorType(),
                restoredDescriptor.tensors().get(0).tensorType().orElseThrow());

        byte[] file = Files.readAllBytes(restored);
        ByteBuffer prefix = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = prefix.getLong();
        assertTrue(headerLength > 0 && headerLength <= file.length - Long.BYTES);
        JsonObject header = JsonParser.parseString(new String(
                file, Long.BYTES, Math.toIntExact(headerLength), StandardCharsets.UTF_8)).getAsJsonObject();
        JsonObject tensor = header.getAsJsonObject("weight");
        assertEquals("F32", tensor.get("dtype").getAsString());
        assertEquals(List.of(2), tensor.getAsJsonArray("shape").asList().stream()
                .map(value -> value.getAsInt()).toList());
        int dataStart = Math.toIntExact(Long.BYTES + headerLength);
        assertArrayEquals(expectedPayload,
                java.util.Arrays.copyOfRange(file, dataStart, dataStart + expectedPayload.length));
    }

    @Test
    void safetensorsToSafetensorsPreservesTypedTensorInformation() throws Exception {
        byte[] payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(4.0f).putFloat(5.0f).array();
        Path source = writeSafetensors("typed-source.safetensors", payload, false);
        Path output = temporaryDirectory.resolve("typed-copy.safetensors");
        ModelConversionFacade facade = facade();

        var prepared = facade.prepare(source, "safetensors", "safetensors");
        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                prepared.plan().compatibility().fidelity());
        facade.execute(prepared, ConversionRequest.to(output.toUri()));

        var restored = new AmirModelDescriptorReader().read(output);
        assertEquals(expectedTensorType(), restored.tensors().get(0).tensorType().orElseThrow());
    }

    @Test
    void safetensorsGgufSafetensorsRoundTripPreservesTypedArchitectureDescriptor() throws Exception {
        byte[] payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.0f).putFloat(2.0f).array();
        ArchitectureDescriptor architecture = new ArchitectureDescriptor(
                "llama",
                new ArchitectureFamily("transformer.decoder", ArchitectureKind.DECODER_ONLY_TRANSFORMER),
                new ArchitectureConfiguration(
                        Map.of(
                                "hidden-size", new ConfigurationValue.IntegerValue(4096L),
                                "rms-epsilon", new ConfigurationValue.DecimalValue(new java.math.BigDecimal("0.00000100"))),
                        Map.of("vendor.config", "opaque")),
                java.util.Set.of("causal-attention"),
                Map.of("vendor.architecture", "custom"));
        Path source = writeSafetensors("architecture-source.safetensors", payload, false, architecture);
        Path gguf = temporaryDirectory.resolve("architecture.gguf");
        Path restoredFile = temporaryDirectory.resolve("architecture-restored.safetensors");
        ModelConversionFacade facade = facade();

        var forward = facade.prepare(source, "safetensors", "gguf");
        assertEquals(CompatibilityReport.ConversionFidelity.FULL, forward.plan().compatibility().fidelity());
        facade.execute(forward, ConversionRequest.to(gguf.toUri()));

        var fromGguf = new AmirModelDescriptorReader().read(gguf);
        assertEquals(architecture, fromGguf.architectureDescriptor().orElseThrow());

        var reverse = facade.prepare(gguf, "gguf", "safetensors");
        assertEquals(CompatibilityReport.ConversionFidelity.FULL, reverse.plan().compatibility().fidelity());
        facade.execute(reverse, ConversionRequest.to(restoredFile.toUri()));

        var restored = new AmirModelDescriptorReader().read(restoredFile);
        assertEquals(architecture, restored.architectureDescriptor().orElseThrow());
    }

    @Test
    void unsupportedTargetIsReportedBeforeAnyOutputIsCreated() throws Exception {
        byte[] payload = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(3.0f).array();
        Path source = writeSafetensors("unsupported.safetensors", payload);
        Path output = temporaryDirectory.resolve("must-not-exist.bin");
        ModelConversionFacade facade = facade();

        var prepared = facade.prepare(source, "safetensors", "unknown-format");
        assertEquals(CompatibilityReport.ConversionFidelity.UNSUPPORTED,
                prepared.plan().compatibility().fidelity());
        assertThrows(IllegalStateException.class,
                () -> facade.execute(prepared, ConversionRequest.to(output.toUri())));
        assertFalse(Files.exists(output));
    }

    private ModelConversionFacade facade() {
        return new ModelConversionFacade(
                List.of(new AmirModelDescriptorReader()),
                new ModelConversionPlanner(List.of(
                        new GgufTargetCapability(),
                        new SafetensorsTargetCapability())),
                List.of(
                        new AmirSafetensorsConversionExecutor(),
                        new AmirGgufConversionExecutor(),
                        new AmirGgufToSafetensorsConversionExecutor()));
    }

    private static TensorType expectedTensorType() {
        return TensorType.of(TensorDataType.F32, TensorShape.of(2))
                .withLayout(TensorLayout.CONTIGUOUS);
    }

    private Path writeSafetensors(String name, byte[] payload) throws Exception {
        return writeSafetensors(name, payload, true, null);
    }

    private Path writeSafetensors(String name, byte[] payload, boolean includeOpaqueExtensions) throws Exception {
        return writeSafetensors(name, payload, includeOpaqueExtensions, null);
    }

    private Path writeSafetensors(String name, byte[] payload, boolean includeOpaqueExtensions,
            ArchitectureDescriptor architectureDescriptor) throws Exception {
        JsonObject tensor = new JsonObject();
        tensor.addProperty("dtype", "F32");
        var shape = new com.google.gson.JsonArray();
        shape.add(2);
        tensor.add("shape", shape);
        var offsets = new com.google.gson.JsonArray();
        offsets.add(0);
        offsets.add(payload.length);
        tensor.add("data_offsets", offsets);

        JsonObject metadata = new JsonObject();
        metadata.addProperty("general.architecture", "llama");
        if (architectureDescriptor != null) {
            metadata.addProperty(AmirArchitectureDescriptorCodec.KEY,
                    AmirArchitectureDescriptorCodec.encode(architectureDescriptor));
        }
        if (includeOpaqueExtensions) {
            metadata.addProperty("aljabr.source.metadata.vendor.metadata/key", "original-value");
            String tensorName = java.util.Base64.getUrlEncoder().withoutPadding()
                    .encodeToString("weight".getBytes(StandardCharsets.UTF_8));
            String extensionName = java.util.Base64.getUrlEncoder().withoutPadding()
                    .encodeToString("vendor.key/雪".getBytes(StandardCharsets.UTF_8));
            metadata.addProperty("aljabr.source.tensor." + tensorName + "." + extensionName, "extension-value");
        }
        AmirTensorTypeCodec.encode(List.of(TensorDescriptor.typed(
                "weight", expectedTensorType(), null, Map.of())))
                .forEach(metadata::addProperty);
        JsonObject root = new JsonObject();
        root.add("weight", tensor);
        root.add("__metadata__", metadata);
        String json = root.toString();
        byte[] header = json.getBytes(StandardCharsets.UTF_8);
        int padding = (int) ((8 - (Long.BYTES + header.length) % 8) % 8);
        byte[] padded = java.util.Arrays.copyOf(header, header.length + padding);
        java.util.Arrays.fill(padded, header.length, padded.length, (byte) ' ');
        ByteBuffer file = ByteBuffer.allocate(Long.BYTES + padded.length + payload.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(padded.length).put(padded).put(payload);
        Path source = temporaryDirectory.resolve(name);
        Files.write(source, file.array());
        return source;
    }
}
