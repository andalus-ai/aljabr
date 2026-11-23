package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.AssetCollection;
import tech.kayys.aljabr.spi.model.ir.AssetReference;
import tech.kayys.aljabr.spi.model.ir.ComponentCollection;
import tech.kayys.aljabr.spi.model.ir.GraphCollection;
import tech.kayys.aljabr.spi.model.ir.GraphDescriptor;
import tech.kayys.aljabr.spi.model.ir.GraphSignature;
import tech.kayys.aljabr.spi.model.ir.ModelComponent;
import tech.kayys.aljabr.spi.model.ir.ModelComponentKind;
import tech.kayys.aljabr.spi.model.ir.AssetCollection;
import tech.kayys.aljabr.spi.model.ir.AssetReference;
import tech.kayys.aljabr.spi.model.ir.ComponentCollection;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;
import tech.kayys.aljabr.spi.model.ir.ModelComponent;
import tech.kayys.aljabr.spi.model.ir.ModelComponentKind;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionExecutionResult;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutor;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutors;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AmirSafetensorsConversionExecutorTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void streamsDataSectionPayloadAndPreservesSafetensorsMetadata() throws Exception {
        Path source = writeSource("model.safetensors");
        byte[] payload = new byte[] {1, 2, 3, 4, 5, 6, 7, 8};
        ModelDescriptor model = model(source, Map.of("safetensors.format", "pt"), Map.of());
        ConversionPlan plan = plan(model, CompatibilityReport.ConversionFidelity.FULL);
        Path output = temporaryDirectory.resolve("out/model.safetensors");

        var result = new AmirSafetensorsConversionExecutor().execute(model, plan, output, false);

        assertEquals(1, result.tensorCount());
        assertEquals(8L, result.payloadBytes());
        assertEquals(output.toAbsolutePath().normalize(), result.output());
        assertTrue(result.untransferredExtensionKeys().isEmpty());
        byte[] written = Files.readAllBytes(output);
        long headerLength = ByteBuffer.wrap(written, 0, Long.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN).getLong();
        String header = new String(written, Long.BYTES, Math.toIntExact(headerLength), StandardCharsets.UTF_8);
        assertTrue(header.contains("\"format\":\"pt\""));
        assertArrayEquals(payload, java.util.Arrays.copyOfRange(
                written, Math.toIntExact(Long.BYTES + headerLength), written.length));
    }

    @Test
    void executesThroughFormatNeutralExecutorContract() throws Exception {
        Path source = writeSource("spi-source.safetensors");
        ModelDescriptor model = model(source, Map.of(), Map.of());
        ConversionPlan plan = plan(model, CompatibilityReport.ConversionFidelity.FULL);
        ModelConversionExecutor executor = new AmirSafetensorsConversionExecutor();
        Path output = temporaryDirectory.resolve("spi-output.safetensors");

        assertTrue(executor.supports("SAFETENSORS", "safetensors"));
        assertFalse(executor.supports("gguf", "safetensors"));

        ConversionExecutionResult result = executor.execute(
                model, plan, ConversionRequest.to(output.toUri()));

        assertEquals("safetensors", result.sourceFormat());
        assertEquals("safetensors", result.targetFormat());
        assertEquals(output.toAbsolutePath().normalize().toUri(), result.output());
        assertEquals(1, result.tensorCount());
        assertEquals(8L, result.payloadBytes());
        assertEquals(CompatibilityReport.ConversionFidelity.FULL, result.compatibility().fidelity());
        assertTrue(result.untransferredMetadataKeys().isEmpty());
    }

    @Test
    void discoversSafetensorsExecutorThroughServiceLoader() {
        var executor = ModelConversionExecutors.find("safetensors", "safetensors").orElseThrow();

        assertInstanceOf(AmirSafetensorsConversionExecutor.class, executor);
    }

    @Test
    void rejectsOptionsThatThisTargetExecutorDoesNotUnderstand() throws Exception {
        Path source = writeSource("options-source.safetensors");
        ModelDescriptor model = model(source, Map.of(), Map.of());
        ConversionPlan plan = plan(model, CompatibilityReport.ConversionFidelity.FULL);
        ConversionRequest request = new ConversionRequest(
                temporaryDirectory.resolve("options-output.safetensors").toUri(),
                false, Map.of("safetensors.unknown", "true"));

        assertThrows(java.io.IOException.class,
                () -> new AmirSafetensorsConversionExecutor().execute(model, plan, request));
    }

    @Test
    void partialPlanRequiresAcknowledgementAndReportsUntransferredExtensions() throws Exception {
        Path source = writeSource("partial.safetensors");
        ModelDescriptor model = model(source, Map.of(), Map.of("vendor.config", "{\"x\":1}"));
        ConversionPlan plan = plan(model, CompatibilityReport.ConversionFidelity.PARTIAL);
        Path output = temporaryDirectory.resolve("partial-output.safetensors");
        AmirSafetensorsConversionExecutor executor = new AmirSafetensorsConversionExecutor();

        assertThrows(java.io.IOException.class, () -> executor.execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL), output, false));
        assertThrows(java.io.IOException.class, () -> executor.execute(model, plan, output, false));
        assertFalse(Files.exists(output));

        var result = executor.execute(model, plan, output, true);
        assertEquals(List.of("model:vendor.config"), result.untransferredExtensionKeys());
        assertTrue(Files.exists(output));
    }

    @Test
    void tensorExtensionsAlsoRequirePartialFidelityAndAreReported() throws Exception {
        Path source = writeSource("tensor-extension.safetensors");
        var payload = new PayloadReference(
                source.toUri(), source.getFileName().toString(), 0L, 8L,
                PayloadReference.OffsetBasis.DATA_SECTION);
        var tensor = new TensorDescriptor(
                "weight", "F32", List.of(2L), "unknown", null, payload,
                Map.of("vendor.layout_hint", "blocked"));
        var model = new ModelDescriptor(
                ModelIdentity.of("tensor-extension"), "test-architecture", "safetensors",
                Map.of(), Map.of(), List.of(tensor));
        var executor = new AmirSafetensorsConversionExecutor();
        var output = temporaryDirectory.resolve("tensor-extension-output.safetensors");

        assertThrows(java.io.IOException.class, () -> executor.execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL), output, false));

        var result = executor.execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL), output, true);
        assertEquals(List.of("tensor:weight:vendor.layout_hint"), result.untransferredExtensionKeys());
    }

    @Test
    void typedModelLayersRequirePartialFidelityAndAreReported() throws Exception {
        Path source = writeSource("typed-model-layers.safetensors");
        ModelDescriptor model = model(source, Map.of(), Map.of())
                .withGraphCollection(new GraphCollection(List.of(
                        GraphDescriptor.of("forward", List.of(), List.of(), GraphSignature.empty()))))
                .withAssetCollection(new AssetCollection(List.of(
                        AssetReference.of("vocab", "assets/tokenizer.json"))))
                .withComponentCollection(new ComponentCollection(List.of(
                        ModelComponent.of("tokenizer", ModelComponentKind.TOKENIZER))));
        var executor = new AmirSafetensorsConversionExecutor();
        var output = temporaryDirectory.resolve("typed-model-layers-output.safetensors");

        assertThrows(java.io.IOException.class, () -> executor.execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL), output, false));

        var result = executor.execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL), output, true);
        assertEquals(List.of(
                "model:graphCollection",
                "model:assetCollection",
                "model:componentCollection"), result.untransferredExtensionKeys());
    }

    @Test
    void roundTripsTypedAssetsAndComponentsThroughSafetensors() throws Exception {
        Path source = writeSource("model-layers-source.safetensors");
        AssetReference tokenizerAsset = new AssetReference(
                "tokenizer", "assets/tokenizer.json",
                java.util.Optional.of("application/json"), java.util.OptionalLong.of(128L),
                java.util.Optional.of("abc123"), java.util.Optional.of("sha256"),
                Map.of("vendor.role", "vocabulary"));
        ModelDescriptor model = model(source, Map.of(), Map.of())
                .withAssetCollection(new AssetCollection(List.of(tokenizerAsset)))
                .withComponentCollection(new ComponentCollection(List.of(new ModelComponent(
                        "text-tokenizer", ModelComponentKind.TOKENIZER,
                        Map.of("normalization", new ConfigurationValue.StringValue("NFKC"),
                                "revision", new ConfigurationValue.IntegerValue(7L),
                                "temperature", new ConfigurationValue.DecimalValue(
                                        new java.math.BigDecimal("0.1250")),
                                "useCache", new ConfigurationValue.BooleanValue(true),
                                "specialTokens", new ConfigurationValue.StringListValue(List.of("bos", "eos")),
                                "tokenIds", new ConfigurationValue.IntegerListValue(List.of(1L, 2L))),
                        List.of("tokenizer"), List.of(), Map.of("vendor.impl", "custom")))));

        Path output = temporaryDirectory.resolve("model-layers-roundtrip.safetensors");
        new AmirSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL), output, false);

        ModelDescriptor restored = new AmirModelDescriptorReader().read(output);
        assertEquals(model.assetCollection(), restored.assetCollection());
        assertEquals(model.componentCollection(), restored.componentCollection());
        assertFalse(restored.metadata().containsKey(AmirModelLayerCodec.ASSETS_KEY));
        assertFalse(restored.metadata().containsKey(AmirModelLayerCodec.COMPONENTS_KEY));
    }

    private ModelDescriptor model(Path source, Map<String, String> metadata, Map<String, String> extensions) {
        var payload = new PayloadReference(
                source.toUri(), source.getFileName().toString(), 0L, 8L,
                PayloadReference.OffsetBasis.DATA_SECTION);
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L), "unknown", null, payload);
        return new ModelDescriptor(
                ModelIdentity.of("tiny"), "test-architecture", "safetensors",
                metadata, extensions, List.of(tensor));
    }

    private static ConversionPlan plan(
            ModelDescriptor model, CompatibilityReport.ConversionFidelity fidelity) {
        var diagnostics = fidelity == CompatibilityReport.ConversionFidelity.PARTIAL
                ? List.of(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.WARNING, "OPAQUE_METADATA",
                        "Opaque extension requires explicit acknowledgement"))
                : List.<CompatibilityReport.Diagnostic>of();
        return new ConversionPlan(
                model.identity(), model.sourceFormat(), "safetensors",
                new CompatibilityReport(fidelity, diagnostics));
    }

    private Path writeSource(String name) throws Exception {
        String json = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        byte[] rawHeader = json.getBytes(StandardCharsets.UTF_8);
        int padding = (int) ((8 - (Long.BYTES + rawHeader.length) % 8) % 8);
        byte[] header = java.util.Arrays.copyOf(rawHeader, rawHeader.length + padding);
        java.util.Arrays.fill(header, rawHeader.length, header.length, (byte) ' ');
        byte[] payload = new byte[] {1, 2, 3, 4, 5, 6, 7, 8};
        ByteBuffer file = ByteBuffer.allocate(Long.BYTES + header.length + payload.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(header.length).put(header).put(payload);
        Path source = temporaryDirectory.resolve(name);
        Files.write(source, file.array());
        return source;
    }
}
