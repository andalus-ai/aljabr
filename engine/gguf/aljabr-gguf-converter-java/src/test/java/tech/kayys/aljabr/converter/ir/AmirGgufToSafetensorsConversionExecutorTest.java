package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutors;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AmirGgufToSafetensorsConversionExecutorTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void discoversExecutorThroughServiceLoader() {
        assertInstanceOf(AmirGgufToSafetensorsConversionExecutor.class,
                ModelConversionExecutors.find("gguf", "safetensors").orElseThrow());
    }

    @Test
    void copiesScalarF32PayloadAndPreservesSourceMetadata() throws Exception {
        byte[] values = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.25f).putFloat(-2.5f).array();
        Path source = writeContainer(values);
        var tensor = tensor("weight", "F32", List.of(2L), source, 8L, values.length);
        var model = model("tiny", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("scalar.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                ConversionRequest.to(output.toUri()));

        assertEquals(8L, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        int dataStart = dataStart(file);
        assertArrayEquals(values, java.util.Arrays.copyOfRange(file, dataStart, dataStart + values.length));
        JsonObject header = header(file);
        assertEquals("F32", header.getAsJsonObject("weight").get("dtype").getAsString());
        assertEquals("llama", header.getAsJsonObject("__metadata__")
                .get("aljabr.source.architecture").getAsString());
    }

    @Test
    void reversesGgmlFastestFirstDimensionsWithoutReorderingPayload() throws Exception {
        byte[] values = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1).putFloat(2).putFloat(3).putFloat(4).putFloat(5).putFloat(6).array();
        Path source = writeContainer(values);
        var payload = new PayloadReference(source.toUri(), source.getFileName().toString(),
                8L, values.length, PayloadReference.OffsetBasis.CONTAINER);
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L, 3L), "unknown", null,
                payload, Map.of("gguf.dimension-order", "ggml-fastest-first"));
        var model = new ModelDescriptor(ModelIdentity.of("ordered"), "llama", "gguf", Map.of(),
                Map.of("gguf.tensor.dimension-order", "ggml-fastest-first"), List.of(tensor));
        Path output = temporaryDirectory.resolve("ordered.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                ConversionRequest.to(output.toUri()));

        JsonObject tensorHeader = header(Files.readAllBytes(output)).getAsJsonObject("weight");
        assertEquals(List.of(3, 2), tensorHeader.getAsJsonArray("shape").asList().stream()
                .map(value -> value.getAsInt()).toList());
        assertArrayEquals(values, java.util.Arrays.copyOfRange(Files.readAllBytes(output),
                dataStart(Files.readAllBytes(output)), dataStart(Files.readAllBytes(output)) + values.length));
    }

    @Test
    void reversesGgufDimensionsByDefaultWithoutReorderingPayload() throws Exception {
        byte[] values = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1).putFloat(2).putFloat(3).putFloat(4).putFloat(5).putFloat(6).array();
        Path source = writeContainer(values);
        var tensor = tensor("weight", "F32", List.of(2L, 3L), source, 8L, values.length);
        var model = model("default-order", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("default-order.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                ConversionRequest.to(output.toUri()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals(List.of(3, 2), tensorHeader.getAsJsonArray("shape").asList().stream()
                .map(value -> value.getAsInt()).toList());
        assertArrayEquals(values, java.util.Arrays.copyOfRange(
                file, dataStart(file), dataStart(file) + values.length));
    }

    @Test
    void dequantizesQ4ZeroBlocksToF32InGgmlElementOrder() throws Exception {
        byte[] block = new byte[18];
        block[0] = 0;
        block[1] = 0x3c; // fp16 scale = 1
        java.util.Arrays.fill(block, 2, block.length, (byte) 0x98);
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q4_0", List.of(32L), source, 8L, block.length);
        var model = model("q4", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q4.safetensors");
        var plan = plan(model, CompatibilityReport.ConversionFidelity.PARTIAL);

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan, new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(32L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        JsonObject header = header(file);
        assertEquals("F32", header.getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 32 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 16; i++) assertEquals(0.0f, values.getFloat(), 0.0f);
        for (int i = 0; i < 16; i++) assertEquals(1.0f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesQ8ZeroBlocksToF32() throws Exception {
        byte[] block = new byte[34];
        block[0] = 0;
        block[1] = 0x3c; // fp16 scale = 1
        for (int i = 0; i < 32; i++) block[2 + i] = (byte) (i - 16);
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q8_0", List.of(32L), source, 8L, block.length);
        var model = model("q8", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q8.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        ByteBuffer values = ByteBuffer.wrap(Files.readAllBytes(output), dataStart(Files.readAllBytes(output)),
                32 * Float.BYTES).slice().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 32; i++) assertEquals((float) (i - 16), values.getFloat(), 0.0f);
    }

    @Test
    void requiresAcknowledgementForLossyDequantization() throws Exception {
        byte[] block = new byte[18];
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q4_0", List.of(32L), source, 8L, block.length);
        var model = model("q4", "llama", List.of(tensor));
        assertThrows(java.io.IOException.class, () ->
                new AmirGgufToSafetensorsConversionExecutor().execute(
                        model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                        ConversionRequest.to(temporaryDirectory.resolve("no-ack.safetensors").toUri())));
    }

    @Test
    void refusesToTreatLossyDequantizationAsFullFidelity() throws Exception {
        byte[] block = new byte[18];
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q4_0", List.of(32L), source, 8L, block.length);
        var model = model("q4", "llama", List.of(tensor));
        assertThrows(java.io.IOException.class, () ->
                new AmirGgufToSafetensorsConversionExecutor().execute(
                        model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                        new ConversionRequest(temporaryDirectory.resolve("incorrect-full.safetensors").toUri(),
                                true, Map.of())));
    }

    @Test
    void rejectsUnsupportedQuantizationWithoutCreatingOutput() throws Exception {
        byte[] payload = new byte[84];
        Path source = writeContainer(payload);
        var tensor = tensor("weight", "Q2_K", List.of(256L), source, 8L, payload.length);
        var model = model("q2", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("unsupported.safetensors");
        assertThrows(java.io.IOException.class, () ->
                new AmirGgufToSafetensorsConversionExecutor().execute(
                        model, plan(model, CompatibilityReport.ConversionFidelity.UNSUPPORTED),
                        new ConversionRequest(output.toUri(), true, Map.of())));
        assertFalse(Files.exists(output));
    }

    private Path writeContainer(byte[] payload) throws Exception {
        byte[] prefix = new byte[8];
        Path source = temporaryDirectory.resolve("model.gguf");
        byte[] file = new byte[prefix.length + payload.length];
        System.arraycopy(prefix, 0, file, 0, prefix.length);
        System.arraycopy(payload, 0, file, prefix.length, payload.length);
        Files.write(source, file);
        return source;
    }

    private static TensorDescriptor tensor(
            String name, String dtype, List<Long> shape, Path source, long offset, long length) {
        var payload = new PayloadReference(source.toUri(), source.getFileName().toString(),
                offset, length, PayloadReference.OffsetBasis.CONTAINER);
        return new TensorDescriptor(name, dtype, shape, "unknown", null, payload);
    }

    private static ModelDescriptor model(String id, String architecture, List<TensorDescriptor> tensors) {
        return new ModelDescriptor(ModelIdentity.of(id), architecture, "gguf",
                Map.of("general.name", id), Map.of("vendor.source", "unit-test"), tensors);
    }

    private static ConversionPlan plan(
            ModelDescriptor model, CompatibilityReport.ConversionFidelity fidelity) {
        return new ConversionPlan(model.identity(), "gguf", "safetensors",
                new CompatibilityReport(fidelity, List.of()));
    }

    private static JsonObject header(byte[] file) {
        ByteBuffer prefix = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        long length = prefix.getLong();
        return JsonParser.parseString(new String(file, Long.BYTES, Math.toIntExact(length),
                java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject();
    }

    private static int dataStart(byte[] file) {
        ByteBuffer prefix = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        return Math.toIntExact(Long.BYTES + prefix.getLong());
    }
}
