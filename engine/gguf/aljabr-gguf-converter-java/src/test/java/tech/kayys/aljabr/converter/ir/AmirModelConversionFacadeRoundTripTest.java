package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionFacade;
import tech.kayys.aljabr.spi.model.ir.ModelConversionPlanner;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

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
        var reverseResult = facade.execute(reverse, ConversionRequest.to(restored.toUri()));
        assertEquals(1, reverseResult.tensorCount());
        assertEquals(expectedPayload.length, reverseResult.payloadBytes());

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

    private Path writeSafetensors(String name, byte[] payload) throws Exception {
        String tensorName = java.util.Base64.getUrlEncoder().withoutPadding()
                .encodeToString("weight".getBytes(StandardCharsets.UTF_8));
        String extensionName = java.util.Base64.getUrlEncoder().withoutPadding()
                .encodeToString("vendor.key/雪".getBytes(StandardCharsets.UTF_8));
        String json = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,"
                + payload.length + "]},\"__metadata__\":{"
                + "\"general.architecture\":\"llama\","
                + "\"aljabr.source.metadata.vendor.metadata/key\":\"original-value\","
                + "\"aljabr.source.tensor." + tensorName + "." + extensionName
                + "\":\"extension-value\"}}";
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
