package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.MetadataValue;
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
    void dequantizesNVFP4UsingFourUE4M3SubBlockScales() throws Exception {
        byte[] block = new byte[68];
        block[0] = 0x40; // UE4M3 2.0; GGML's half-scale convention gives 1.0
        block[1] = 0x38; // UE4M3 1.0; GGML's half-scale convention gives 0.5
        block[2] = 0x00; // zero scale
        block[3] = 0x7f; // reserved NaN encoding maps to zero in GGML
        for (int subBlock = 0; subBlock < 4; subBlock++) {
            for (int lane = 0; lane < 8; lane++) {
                int value = lane + 1;
                block[4 + subBlock * 8 + lane] = (byte) (value | (value << 4));
            }
        }
        Path source = writeContainer(block);
        var tensor = tensor("weight", "NVFP4", List.of(64L), source, 8L, block.length);
        var model = model("nvfp4", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("nvfp4.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(64L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 64 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        float[] codebook = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        for (int subBlock = 0; subBlock < 4; subBlock++) {
            float scale = switch (subBlock) {
                case 0 -> 1.0f;
                case 1 -> 0.5f;
                default -> 0.0f;
            };
            for (int lane = 0; lane < 8; lane++) {
                assertEquals(codebook[lane + 1] * scale,
                        values.getFloat((subBlock * 16 + lane) * Float.BYTES), 0.0f);
                assertEquals(codebook[lane + 1] * scale,
                        values.getFloat((subBlock * 16 + lane + 8) * Float.BYTES), 0.0f);
            }
        }
    }

    @Test
    void dequantizesTQ1BlocksInReferenceBaseThreePlaneOrder() throws Exception {
        byte[] block = new byte[54];
        block[0] = 0x55; // packed base-3 digits: -1, +1, +1, +1, +1
        block[53] = 0x3c; // fp16 scale = 1
        Path source = writeContainer(block);
        var tensor = tensor("weight", "TQ1_0", List.of(256L), source, 8L, block.length);
        var model = model("tq1_0", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("tq1_0.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(256L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(-1.0f, values.getFloat(0), 0.0f);
        assertEquals(1.0f, values.getFloat(32 * Float.BYTES), 0.0f);
        assertEquals(1.0f, values.getFloat(64 * Float.BYTES), 0.0f);
        assertEquals(1.0f, values.getFloat(96 * Float.BYTES), 0.0f);
        assertEquals(1.0f, values.getFloat(128 * Float.BYTES), 0.0f);
        for (int i = 1; i < 256; i++) {
            if (i == 32 || i == 64 || i == 96 || i == 128) continue;
            assertEquals(-1.0f, values.getFloat(i * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesTQ2BlocksInReferenceBitPlaneOrder() throws Exception {
        byte[] block = new byte[66];
        block[1] = 0x3c; // fp16 scale = 1
        block[2] = 0x03; // first lane in plane 0: 2; next lane: -1
        block[3] = 0x01; // second lane in plane 0: 0
        Path source = writeContainer(block);
        var tensor = tensor("weight", "TQ2_0", List.of(256L), source, 8L, block.length);
        var model = model("tq2_0", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("tq2_0.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(256L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(2.0f, values.getFloat(), 0.0f);
        assertEquals(0.0f, values.getFloat(), 0.0f);
        for (int i = 2; i < 256; i++) assertEquals(-1.0f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesMXFP4UsingE8M0ScaleAndBothE2M1Nibbles() throws Exception {
        byte[] block = new byte[34];
        block[0] = (byte) 128; // E8M0 / 2 = 1.0
        block[17] = (byte) 127; // E8M0 / 2 = 0.5
        for (int i = 0; i < 16; i++) {
            block[i + 1] = (byte) (i | (i << 4));
            block[i + 18] = (byte) (i | (i << 4));
        }
        Path source = writeContainer(block);
        var tensor = tensor("weight", "MXFP4", List.of(64L), source, 8L, block.length);
        var model = model("mxfp4", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("mxfp4.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(64L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 64 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        float[] codebook = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        for (int i = 0; i < 16; i++) {
            assertEquals(codebook[i], values.getFloat(i * Float.BYTES), 0.0f);
            assertEquals(codebook[i], values.getFloat((i + 16) * Float.BYTES), 0.0f);
            assertEquals(codebook[i] * 0.5f, values.getFloat((i + 32) * Float.BYTES), 0.0f);
            assertEquals(codebook[i] * 0.5f, values.getFloat((i + 48) * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesIQ1MUsingSharedGridAndPackedMixedScales() throws Exception {
        byte[] block = new byte[56];
        block[48] = 0x08; // second scale of first sub-block is 3x
        block[52] = (byte) 0xc0; // packed fp16 scale high nibble -> 0xC
        block[54] = 0x30; // packed fp16 scale high nibble -> 0x3, giving fp16 1
        block[32] = 0x09; // vector 0 grid index 256 and negative delta
        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ1_M", List.of(256L), source, 8L, block.length);
        var model = model("iq1_m", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq1_m.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(256L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        float[] firstSubBlock = {
                -1.125f, -0.125f, 0.875f, -0.125f, 0.875f, -1.125f, -0.125f, -1.125f,
                -0.875f, -0.875f, -0.875f, -0.875f, -0.875f, -0.875f, -0.875f, -0.875f,
                -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f,
                -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f, -2.625f
        };
        for (float expected : firstSubBlock) assertEquals(expected, values.getFloat(), 0.0f);
        for (int i = 32; i < 256; i++) assertEquals(-0.875f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesIQ1SBlocksUsingCanonicalGridAndPackedScaleAndSign() throws Exception {
        byte[] block = new byte[50];
        block[1] = 0x3c; // fp16 base scale = 1; zero qh selects scale 1 and positive delta
        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ1_S", List.of(256L), source, 8L, block.length);
        var model = model("iq1_s", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq1_s.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(256L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 256; i++) assertEquals(-0.875f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesIQ1SHighGridBitsScaleMultiplierAndNegativeDelta() throws Exception {
        byte[] block = new byte[50];
        block[1] = 0x3c; // fp16 base scale = 1
        block[34] = 0x01; // first grid index high bits = 1
        block[35] = (byte) 0x90; // scale multiplier = 3; negative delta
        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ1_S", List.of(256L), source, 8L, block.length);
        var model = model("iq1_s-packed", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq1_s-packed.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        float[] firstGrid = {-3.375f, -0.375f, 2.625f, -0.375f,
                2.625f, -3.375f, -0.375f, -3.375f};
        for (float expected : firstGrid) assertEquals(expected, values.getFloat(), 0.0f);
        for (int i = 8; i < 32; i++) assertEquals(-3.375f, values.getFloat(), 0.0f);
        for (int i = 32; i < 256; i++) assertEquals(-0.875f, values.getFloat(), 0.0f);
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
    void dequantizesQ2ZeroBlocksUsingTwoBitCodesInGgmlElementOrder() throws Exception {
        byte[] block = new byte[18];
        block[1] = 0x40; // fp16 scale = 2
        java.util.Arrays.fill(block, 2, block.length, (byte) 0xe4);
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q2_0", List.of(64L), source, 8L, block.length);
        var model = model("q2_0", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q2_0.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        assertEquals(64L * Float.BYTES, result.payloadBytes());
        byte[] file = Files.readAllBytes(output);
        assertEquals("F32", header(file).getAsJsonObject("weight").get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 64 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        float[] expected = {-2.0f, 0.0f, 2.0f, 4.0f};
        for (int i = 0; i < 64; i++) {
            assertEquals(expected[i & 3], values.getFloat(), 0.0f);
        }
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
    void dequantizesQ4OneBlocksUsingScaleAndMinimum() throws Exception {
        byte[] block = new byte[20];
        block[1] = 0x3c;
        block[3] = 0x40;
        java.util.Arrays.fill(block, 4, block.length, (byte) 0x98);
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q4_1", List.of(32L), source, 8L, block.length);
        var model = model("q4_1", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q4_1.safetensors");
        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));
        byte[] file = Files.readAllBytes(output);
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 32 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 16; i++) assertEquals(10.0f, values.getFloat(), 0.0f);
        for (int i = 0; i < 16; i++) assertEquals(11.0f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesQ5ZeroBlocksIncludingHighBits() throws Exception {
        byte[] block = new byte[22];
        block[1] = 0x3c;
        block[2] = 0x01;
        block[4] = 0x01;
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q5_0", List.of(32L), source, 8L, block.length);
        var model = model("q5_0", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q5_0.safetensors");
        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));
        byte[] file = Files.readAllBytes(output);
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 32 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0.0f, values.getFloat(), 0.0f);
        for (int i = 1; i < 16; i++) assertEquals(-16.0f, values.getFloat(), 0.0f);
        assertEquals(0.0f, values.getFloat(), 0.0f);
        for (int i = 17; i < 32; i++) assertEquals(-16.0f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesQ5OneBlocksUsingScaleMinimumAndHighBits() throws Exception {
        byte[] block = new byte[24];
        block[1] = 0x3c;
        block[3] = 0x40;
        block[4] = 0x01;
        block[6] = 0x01;
        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q5_1", List.of(32L), source, 8L, block.length);
        var model = model("q5_1", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q5_1.safetensors");
        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));
        byte[] file = Files.readAllBytes(output);
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 32 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(18.0f, values.getFloat(), 0.0f);
        for (int i = 1; i < 16; i++) assertEquals(2.0f, values.getFloat(), 0.0f);
        assertEquals(18.0f, values.getFloat(), 0.0f);
        for (int i = 17; i < 32; i++) assertEquals(2.0f, values.getFloat(), 0.0f);
    }

    @Test
    void dequantizesQ2KBlockUsingFourBitPlanesAndPerGroupScaleAndMinimum() throws Exception {
        byte[] block = new byte[84];
        for (int i = 0; i < 16; i++) block[i] = (byte) ((i + 1) * 0x11);
        java.util.Arrays.fill(block, 16, 80, (byte) 0xe4); // packed 2-bit values: 0, 1, 2, 3
        block[81] = 0x3c; // fp16 d = 1
        block[83] = 0x40; // fp16 dmin = 2

        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q2_K", List.of(256L), source, 8L, block.length);
        var model = model("q2_k", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q2_k.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(-2.0f, values.getFloat(0 * Float.BYTES), 0.0f);
        assertEquals(-4.0f, values.getFloat(16 * Float.BYTES), 0.0f);
        assertEquals(-3.0f, values.getFloat(32 * Float.BYTES), 0.0f);
        assertEquals(-4.0f, values.getFloat(48 * Float.BYTES), 0.0f);
        assertEquals(0.0f, values.getFloat(64 * Float.BYTES), 0.0f);
        assertEquals(0.0f, values.getFloat(80 * Float.BYTES), 0.0f);
        assertEquals(7.0f, values.getFloat(96 * Float.BYTES), 0.0f);
        assertEquals(8.0f, values.getFloat(112 * Float.BYTES), 0.0f);
        assertEquals(-18.0f, values.getFloat(128 * Float.BYTES), 0.0f);
        assertEquals(-20.0f, values.getFloat(144 * Float.BYTES), 0.0f);
        assertEquals(-11.0f, values.getFloat(160 * Float.BYTES), 0.0f);
        assertEquals(-12.0f, values.getFloat(176 * Float.BYTES), 0.0f);
        assertEquals(0.0f, values.getFloat(192 * Float.BYTES), 0.0f);
        assertEquals(0.0f, values.getFloat(208 * Float.BYTES), 0.0f);
        assertEquals(15.0f, values.getFloat(224 * Float.BYTES), 0.0f);
        assertEquals(16.0f, values.getFloat(240 * Float.BYTES), 0.0f);
    }

    @Test
    void dequantizesQ3KBlockUsingPackedSixBitScalesLowBitsAndHighMask() throws Exception {
        byte[] block = new byte[110];
        java.util.Arrays.fill(block, 32, 96, (byte) 0xe4); // low two-bit planes: 0, 1, 2, 3
        for (int i = 0; i < 8; i++) {
            int lowScale = 33 + i;
            int highScale = 41 + i;
            block[96 + i] = (byte) ((lowScale & 0x0f) | ((highScale & 0x0f) << 4));
        }
        java.util.Arrays.fill(block, 104, 108, (byte) 0xaa); // upper two bits encode 2 for every scale
        java.util.Arrays.fill(block, 0, 32, (byte) 0xff); // high bit set => no -4 offset
        block[0] = (byte) 0xfe; // first quantized value uses the negative branch
        block[109] = 0x3c; // fp16 d = 1

        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q3_K", List.of(256L), source, 8L, block.length);
        var model = model("q3_k", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q3_k.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(-4.0f, values.getFloat(0 * Float.BYTES), 0.0f);
        assertEquals(1.0f, values.getFloat(1 * Float.BYTES), 0.0f);
        assertEquals(2.0f, values.getFloat(2 * Float.BYTES), 0.0f);
        assertEquals(3.0f, values.getFloat(3 * Float.BYTES), 0.0f);
        assertEquals(2.0f, values.getFloat(17 * Float.BYTES), 0.0f);
        assertEquals(3.0f, values.getFloat(33 * Float.BYTES), 0.0f);
        assertEquals(4.0f, values.getFloat(49 * Float.BYTES), 0.0f);
        assertEquals(5.0f, values.getFloat(65 * Float.BYTES), 0.0f);
        assertEquals(8.0f, values.getFloat(113 * Float.BYTES), 0.0f);
        assertEquals(9.0f, values.getFloat(129 * Float.BYTES), 0.0f);
        assertEquals(12.0f, values.getFloat(177 * Float.BYTES), 0.0f);
        assertEquals(16.0f, values.getFloat(241 * Float.BYTES), 0.0f);
    }

    @Test
    void dequantizesIQ2XXSBlockUsingPackedGridSignsAndScale() throws Exception {
        byte[] block = new byte[66];
        block[1] = 0x3c; // fp16 scale = 1
        block[2] = 0; // grid 0: eight values of 8
        block[3] = 1; // grid 1: [43, 8, 8, 8, 8, 8, 8, 8]
        block[6] = 1; // first seven-bit sign code expands to 0x81
        block[9] = 0x10; // group scale nibble = 1 => d * 0.375

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ2_XXS", List.of(256L), source, 8L, block.length);
        var model = model("iq2_xxs", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq2_xxs.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] expected = {-3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, -3.0f,
                16.125f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values.getFloat(i * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesIQ2SBlockUsingGridHighBitsSignsAndIndependentScaleNibbles() throws Exception {
        byte[] block = new byte[82];
        block[1] = 0x3c; // fp16 d = 1
        // The low eight-bit grid indices are zero; qh selects 256/512/768/0.
        block[66] = 0x39;
        // Next 32-value group selects 0/256/512/768.
        block[67] = (byte) 0xe4;
        block[34] = (byte) 0x81; // first and eighth lanes negative
        block[35] = (byte) 0x82; // second and eighth lanes negative
        block[74] = 0x31; // first pair scale = 0.375, second pair = 0.875
        block[75] = 0x20; // next group scale = 0.125 / 0.625

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ2_S", List.of(256L), source, 8L, block.length);
        var model = model("iq2_s", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq2_s.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        int[][] grids = {
                {25, 25, 25, 25, 25, 8, 25, 8},       // grid 256
                {25, 25, 43, 8, 25, 8, 8, 25},        // grid 512
                {8, 43, 8, 25, 8, 8, 43, 25},         // grid 768
                {8, 8, 8, 8, 8, 8, 8, 8}              // grid 0
        };
        int[] signMasks = {0x81, 0x82, 0, 0};
        float[] scales = {0.375f, 0.375f, 0.875f, 0.875f};
        for (int vector = 0; vector < 4; vector++) {
            for (int lane = 0; lane < 8; lane++) {
                float sign = (signMasks[vector] & (1 << lane)) != 0 ? -1.0f : 1.0f;
                assertEquals(scales[vector] * grids[vector][lane] * sign,
                        values.getFloat((vector * 8 + lane) * Float.BYTES), 0.0f);
            }
        }

        // The next 32 values exercise the second qh byte and the next scale byte.
        int[][] secondGrids = {grids[3], grids[0], grids[1], grids[2]};
        float[] secondScales = {0.125f, 0.125f, 0.625f, 0.625f};
        for (int vector = 0; vector < 4; vector++) {
            for (int lane = 0; lane < 8; lane++) {
                assertEquals(secondScales[vector] * secondGrids[vector][lane],
                        values.getFloat((32 + vector * 8 + lane) * Float.BYTES), 0.0f);
            }
        }
    }

    @Test
    void dequantizesIQ3XXSBlockUsingPackedGridSignsAndScale() throws Exception {
        byte[] block = new byte[98];
        block[1] = 0x3c; // fp16 scale = 1
        block[2] = 1; // grid 1 = [4, 4, 4, 20]
        block[3] = 0; // grid 0 = [4, 4, 4, 4]
        // aux32: sign codes in seven-bit fields; high nibble sets scale = 0.5.
        // A single low sign bit yields parity-expanded mask 0x81.
        block[66] = 1;
        block[69] = 0x10; // group scale nibble = 1 => d * 0.75

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ3_XXS", List.of(256L), source, 8L, block.length);
        var model = model("iq3_xxs", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq3_xxs.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] expected = {-15.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, -3.0f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values.getFloat(i * Float.BYTES), 0.0f);
        }
        // The next 32-value group retains the default 0.5 multiplier.
        assertEquals(1.0f, values.getFloat(32 * Float.BYTES), 0.0f);
    }

    @Test
    void dequantizesIQ3SBlockUsingGridHighBitsPackedSignsAndGroupScales() throws Exception {
        byte[] block = new byte[110];
        block[1] = 0x3c; // fp16 scale = 1
        block[2] = 1; // grid index 1: [3, 1, 1, 1]
        block[3] = 0; // grid index 0: [1, 1, 1, 1]
        block[67] = 1; // high bit selects grid index 256 for first grid in second 32-value half
        block[74] = 1; // parity-expanded sign mask 0x81: first and eighth values negative
        block[106] = 0x10; // first group scale = 1, second group scale = 3

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ3_S", List.of(256L), source, 8L, block.length);
        var model = model("iq3_s", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq3_s.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        // The first eight values exercise the first grid, sign parity, and ordering.
        float[] expected = {-3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values.getFloat(i * Float.BYTES), 0.0f);
        }
        // The next 32-value group uses the second packed scale nibble (3x).
        assertEquals(21.0f, values.getFloat(32 * Float.BYTES), 0.0f); // grid[256][0] = 7, scaled by 3
    }

    @Test
    void dequantizesIQ4NLBlockUsingCanonicalNonlinearLookupTable() throws Exception {
        byte[] block = new byte[18];
        block[1] = 0x3c; // fp16 d = 1
        for (int i = 0; i < 16; i++) {
            block[2 + i] = (byte) (((i + 8) & 0x0f) << 4 | i);
        }

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ4_NL", List.of(32L), source, 8L, block.length);
        var model = model("iq4_nl", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq4_nl.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 32 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] codebook = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};
        for (int i = 0; i < 16; i++) {
            assertEquals(codebook[i], values.getFloat(i * Float.BYTES), 0.0f);
            assertEquals(codebook[(i + 8) & 0x0f],
                    values.getFloat((i + 16) * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesIQ4XSBlockUsingPackedSixBitGroupScalesAndNonlinearCodebook() throws Exception {
        byte[] block = new byte[136];
        block[1] = 0x3c; // fp16 super-block scale = 1
        block[2] = (byte) 0xaa; // upper two scale bits: 2 for all eight groups
        block[3] = (byte) 0xaa;

        for (int group = 0; group < 8; group++) {
            int lowScale = group + 1;
            int scaleOffset = 4 + group / 2;
            block[scaleOffset] |= (byte) (lowScale << (4 * (group & 1)));
            for (int i = 0; i < 16; i++) {
                block[8 + group * 16 + i] = (byte) ((15 - i) << 4 | i);
            }
        }

        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ4_XS", List.of(256L), source, 8L, block.length);
        var model = model("iq4_xs", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq4_xs.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] codebook = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};
        for (int group = 0; group < 8; group++) {
            float groupScale = group + 1.0f;
            int outputOffset = group * 32;
            for (int i = 0; i < 16; i++) {
                assertEquals(groupScale * codebook[i],
                        values.getFloat((outputOffset + i) * Float.BYTES), 0.0f);
                assertEquals(groupScale * codebook[15 - i],
                        values.getFloat((outputOffset + 16 + i) * Float.BYTES), 0.0f);
            }
        }
    }

    @Test
    void dequantizesQ4KBlockUsingPackedSixBitScalesAndMinimaAndNibbleGroups() throws Exception {
        byte[] block = new byte[144];
        block[1] = 0x3c; // fp16 d = 1
        block[3] = 0x40; // fp16 dmin = 2

        int[] scales = {17, 18, 19, 20, 21, 22, 23, 24};
        int[] mins = {25, 26, 27, 28, 29, 30, 31, 32};
        for (int i = 0; i < 4; i++) {
            block[4 + i] = (byte) ((scales[i] & 0x3f) | ((scales[i + 4] >>> 4) << 6));
            block[8 + i] = (byte) ((mins[i] & 0x3f) | ((mins[i + 4] >>> 4) << 6));
            block[12 + i] = (byte) ((scales[i + 4] & 0x0f) | ((mins[i + 4] & 0x0f) << 4));
        }
        java.util.Arrays.fill(block, 16, 144, (byte) 0x21); // low nibble=1, high nibble=2

        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q4_K", List.of(256L), source, 8L, block.length);
        var model = model("q4_k", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q4_k.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] expectedGroupValues = {-33.0f, -16.0f, -35.0f, -16.0f,
                -37.0f, -16.0f, -39.0f, -16.0f};
        for (int group = 0; group < expectedGroupValues.length; group++) {
            assertEquals(expectedGroupValues[group], values.getFloat(group * 32 * Float.BYTES), 0.0f);
            assertEquals(expectedGroupValues[group],
                    values.getFloat((group * 32 + 31) * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesQ5KBlockUsingSharedHighBitPlaneAndPackedScaleMinimumGroups() throws Exception {
        byte[] block = new byte[176];
        block[1] = 0x3c; // fp16 d = 1
        block[3] = 0x40; // fp16 dmin = 2

        int[] scales = {17, 18, 19, 20, 21, 22, 23, 24};
        int[] mins = {25, 26, 27, 28, 29, 30, 31, 32};
        for (int i = 0; i < 4; i++) {
            block[4 + i] = (byte) ((scales[i] & 0x3f) | ((scales[i + 4] >>> 4) << 6));
            block[8 + i] = (byte) ((mins[i] & 0x3f) | ((mins[i + 4] >>> 4) << 6));
            block[12 + i] = (byte) ((scales[i + 4] & 0x0f) | ((mins[i + 4] & 0x0f) << 4));
        }
        block[16] = (byte) 0xa5; // segment masks: low/high = 1/0, 1/0, 0/1, 0/1
        java.util.Arrays.fill(block, 48, 176, (byte) 0x21); // low nibble=1, high nibble=2

        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q5_K", List.of(256L), source, 8L, block.length);
        var model = model("q5_k", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q5_k.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        float[] expectedFirstValues = {239.0f, -16.0f, 269.0f, -16.0f,
                -37.0f, 336.0f, -39.0f, 368.0f};
        for (int group = 0; group < expectedFirstValues.length; group++) {
            assertEquals(expectedFirstValues[group],
                    values.getFloat(group * 32 * Float.BYTES), 0.0f);
        }
    }

    @Test
    void dequantizesQ6KBlockUsingLowBitsHighBitsAndSubgroupScales() throws Exception {
        byte[] block = new byte[210];
        java.util.Arrays.fill(block, 192, 208, (byte) 1);
        for (int scale : new int[] {193, 195, 197, 199}) block[scale] = 2;
        block[209] = 0x3c; // fp16 super-scale = 1

        block[0] = 0x21;
        block[32] = 0x43;
        block[128] = (byte) 0b11_10_01_00;
        block[64] = 0x65;
        block[96] = (byte) 0x87;
        block[160] = (byte) 0b00_01_10_11;

        Path source = writeContainer(block);
        var tensor = tensor("weight", "Q6_K", List.of(256L), source, 8L, block.length);
        var model = model("q6_k", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("q6_k.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(-31.0f, values.getFloat(0 * Float.BYTES), 0.0f);
        assertEquals(-64.0f, values.getFloat(16 * Float.BYTES), 0.0f);
        assertEquals(-13.0f, values.getFloat(32 * Float.BYTES), 0.0f);
        assertEquals(-64.0f, values.getFloat(48 * Float.BYTES), 0.0f);
        assertEquals(2.0f, values.getFloat(64 * Float.BYTES), 0.0f);
        assertEquals(-64.0f, values.getFloat(80 * Float.BYTES), 0.0f);
        assertEquals(20.0f, values.getFloat(96 * Float.BYTES), 0.0f);
        assertEquals(-64.0f, values.getFloat(112 * Float.BYTES), 0.0f);
        assertEquals(21.0f, values.getFloat(128 * Float.BYTES), 0.0f);
        assertEquals(7.0f, values.getFloat(160 * Float.BYTES), 0.0f);
        assertEquals(-10.0f, values.getFloat(192 * Float.BYTES), 0.0f);
        assertEquals(-24.0f, values.getFloat(224 * Float.BYTES), 0.0f);
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
    void dequantizesIq2XsUsingCanonicalGridAndSubgroupScale() throws Exception {
        byte[] block = new byte[74];
        block[1] = 0x3c; // fp16 scale = 1
        block[4] = 1; // second vector uses grid entry 1 (first lane = 43)
        // All packed scale nibbles are zero: effective scale = 0.125.
        Path source = writeContainer(block);
        var tensor = tensor("weight", "IQ2_XS", List.of(256L), source, 8L, block.length);
        var model = model("iq2_xs", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("iq2_xs.safetensors");

        new AmirGgufToSafetensorsConversionExecutor().execute(model,
                plan(model, CompatibilityReport.ConversionFidelity.PARTIAL),
                new ConversionRequest(output.toUri(), true, Map.of()));

        byte[] file = Files.readAllBytes(output);
        JsonObject tensorHeader = header(file).getAsJsonObject("weight");
        assertEquals("F32", tensorHeader.get("dtype").getAsString());
        ByteBuffer values = ByteBuffer.wrap(file, dataStart(file), 256 * Float.BYTES)
                .slice().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1.0f, values.getFloat(0), 0.0001f);
        assertEquals(5.375f, values.getFloat(Float.BYTES), 0.0001f);
        for (int i = 8; i < 256; i++) {
            assertEquals(1.0f, values.getFloat(i * Float.BYTES), 0.0001f);
        }
    }

    @Test
    void rejectsUnsupportedQuantizationWithoutCreatingOutput() throws Exception {
        byte[] payload = new byte[66];
        Path source = writeContainer(payload);
        var tensor = tensor("weight", "IQ2_XXS", List.of(256L), source, 8L, payload.length);
        var model = model("unsupported_quantization", "llama", List.of(tensor));
        Path output = temporaryDirectory.resolve("unsupported.safetensors");
        assertThrows(java.io.IOException.class, () ->
                new AmirGgufToSafetensorsConversionExecutor().execute(
                        model, plan(model, CompatibilityReport.ConversionFidelity.UNSUPPORTED),
                        new ConversionRequest(output.toUri(), true, Map.of())));
        assertFalse(Files.exists(output));
    }


    @Test
    void roundTripsTypedGgufMetadataThroughSafetensorsEnvelope() throws Exception {
        byte[] values = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.0f).array();
        Path source = writeContainer(values);
        var tensor = tensor("weight", "F32", List.of(1L), source, 8L, values.length);
        var model = new ModelDescriptor(ModelIdentity.of("typed-metadata"), "llama", "gguf",
                Map.of("general.architecture", "llama"), Map.of(), List.of(tensor))
                .withTypedMetadata(Map.of("general.alignment",
                        new MetadataValue.Scalar(MetadataValue.Kind.UINT32, 32L)));
        Map<String, MetadataValue> typed = Map.of(
                "general.alignment", new MetadataValue.Scalar(MetadataValue.Kind.UINT32, 32L),
                "general.quantization_version", new MetadataValue.Scalar(MetadataValue.Kind.UINT64, -1L),
                "general.use_mmap", new MetadataValue.Scalar(MetadataValue.Kind.BOOL, true),
                "general.scale", new MetadataValue.Scalar(MetadataValue.Kind.FLOAT64, Double.NaN),
                "tokenizer.ggml.tokens", new MetadataValue.Array(MetadataValue.Kind.STRING, List.of(
                        MetadataValue.string("alpha"), MetadataValue.string("beta"))));
        model = model.withTypedMetadata(typed);
        Path output = temporaryDirectory.resolve("typed-metadata.safetensors");

        var result = new AmirGgufToSafetensorsConversionExecutor().execute(
                model, plan(model, CompatibilityReport.ConversionFidelity.FULL),
                new ConversionRequest(output.toUri(), false, Map.of()));

        assertTrue(result.untransferredMetadataKeys().isEmpty());
        ModelDescriptor restored = new AmirModelDescriptorReader().read(output);
        assertEquals(typed, restored.typedMetadata());
        assertEquals("llama", restored.architecture());
        assertEquals("llama", restored.metadata().get("aljabr.source.metadata.general.architecture"));

        Path ggufOutput = temporaryDirectory.resolve("typed-metadata-roundtrip.gguf");
        var ggufPlan = new ConversionPlan(restored.identity(), "safetensors", "gguf",
                new CompatibilityReport(CompatibilityReport.ConversionFidelity.FULL, List.of()));
        new AmirGgufConversionExecutor().execute(restored, ggufPlan,
                ConversionRequest.to(ggufOutput.toUri()));
        try (var ggufReader = new tech.kayys.aljabr.gguf.loader.GGUFReader(ggufOutput)) {
            var parsed = new tech.kayys.aljabr.gguf.loader.GGUFParser()
                    .parse(ggufReader.segment(), ggufReader.arena());
            assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt32Val.class,
                    parsed.typedMetadata().get("general.alignment"));
            assertEquals(-1L, assertInstanceOf(
                    tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt64Val.class,
                    parsed.typedMetadata().get("general.quantization_version")).value());
            assertInstanceOf(tech.kayys.aljabr.gguf.core.GgufMetaValue.ArrayVal.class,
                    parsed.typedMetadata().get("tokenizer.ggml.tokens"));
        }
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
