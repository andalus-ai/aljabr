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
