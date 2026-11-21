package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.core.GgufMetaValue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class StreamingGgufWriterTest {
    @TempDir Path temporaryDirectory;

    @Test
    void writesHeaderAndStreamsPayloadAtAlignedDataOffset() throws Exception {
        Path source = temporaryDirectory.resolve("weights.bin");
        byte[] payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.25f).putFloat(-2.5f).array();
        Files.write(source, payload);
        Path output = temporaryDirectory.resolve("model.gguf");

        var result = StreamingGgufWriter.write(output,
                Map.of("general.architecture", GgufMetaValue.ofString("test")),
                List.of(new StreamingGgufWriter.TensorPayload(
                        "weight", new long[]{2}, GgmlType.F32, source, 0, payload.length)));

        byte[] written = Files.readAllBytes(output);
        ByteBuffer header = ByteBuffer.wrap(written).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x46554747, header.getInt());
        assertEquals(3, header.getInt());
        assertEquals(1L, header.getLong());
        assertEquals(2L, header.getLong()); // caller metadata + generated alignment
        assertEquals(32L, result.dataStart() % 32L);
        assertEquals(0L, result.tensorOffsets().get("weight"));
        assertEquals(8L, result.payloadBytes());
        assertArrayEquals(payload, java.util.Arrays.copyOfRange(
                written, Math.toIntExact(result.dataStart()), Math.toIntExact(result.dataStart() + payload.length)));
    }

    @Test
    void rejectsPayloadLengthThatDoesNotMatchTensorTypeAndShape() throws Exception {
        Path source = temporaryDirectory.resolve("short.bin");
        Files.write(source, new byte[4]);
        assertThrows(java.io.IOException.class, () -> StreamingGgufWriter.write(
                temporaryDirectory.resolve("invalid.gguf"), Map.of(),
                List.of(new StreamingGgufWriter.TensorPayload(
                        "weight", new long[]{2}, GgmlType.F32, source, 0, 4))));
        assertFalse(Files.exists(temporaryDirectory.resolve("invalid.gguf")));
    }

    @Test
    void leavesExistingDestinationUntouchedWhenSourceRangeIsInvalid() throws Exception {
        Path source = temporaryDirectory.resolve("short-range.bin");
        Files.write(source, new byte[4]);
        Path output = temporaryDirectory.resolve("existing.gguf");
        byte[] original = {9, 8, 7};
        Files.write(output, original);

        assertThrows(java.io.IOException.class, () -> StreamingGgufWriter.write(
                output, Map.of(), List.of(new StreamingGgufWriter.TensorPayload(
                        "weight", new long[]{1}, GgmlType.F32, source, 2, 4))));
        assertArrayEquals(original, Files.readAllBytes(output));
    }
}
