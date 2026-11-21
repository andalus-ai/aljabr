package tech.kayys.aljabr.safetensor.core;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter.TensorSpec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StreamingSafetensorWriterTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void writesHeaderThenStreamsPayloadsInDeclaredOrder() throws IOException {
        Path output = temporaryDirectory.resolve("streamed.safetensors");
        byte[] first = {1, 2, 3, 4};
        byte[] second = {5, 6};

        StreamingSafetensorWriter.write(output, List.of(
                new TensorSpec("first", "I8", List.of(4L), first.length,
                        channel -> writeBytes(channel, first)),
                new TensorSpec("second", "I8", List.of(2L), second.length,
                        channel -> writeBytes(channel, second))));

        byte[] file = Files.readAllBytes(output);
        long headerLength = ByteBuffer.wrap(file, 0, Long.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN).getLong();
        int dataStart = Math.toIntExact(Long.BYTES + headerLength);
        String headerJson = new String(file, Long.BYTES, Math.toIntExact(headerLength),
                java.nio.charset.StandardCharsets.UTF_8).trim();
        JsonObject header = JsonParser.parseString(headerJson).getAsJsonObject();

        assertEquals(0L, header.getAsJsonObject("first").getAsJsonArray("data_offsets").get(0).getAsLong());
        assertEquals(4L, header.getAsJsonObject("first").getAsJsonArray("data_offsets").get(1).getAsLong());
        assertEquals(4L, header.getAsJsonObject("second").getAsJsonArray("data_offsets").get(0).getAsLong());
        assertEquals(6L, header.getAsJsonObject("second").getAsJsonArray("data_offsets").get(1).getAsLong());
        assertArrayEquals(new byte[]{1, 2, 3, 4, 5, 6},
                java.util.Arrays.copyOfRange(file, dataStart, file.length));
    }

    @Test
    void rejectsPayloadSourceThatWritesTooFewBytes() {
        Path output = temporaryDirectory.resolve("short.safetensors");
        TensorSpec shortTensor = new TensorSpec("short", "I8", List.of(3L), 3L,
                channel -> writeBytes(channel, new byte[]{1, 2}));

        assertThrows(IOException.class,
                () -> StreamingSafetensorWriter.write(output, List.of(shortTensor)));
    }

    @Test
    void rejectsPayloadSourceThatWritesTooManyBytes() {
        Path output = temporaryDirectory.resolve("long.safetensors");
        TensorSpec longTensor = new TensorSpec("long", "I8", List.of(1L), 1L,
                channel -> writeBytes(channel, new byte[]{1, 2}));

        assertThrows(IOException.class,
                () -> StreamingSafetensorWriter.write(output, List.of(longTensor)));
    }

    private static void writeBytes(WritableByteChannel channel, byte[] bytes) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }
}
