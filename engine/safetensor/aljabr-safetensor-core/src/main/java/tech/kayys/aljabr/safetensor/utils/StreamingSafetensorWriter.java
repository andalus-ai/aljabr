package tech.kayys.aljabr.safetensor.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Writes Safetensors payloads sequentially from caller-provided sources.
 *
 * <p>The format requires the complete tensor metadata and byte offsets before
 * the payload, so callers provide a metadata-only plan first. Payload sources
 * are then invoked one at a time after the header has been written. A source
 * should reopen/read its backing shard when invoked rather than capture a
 * callback-scoped memory-mapped tensor view.
 *
 * <p>At most one source needs to be resident at a time; this writer never
 * accumulates tensor payloads in memory.
 */
public final class StreamingSafetensorWriter {

    private static final Gson GSON = new GsonBuilder().create();

    private StreamingSafetensorWriter() {
    }

    /**
     * Write a Safetensors file from a metadata plan and sequential payload sources.
     *
     * @param path destination file
     * @param tensors tensor metadata and payload sources in output order
     * @throws IOException if a source or the destination fails
     */
    public static void write(Path path, List<TensorSpec> tensors) throws IOException {
        write(path, tensors, Map.of());
    }

    /** Write a Safetensors file while preserving file-level string metadata. */
    public static void write(Path path, List<TensorSpec> tensors, Map<String, String> fileMetadata)
            throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        Objects.requireNonNull(fileMetadata, "fileMetadata must not be null");

        JsonObject header = new JsonObject();
        if (!fileMetadata.isEmpty()) {
            JsonObject metadata = new JsonObject();
            fileMetadata.forEach((key, value) -> {
                if (key == null || key.isBlank() || value == null) {
                    throw new IllegalArgumentException("Safetensors file metadata keys and values must be non-null");
                }
                metadata.addProperty(key, value);
            });
            header.add("__metadata__", metadata);
        }
        Set<String> names = new HashSet<>();
        long offset = 0L;
        for (TensorSpec tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor spec must not be null");
            if (!names.add(tensor.name())) {
                throw new IllegalArgumentException("Duplicate tensor name: " + tensor.name());
            }
            long end;
            try {
                end = Math.addExact(offset, tensor.byteSize());
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("Tensor payload offsets overflow for " + tensor.name(), e);
            }

            JsonObject metadata = new JsonObject();
            metadata.addProperty("dtype", tensor.dtype());
            JsonArray shape = new JsonArray();
            tensor.shape().forEach(shape::add);
            metadata.add("shape", shape);
            JsonArray dataOffsets = new JsonArray();
            dataOffsets.add(offset);
            dataOffsets.add(end);
            metadata.add("data_offsets", dataOffsets);
            header.add(tensor.name(), metadata);
            offset = end;
        }

        byte[] headerBytes = alignedHeader(GSON.toJson(header));
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            writeFully(channel, ByteBuffer.allocate(Long.BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.length).flip());
            writeFully(channel, ByteBuffer.wrap(headerBytes));

            for (TensorSpec tensor : tensors) {
                BoundedChannel bounded = new BoundedChannel(channel, tensor.byteSize(), tensor.name());
                tensor.source().writeTo(bounded);
                if (bounded.remaining() != 0L) {
                    throw new IOException("Tensor source '" + tensor.name() + "' wrote "
                            + (tensor.byteSize() - bounded.remaining()) + " bytes; expected "
                            + tensor.byteSize());
                }
            }
        }
    }

    private static byte[] alignedHeader(String json) {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        int padding = (int) ((8 - (Long.BYTES + bytes.length) % 8) % 8);
        if (padding == 0) {
            return bytes;
        }
        byte[] aligned = new byte[bytes.length + padding];
        System.arraycopy(bytes, 0, aligned, 0, bytes.length);
        java.util.Arrays.fill(aligned, bytes.length, aligned.length, (byte) ' ');
        return aligned;
    }

    private static void writeFully(WritableByteChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    /** Metadata for one tensor; payload bytes are supplied only when writing begins. */
    public record TensorSpec(
            String name,
            String dtype,
            List<Long> shape,
            long byteSize,
            TensorDataSource source) {

        public TensorSpec {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Tensor name must not be blank");
            }
            if (dtype == null || dtype.isBlank()) {
                throw new IllegalArgumentException("Tensor dtype must not be blank");
            }
            shape = List.copyOf(Objects.requireNonNull(shape, "shape must not be null"));
            if (shape.stream().anyMatch(dimension -> dimension == null || dimension < 0L)) {
                throw new IllegalArgumentException("Tensor dimensions must be non-negative");
            }
            if (byteSize < 0L) {
                throw new IllegalArgumentException("Tensor byteSize must be non-negative");
            }
            Objects.requireNonNull(source, "source must not be null");
        }
    }

    /** Writes exactly the planned payload bytes to the supplied channel. */
    @FunctionalInterface
    public interface TensorDataSource {
        void writeTo(WritableByteChannel channel) throws IOException;
    }

    private static final class BoundedChannel implements WritableByteChannel {
        private final WritableByteChannel delegate;
        private final String tensorName;
        private long remaining;
        private boolean open = true;

        private BoundedChannel(WritableByteChannel delegate, long expectedBytes, String tensorName) {
            this.delegate = delegate;
            this.remaining = expectedBytes;
            this.tensorName = tensorName;
        }

        @Override
        public int write(ByteBuffer source) throws IOException {
            if (!open) {
                throw new IOException("Tensor output channel is closed");
            }
            if (source.remaining() > remaining) {
                throw new IOException("Tensor source '" + tensorName + "' attempted to write more than "
                        + "its declared byte size");
            }
            int written = delegate.write(source);
            remaining -= written;
            return written;
        }

        @Override
        public boolean isOpen() {
            return open && delegate.isOpen();
        }

        @Override
        public void close() {
            // The writer owns the destination channel; a tensor source does not.
            open = false;
        }

        private long remaining() {
            return remaining;
        }
    }
}
