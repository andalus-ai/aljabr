package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.core.GgufMetaType;
import tech.kayys.aljabr.gguf.core.GgufMetaValue;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Metadata-first GGUF v3 writer that streams tensor payloads from source files.
 *
 * <p>Only the GGUF header and metadata are buffered. Tensor data is copied with
 * a bounded direct buffer, so peak heap use does not grow with model size.
 * Source offsets supplied to this writer are absolute container offsets.
 */
public final class StreamingGgufWriter {
    private static final int MAGIC = 0x46554747;
    private static final int VERSION = 3;
    private static final int COPY_BUFFER_BYTES = 1024 * 1024;
    private static final int DEFAULT_ALIGNMENT = 32;

    private StreamingGgufWriter() {}

    public record TensorPayload(
            String name, long[] shape, GgmlType type, Path source, long sourceOffset, long length) {
        public TensorPayload {
            if (name == null || name.isBlank()) throw new IllegalArgumentException("tensor name must not be blank");
            shape = Objects.requireNonNull(shape, "shape").clone();
            if (shape.length == 0 || shape.length > 4) throw new IllegalArgumentException("GGUF tensors require 1-4 dimensions");
            for (long dimension : shape) if (dimension <= 0) throw new IllegalArgumentException("tensor dimensions must be positive");
            Objects.requireNonNull(type, "type");
            source = Objects.requireNonNull(source, "source").toAbsolutePath().normalize();
            if (sourceOffset < 0 || length < 0) throw new IllegalArgumentException("source range must be non-negative");
        }
        @Override public long[] shape() { return shape.clone(); }
    }

    public record WriteResult(Path output, long dataStart, long payloadBytes, Map<String, Long> tensorOffsets) {
        public WriteResult {
            tensorOffsets = Map.copyOf(tensorOffsets);
        }
    }

    public static WriteResult write(
            Path output, Map<String, GgufMetaValue> metadata, List<TensorPayload> tensors) throws IOException {
        return write(output, metadata, tensors, DEFAULT_ALIGNMENT);
    }

    public static WriteResult write(
            Path output,
            Map<String, GgufMetaValue> metadata,
            List<TensorPayload> tensors,
            int alignment) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(metadata, "metadata");
        Objects.requireNonNull(tensors, "tensors");
        validateAlignment(alignment);

        Path target = output.toAbsolutePath().normalize();
        Map<String, GgufMetaValue> orderedMetadata = new LinkedHashMap<>(metadata);
        List<TensorPayload> orderedTensors = List.copyOf(tensors);
        validateInputs(target, orderedMetadata, orderedTensors);

        Map<String, Long> offsets = new LinkedHashMap<>();
        long relativeOffset = 0;
        long payloadBytes = 0;
        for (TensorPayload tensor : orderedTensors) {
            offsets.put(tensor.name(), relativeOffset);
            long expected = expectedBytes(tensor);
            if (expected != tensor.length()) {
                throw new IOException("Tensor '" + tensor.name() + "' payload length " + tensor.length()
                        + " does not match GGML " + tensor.type().label + " shape size " + expected);
            }
            try {
                payloadBytes = Math.addExact(payloadBytes, expected);
                relativeOffset = alignUp(Math.addExact(relativeOffset, expected), alignment);
            } catch (ArithmeticException overflow) {
                throw new IOException("GGUF tensor data layout overflows long", overflow);
            }
        }

        byte[] header = encodeHeader(orderedMetadata, orderedTensors, offsets, alignment);
        long dataStart = alignUp(header.length, alignment);
        Path parent = target.getParent();
        if (parent == null) throw new IOException("Output must have a parent directory: " + target);
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, "aljabr-gguf-", ".partial");
        boolean committed = false;
        try {
            try (FileChannel destination = FileChannel.open(temporary,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                writeFully(destination, ByteBuffer.wrap(header));
                writePadding(destination, dataStart - header.length);
                ByteBuffer copyBuffer = ByteBuffer.allocateDirect(COPY_BUFFER_BYTES);
                long writtenRelative = 0;
                for (TensorPayload tensor : orderedTensors) {
                    long targetOffset = Math.addExact(dataStart, offsets.get(tensor.name()));
                    writePadding(destination, targetOffset - destination.position());
                    try (FileChannel source = FileChannel.open(tensor.source(), StandardOpenOption.READ)) {
                        long sourceEnd = Math.addExact(tensor.sourceOffset(), tensor.length());
                        if (sourceEnd > source.size()) {
                            throw new IOException("Tensor '" + tensor.name() + "' source range exceeds file: " + tensor.source());
                        }
                        copyRange(source, tensor.sourceOffset(), tensor.length(), destination, copyBuffer);
                    }
                    writtenRelative = Math.addExact(offsets.get(tensor.name()), tensor.length());
                }
                long expectedFileSize = Math.addExact(dataStart, relativeOffset);
                if (destination.position() > expectedFileSize) {
                    throw new IOException("GGUF writer exceeded its precomputed data layout");
                }
                destination.force(true);
            }
            try {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            }
            committed = true;
        } catch (ArithmeticException overflow) {
            throw new IOException("GGUF output offset overflow", overflow);
        } finally {
            if (!committed) Files.deleteIfExists(temporary);
        }
        return new WriteResult(target, dataStart, payloadBytes, offsets);
    }

    private static void validateInputs(Path target, Map<String, GgufMetaValue> metadata,
                                       List<TensorPayload> tensors) throws IOException {
        java.util.HashSet<String> names = new java.util.HashSet<>();
        for (TensorPayload tensor : tensors) {
            if (!names.add(tensor.name())) throw new IOException("Duplicate GGUF tensor name: " + tensor.name());
            if (target.equals(tensor.source())) throw new IOException("Output must not replace a tensor source: " + target);
            if (!Files.isRegularFile(tensor.source())) throw new IOException("Tensor source is not a regular file: " + tensor.source());
            expectedBytes(tensor);
        }
        for (Map.Entry<String, GgufMetaValue> entry : metadata.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null) {
                throw new IOException("GGUF metadata requires non-blank keys and non-null values");
            }
        }
    }

    private static long expectedBytes(TensorPayload tensor) throws IOException {
        try {
            long elements = 1;
            for (long dimension : tensor.shape()) elements = Math.multiplyExact(elements, dimension);
            return tensor.type().bytesFor(elements);
        } catch (ArithmeticException | IllegalArgumentException invalid) {
            throw new IOException("Invalid GGUF tensor layout for '" + tensor.name() + "'", invalid);
        }
    }

    private static byte[] encodeHeader(Map<String, GgufMetaValue> metadata, List<TensorPayload> tensors,
                                       Map<String, Long> offsets, int alignment) throws IOException {
        Map<String, GgufMetaValue> values = new LinkedHashMap<>(metadata);
        values.put("general.alignment", new GgufMetaValue.UInt32Val(alignment));
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            writeInt(out, MAGIC);
            writeInt(out, VERSION);
            writeLong(out, tensors.size());
            writeLong(out, values.size());
            for (Map.Entry<String, GgufMetaValue> entry : values.entrySet()) {
                writeString(out, entry.getKey());
                writeValue(out, entry.getValue());
            }
            for (TensorPayload tensor : tensors) {
                writeString(out, tensor.name());
                long[] shape = tensor.shape();
                writeInt(out, shape.length);
                for (long dimension : shape) writeLong(out, dimension);
                writeInt(out, tensor.type().id);
                writeLong(out, offsets.get(tensor.name()));
            }
        }
        return bytes.toByteArray();
    }

    private static void writeValue(DataOutputStream out, GgufMetaValue value) throws IOException {
        writeInt(out, value.type().id);
        switch (value) {
            case GgufMetaValue.UInt8Val v -> out.writeByte(v.value());
            case GgufMetaValue.Int8Val v -> out.writeByte(v.value());
            case GgufMetaValue.UInt16Val v -> writeShort(out, v.value());
            case GgufMetaValue.Int16Val v -> writeShort(out, v.value());
            case GgufMetaValue.UInt32Val v -> writeInt(out, (int) v.value());
            case GgufMetaValue.Int32Val v -> writeInt(out, v.value());
            case GgufMetaValue.Float32Val v -> writeInt(out, Float.floatToRawIntBits(v.value()));
            case GgufMetaValue.BoolVal v -> out.writeByte(v.value() ? 1 : 0);
            case GgufMetaValue.StringVal v -> writeString(out, v.value());
            case GgufMetaValue.UInt64Val v -> writeLong(out, v.value());
            case GgufMetaValue.Int64Val v -> writeLong(out, v.value());
            case GgufMetaValue.Float64Val v -> writeLong(out, Double.doubleToRawLongBits(v.value()));
            case GgufMetaValue.ArrayVal v -> {
                writeInt(out, v.elementType().id);
                writeLong(out, v.elements().size());
                for (GgufMetaValue element : v.elements()) writeArrayElement(out, element);
            }
        }
    }

    private static void writeArrayElement(DataOutputStream out, GgufMetaValue value) throws IOException {
        switch (value) {
            case GgufMetaValue.UInt8Val v -> out.writeByte(v.value());
            case GgufMetaValue.Int8Val v -> out.writeByte(v.value());
            case GgufMetaValue.UInt16Val v -> writeShort(out, v.value());
            case GgufMetaValue.Int16Val v -> writeShort(out, v.value());
            case GgufMetaValue.UInt32Val v -> writeInt(out, (int) v.value());
            case GgufMetaValue.Int32Val v -> writeInt(out, v.value());
            case GgufMetaValue.Float32Val v -> writeInt(out, Float.floatToRawIntBits(v.value()));
            case GgufMetaValue.BoolVal v -> out.writeByte(v.value() ? 1 : 0);
            case GgufMetaValue.StringVal v -> writeString(out, v.value());
            case GgufMetaValue.UInt64Val v -> writeLong(out, v.value());
            case GgufMetaValue.Int64Val v -> writeLong(out, v.value());
            case GgufMetaValue.Float64Val v -> writeLong(out, Double.doubleToRawLongBits(v.value()));
            case GgufMetaValue.ArrayVal ignored -> throw new IOException("Nested GGUF metadata arrays are not supported");
        }
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
        writeLong(out, encoded.length);
        out.write(encoded);
    }

    private static void writeShort(DataOutputStream out, int value) throws IOException {
        out.writeByte(value & 0xff);
        out.writeByte((value >>> 8) & 0xff);
    }

    private static void writeInt(DataOutputStream out, int value) throws IOException {
        out.writeByte(value & 0xff);
        out.writeByte((value >>> 8) & 0xff);
        out.writeByte((value >>> 16) & 0xff);
        out.writeByte((value >>> 24) & 0xff);
    }

    private static void writeLong(DataOutputStream out, long value) throws IOException {
        writeInt(out, (int) value);
        writeInt(out, (int) (value >>> 32));
    }

    private static void copyRange(FileChannel source, long sourceOffset, long length,
                                  FileChannel destination, ByteBuffer buffer) throws IOException {
        long remaining = length;
        long position = sourceOffset;
        while (remaining > 0) {
            buffer.clear();
            buffer.limit((int) Math.min(buffer.capacity(), remaining));
            int read = source.read(buffer, position);
            if (read < 0) throw new IOException("Unexpected end of tensor payload");
            if (read == 0) continue;
            buffer.flip();
            writeFully(destination, buffer);
            position += read;
            remaining -= read;
        }
    }

    private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    private static void writePadding(FileChannel channel, long count) throws IOException {
        if (count < 0) throw new IOException("GGUF data layout moved backwards");
        ByteBuffer zeros = ByteBuffer.allocate(4096);
        while (count > 0) {
            zeros.clear();
            zeros.limit((int) Math.min(zeros.capacity(), count));
            writeFully(channel, zeros);
            count -= zeros.limit();
        }
    }

    private static long alignUp(long value, long alignment) {
        return Math.addExact(value, alignment - 1) & -((long) alignment);
    }

    private static void validateAlignment(int alignment) {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0 || alignment > 4096) {
            throw new IllegalArgumentException("GGUF alignment must be a power of two in [1, 4096]");
        }
    }
}
