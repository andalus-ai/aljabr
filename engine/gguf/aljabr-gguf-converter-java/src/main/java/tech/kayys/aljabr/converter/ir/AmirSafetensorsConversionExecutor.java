package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter.TensorSpec;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Executes the first AMIR-backed conversion vertical slice: Safetensors to
 * Safetensors. Tensor payloads are copied directly from their referenced
 * source ranges in bounded chunks; no tensor is materialized as a heap array.
 *
 * <p>Opaque AMIR extensions cannot be represented in a Safetensors file.
 * A PARTIAL plan therefore requires explicit acknowledgement, and the result
 * reports exactly which extension keys were not transferred.
 */
public final class AmirSafetensorsConversionExecutor {

    private static final long COPY_CHUNK_BYTES = 1024L * 1024L;
    private static final String TARGET_FORMAT = "safetensors";

    /**
     * Execute a previously-created compatibility plan.
     *
     * @param model AMIR descriptor containing tensor payload references
     * @param plan preflight plan for this exact model and target
     * @param output destination Safetensors file
     * @param acknowledgePartial whether the caller explicitly accepts PARTIAL fidelity
     */
    public ConversionResult execute(
            ModelDescriptor model,
            ConversionPlan plan,
            Path output,
            boolean acknowledgePartial) throws IOException {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(output, "output must not be null");
        validatePlan(model, plan, acknowledgePartial);

        Path target = output.toAbsolutePath().normalize();
        List<TensorSpec> tensorSpecs = new ArrayList<>(model.tensors().size());
        long totalBytes = 0L;
        for (TensorDescriptor tensor : model.tensors()) {
            PayloadReference payload = tensor.payload();
            if (payload == null) {
                throw new IOException("Tensor '" + tensor.name() + "' has no payload reference");
            }
            Path source = filePath(payload.source());
            if (source.equals(target)) {
                throw new IOException("Output must not replace a source tensor container: " + source);
            }
            validateTensorPayload(tensor, payload);
            totalBytes = Math.addExact(totalBytes, payload.length());
            tensorSpecs.add(new TensorSpec(
                    tensor.name(),
                    tensor.dataType().toUpperCase(Locale.ROOT),
                    tensor.shape(),
                    payload.length(),
                    channel -> copyPayload(payload, source, channel)));
        }

        Map<String, String> metadata = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : model.metadata().entrySet()) {
            String key = entry.getKey();
            String outputKey = key.startsWith("safetensors.")
                    ? key.substring("safetensors.".length()) : key;
            String previous = metadata.putIfAbsent(outputKey, entry.getValue());
            if (previous != null && !previous.equals(entry.getValue())) {
                throw new IOException("Conflicting Safetensors metadata key after namespace mapping: "
                        + outputKey);
            }
        }

        Path parent = target.getParent();
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, "aljabr-conversion-", ".partial");
        boolean committed = false;
        try {
            StreamingSafetensorWriter.write(temporary, tensorSpecs, metadata);
            try {
                Files.move(temporary, target,
                        StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            }
            committed = true;
        } finally {
            if (!committed) {
                Files.deleteIfExists(temporary);
            }
        }

        return new ConversionResult(
                tensorSpecs.size(),
                totalBytes,
                target,
                List.copyOf(model.extensions().keySet()),
                plan);
    }

    private static void validatePlan(
            ModelDescriptor model, ConversionPlan plan, boolean acknowledgePartial) throws IOException {
        if (!plan.sourceIdentity().equals(model.identity())) {
            throw new IOException("Conversion plan belongs to a different model identity");
        }
        if (!plan.sourceFormat().equalsIgnoreCase(model.sourceFormat())) {
            throw new IOException("Conversion plan source format does not match the AMIR descriptor");
        }
        if (!TARGET_FORMAT.equalsIgnoreCase(model.sourceFormat())) {
            throw new IOException("This first executor only supports Safetensors source models, not '"
                    + model.sourceFormat() + "'");
        }
        if (!TARGET_FORMAT.equalsIgnoreCase(plan.targetFormat())) {
            throw new IOException("This executor only writes Safetensors, not '" + plan.targetFormat() + "'");
        }
        CompatibilityReport.ConversionFidelity fidelity = plan.compatibility().fidelity();
        if (fidelity == CompatibilityReport.ConversionFidelity.UNSUPPORTED) {
            throw new IOException("Conversion plan is UNSUPPORTED: " + plan.compatibility().diagnostics());
        }
        if (fidelity == CompatibilityReport.ConversionFidelity.PARTIAL && !acknowledgePartial) {
            throw new IOException("Conversion plan is PARTIAL; explicit acknowledgement is required: "
                    + plan.compatibility().diagnostics());
        }
    }

    private static void validateTensorPayload(TensorDescriptor tensor, PayloadReference payload)
            throws IOException {
        long elements = 1L;
        try {
            for (long dimension : tensor.shape()) {
                elements = Math.multiplyExact(elements, dimension);
            }
            long expectedBytes = Math.multiplyExact(elements, bytesPerElement(tensor.dataType()));
            if (expectedBytes != payload.length()) {
                throw new IOException("Tensor '" + tensor.name() + "' declares " + payload.length()
                        + " payload bytes but shape and dtype require " + expectedBytes);
            }
        } catch (ArithmeticException overflow) {
            throw new IOException("Tensor shape or byte size overflows for '" + tensor.name() + "'", overflow);
        }
    }

    private static int bytesPerElement(String dataType) throws IOException {
        return switch (dataType.toUpperCase(Locale.ROOT)) {
            case "BOOL", "F8_E4M3", "F8_E5M2", "I8", "U8" -> 1;
            case "BF16", "F16", "I16", "U16" -> 2;
            case "F32", "I32", "U32" -> 4;
            case "F64", "I64", "U64" -> 8;
            default -> throw new IOException("No scalar byte width is defined for dtype '" + dataType + "'");
        };
    }

    private static Path filePath(URI source) throws IOException {
        if (!"file".equalsIgnoreCase(source.getScheme())) {
            throw new IOException("Only file: payload references are supported by this executor: " + source);
        }
        try {
            return Path.of(source).toAbsolutePath().normalize();
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid file payload URI: " + source, invalid);
        }
    }

    private static void copyPayload(
            PayloadReference payload, Path source, WritableByteChannel output) throws IOException {
        try (FileChannel input = FileChannel.open(source, StandardOpenOption.READ)) {
            long baseOffset = switch (payload.offsetBasis()) {
                case CONTAINER -> 0L;
                case DATA_SECTION -> dataSectionOffset(input, source);
            };
            long start;
            long end;
            try {
                start = Math.addExact(baseOffset, payload.offset());
                end = Math.addExact(start, payload.length());
            } catch (ArithmeticException overflow) {
                throw new IOException("Payload range overflows for " + source, overflow);
            }
            if (end > input.size()) {
                throw new IOException("Payload range [" + start + ", " + end
                        + ") exceeds source container size " + input.size() + ": " + source);
            }

            ByteBuffer buffer = ByteBuffer.allocateDirect((int) COPY_CHUNK_BYTES);
            long position = start;
            long remaining = payload.length();
            while (remaining > 0L) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new IOException("Safetensors conversion interrupted");
                }
                buffer.clear();
                buffer.limit((int) Math.min(buffer.capacity(), remaining));
                int read = input.read(buffer, position);
                if (read < 0) {
                    throw new IOException("Unexpected end of payload in " + source);
                }
                if (read == 0) {
                    Thread.onSpinWait();
                    continue;
                }
                buffer.flip();
                while (buffer.hasRemaining()) {
                    output.write(buffer);
                }
                position += read;
                remaining -= read;
            }
        }
    }

    private static long dataSectionOffset(FileChannel input, Path source) throws IOException {
        if (input.size() < Long.BYTES) {
            throw new IOException("Safetensors container is shorter than its header prefix: " + source);
        }
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(input, prefix, 0L);
        prefix.flip();
        long headerLength = prefix.getLong();
        long dataOffset;
        try {
            dataOffset = Math.addExact(Long.BYTES, headerLength);
        } catch (ArithmeticException overflow) {
            throw new IOException("Safetensors header length overflows: " + source, overflow);
        }
        if (headerLength < 0L || dataOffset > input.size()) {
            throw new IOException("Invalid Safetensors header length " + headerLength + ": " + source);
        }
        return dataOffset;
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) {
                throw new IOException("Unexpected end of file while reading Safetensors header");
            }
            if (read == 0) {
                Thread.onSpinWait();
                continue;
            }
            position += read;
        }
    }

    /** Summary of a committed conversion. */
    public record ConversionResult(
            int tensorCount,
            long payloadBytes,
            Path output,
            List<String> untransferredExtensionKeys,
            ConversionPlan plan) {
        public ConversionResult {
            untransferredExtensionKeys = List.copyOf(untransferredExtensionKeys);
        }
    }
}
