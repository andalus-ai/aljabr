package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter.TensorSpec;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionExecutionResult;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutor;
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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Converts GGUF tensor payloads into Safetensors using bounded-memory streaming.
 *
 * <p>F32/F16/BF16/F64 and integer scalar payloads are copied without changing
 * their representation. Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 are dequantized to F32. Other quantized
 * GGML encodings are rejected until a verified decoder is available.
 *
 * <p>GGUF metadata and opaque AMIR extensions are retained as Safetensors
 * string metadata under an Aljabr namespace. Typed graph, asset, and component
 * layers are not representable in the Safetensors file and are reported.
 */
public final class AmirGgufToSafetensorsConversionExecutor implements ModelConversionExecutor {

    private static final String SOURCE_FORMAT = "gguf";
    private static final String TARGET_FORMAT = "safetensors";
    private static final int COPY_BUFFER_BYTES = 1024 * 1024;
    private static final int BLOCKS_PER_BATCH = 1024;

    @Override
    public boolean supports(String sourceFormat, String targetFormat) {
        return SOURCE_FORMAT.equalsIgnoreCase(sourceFormat)
                && TARGET_FORMAT.equalsIgnoreCase(targetFormat);
    }

    @Override
    public ConversionExecutionResult execute(
            ModelDescriptor model, ConversionPlan plan, ConversionRequest request) throws IOException {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(request, "request must not be null");
        validatePlan(model, plan, request);

        if (!"file".equalsIgnoreCase(request.output().getScheme())) {
            throw new IOException("Safetensors executor requires a file: output URI: " + request.output());
        }
        if (!request.options().isEmpty()) {
            throw new IOException("Unsupported Safetensors conversion options: " + request.options().keySet());
        }

        Path target = Path.of(request.output()).toAbsolutePath().normalize();
        List<TensorSpec> specs = new ArrayList<>(model.tensors().size());
        long outputBytes = 0L;
        for (TensorDescriptor tensor : model.tensors()) {
            PayloadReference payload = tensor.payload();
            if (payload == null) {
                throw new IOException("Tensor '" + tensor.name() + "' has no GGUF payload reference");
            }
            if (payload.offsetBasis() != PayloadReference.OffsetBasis.CONTAINER) {
                throw new IOException("GGUF payload references must use CONTAINER offsets: " + tensor.name());
            }
            Path source = filePath(payload.source());
            if (target.equals(source)) {
                throw new IOException("Output must not replace a source GGUF container: " + source);
            }
            TensorEncoding encoding = encoding(tensor);
            validatePayload(tensor, payload, source, encoding);
            try {
                outputBytes = Math.addExact(outputBytes, encoding.outputBytes(tensor));
            } catch (ArithmeticException overflow) {
                throw new IOException("Total Safetensors payload size overflows long", overflow);
            }
            specs.add(new TensorSpec(
                    tensor.name(),
                    encoding.safetensorsDtype(),
                    safetensorsShape(model, tensor),
                    encoding.outputBytes(tensor),
                    channel -> writePayload(payload, source, tensor, encoding, channel)));
        }

        Map<String, String> metadata = metadata(model);
        Path parent = target.getParent();
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, "aljabr-gguf-conversion-", ".partial");
        boolean committed = false;
        try {
            StreamingSafetensorWriter.write(temporary, specs, metadata);
            try {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            }
            committed = true;
        } finally {
            if (!committed) {
                Files.deleteIfExists(temporary);
            }
        }

        return new ConversionExecutionResult(
                SOURCE_FORMAT, TARGET_FORMAT, target.toUri(), specs.size(), outputBytes,
                plan.compatibility(), untransferredKeys(model));
    }

    private static void validatePlan(ModelDescriptor model, ConversionPlan plan, ConversionRequest request)
            throws IOException {
        if (!SOURCE_FORMAT.equalsIgnoreCase(model.sourceFormat())
                || !SOURCE_FORMAT.equalsIgnoreCase(plan.sourceFormat())
                || !TARGET_FORMAT.equalsIgnoreCase(plan.targetFormat())
                || !plan.sourceIdentity().equals(model.identity())) {
            throw new IOException("Conversion plan does not match this GGUF model and Safetensors target");
        }
        CompatibilityReport.ConversionFidelity fidelity = plan.compatibility().fidelity();
        if (fidelity == CompatibilityReport.ConversionFidelity.UNSUPPORTED) {
            throw new IOException("Conversion plan is UNSUPPORTED: " + plan.compatibility().diagnostics());
        }
        boolean hasLossyQuantization = model.tensors().stream().anyMatch(tensor ->
                isDequantizedType(tensor.dataType()));
        boolean partial = hasUnrepresentableLayers(model)
                || model.architecture().equalsIgnoreCase("unknown") || hasLossyQuantization;
        if (partial && fidelity != CompatibilityReport.ConversionFidelity.PARTIAL) {
            throw new IOException("Dequantization or unrepresented AMIR layers require a PARTIAL plan");
        }
        if (fidelity == CompatibilityReport.ConversionFidelity.PARTIAL && !request.acknowledgePartial()) {
            throw new IOException("GGUF-to-Safetensors conversion is PARTIAL; explicit acknowledgement is required");
        }
    }

    private static TensorEncoding encoding(TensorDescriptor tensor) throws IOException {
        String dtype = tensor.dataType().toUpperCase(Locale.ROOT);
        return switch (dtype) {
            case "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64" ->
                    TensorEncoding.raw(dtype, bytesPerElement(dtype));
            case "Q4_0" -> TensorEncoding.dequantized("Q4_0", 18);
            case "Q4_1" -> TensorEncoding.dequantized("Q4_1", 20);
            case "Q5_0" -> TensorEncoding.dequantized("Q5_0", 22);
            case "Q5_1" -> TensorEncoding.dequantized("Q5_1", 24);
            case "Q8_0" -> TensorEncoding.dequantized("Q8_0", 34);
            default -> throw new IOException("No verified GGUF-to-Safetensors decoder for dtype '"
                    + tensor.dataType() + "' (tensor " + tensor.name() + ")");
        };
    }

    private static boolean isDequantizedType(String dtype) {
        return switch (dtype.toUpperCase(Locale.ROOT)) {
            case "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0" -> true;
            default -> false;
        };
    }

    private static int bytesPerElement(String dtype) {
        return switch (dtype) {
            case "I8" -> 1;
            case "F16", "BF16", "I16" -> 2;
            case "F32", "I32" -> 4;
            case "F64", "I64" -> 8;
            default -> throw new IllegalArgumentException("Unsupported scalar dtype " + dtype);
        };
    }

    private static void validatePayload(
            TensorDescriptor tensor, PayloadReference payload, Path source, TensorEncoding encoding)
            throws IOException {
        long elements = 1L;
        try {
            if (tensor.shape().isEmpty() || tensor.shape().stream().anyMatch(d -> d == null || d <= 0L)) {
                throw new IOException("Tensor dimensions must be positive: " + tensor.name());
            }
            for (long dimension : tensor.shape()) {
                elements = Math.multiplyExact(elements, dimension);
            }
            if (encoding.dequantized() && elements % 32L != 0L) {
                throw new IOException("Tensor '" + tensor.name() + "' element count must be divisible by 32 "
                        + "for " + encoding.ggmlType());
            }
            long expectedInput = encoding.inputBytes(elements);
            long expectedOutput = encoding.outputBytes(elements);
            if (payload.length() != expectedInput) {
                throw new IOException("Tensor '" + tensor.name() + "' declares " + payload.length()
                        + " input bytes but " + encoding.ggmlType() + " requires " + expectedInput);
            }
            long end = Math.addExact(payload.offset(), payload.length());
            if (!Files.isRegularFile(source) || end > Files.size(source)) {
                throw new IOException("Tensor '" + tensor.name() + "' payload range exceeds source file");
            }
            if (expectedOutput <= 0L) {
                throw new IOException("Tensor output size must be positive: " + tensor.name());
            }
        } catch (ArithmeticException overflow) {
            throw new IOException("Tensor shape or payload size overflows for " + tensor.name(), overflow);
        }
    }

    private static void writePayload(
            PayloadReference payload, Path source, TensorDescriptor tensor,
            TensorEncoding encoding, WritableByteChannel output) throws IOException {
        try (FileChannel input = FileChannel.open(source, StandardOpenOption.READ)) {
            if (encoding.dequantized()) {
                dequantize(input, payload.offset(), tensor.shape(), encoding.ggmlType(), output);
            } else {
                copy(input, payload.offset(), payload.length(), output);
            }
        }
    }

    private static void copy(FileChannel input, long offset, long length, WritableByteChannel output)
            throws IOException {
        ByteBuffer buffer = ByteBuffer.allocateDirect(COPY_BUFFER_BYTES);
        long position = offset;
        long remaining = length;
        while (remaining > 0L) {
            checkInterrupted();
            buffer.clear();
            buffer.limit((int) Math.min(buffer.capacity(), remaining));
            int read = input.read(buffer, position);
            if (read < 0) throw new IOException("Unexpected end of GGUF tensor payload");
            if (read == 0) continue;
            buffer.flip();
            writeFully(output, buffer);
            position += read;
            remaining -= read;
        }
    }

    private static void dequantize(
            FileChannel input, long offset, List<Long> shape, String type, WritableByteChannel output)
            throws IOException {
        long elements = 1L;
        for (long dimension : shape) elements = Math.multiplyExact(elements, dimension);
        int blockBytes = switch (type) {
            case "Q4_0" -> 18;
            case "Q4_1" -> 20;
            case "Q5_0" -> 22;
            case "Q5_1" -> 24;
            case "Q8_0" -> 34;
            default -> throw new IOException("Unsupported dequantization type: " + type);
        };
        int valuesPerBlock = 32;
        long blocks = elements / valuesPerBlock;
        byte[] source = new byte[blockBytes * BLOCKS_PER_BATCH];
        ByteBuffer values = ByteBuffer.allocateDirect(valuesPerBlock * Float.BYTES * BLOCKS_PER_BATCH)
                .order(ByteOrder.LITTLE_ENDIAN);
        long block = 0L;
        while (block < blocks) {
            checkInterrupted();
            int count = (int) Math.min(BLOCKS_PER_BATCH, blocks - block);
            int bytes = count * blockBytes;
            ByteBuffer packed = ByteBuffer.wrap(source, 0, bytes);
            readFully(input, packed, offset + block * blockBytes);
            values.clear();
            for (int i = 0; i < count; i++) {
                int base = i * blockBytes;
                float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
                if (type.equals("Q8_0")) {
                    for (int j = 0; j < 32; j++) {
                        values.putFloat(source[base + 2 + j] * scale);
                    }
                    continue;
                }

                float minimum = switch (type) {
                    case "Q4_1", "Q5_1" -> halfToFloat(source[base + 2] & 0xff, source[base + 3] & 0xff);
                    default -> 0.0f;
                };
                int highBitsOffset = type.equals("Q5_0") ? base + 2 : base + 4;
                int quantizedValuesOffset = switch (type) {
                    case "Q4_0" -> base + 2;
                    case "Q4_1" -> base + 4;
                    case "Q5_0" -> base + 6;
                    case "Q5_1" -> base + 8;
                    default -> throw new IOException("Unsupported dequantization type: " + type);
                };
                long highBits = type.startsWith("Q5") ? readUInt32LE(source, highBitsOffset) : 0L;
                for (int j = 0; j < 16; j++) {
                    int packedValue = source[quantizedValuesOffset + j] & 0xff;
                    int quantized = packedValue & 0x0f;
                    if (type.startsWith("Q5")) quantized |= (int) ((highBits >>> j) & 1L) << 4;
                    values.putFloat(dequantizeValue(type, quantized, scale, minimum));
                }
                for (int j = 0; j < 16; j++) {
                    int packedValue = source[quantizedValuesOffset + j] & 0xff;
                    int quantized = packedValue >>> 4;
                    if (type.startsWith("Q5")) quantized |= (int) ((highBits >>> (j + 16)) & 1L) << 4;
                    values.putFloat(dequantizeValue(type, quantized, scale, minimum));
                }
            }
            values.flip();
            writeFully(output, values);
            block += count;
        }
    }

    private static float dequantizeValue(String type, int quantized, float scale, float minimum) {
        return switch (type) {
            case "Q4_0" -> (quantized - 8) * scale;
            case "Q4_1", "Q5_1" -> quantized * scale + minimum;
            case "Q5_0" -> (quantized - 16) * scale;
            default -> throw new IllegalArgumentException("Unsupported quantization type: " + type);
        };
    }

    private static long readUInt32LE(byte[] bytes, int offset) {
        return (bytes[offset] & 0xffL)
                | ((bytes[offset + 1] & 0xffL) << 8)
                | ((bytes[offset + 2] & 0xffL) << 16)
                | ((bytes[offset + 3] & 0xffL) << 24);
    }

    private static float halfToFloat(int low, int high) {
        return Float.float16ToFloat((short) (low | (high << 8)));
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("Unexpected end of GGUF tensor payload");
            if (read == 0) continue;
            position += read;
        }
    }

    private static void writeFully(WritableByteChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    private static void checkInterrupted() throws IOException {
        if (Thread.currentThread().isInterrupted()) {
            throw new IOException("GGUF-to-Safetensors conversion interrupted");
        }
    }

    private static List<Long> safetensorsShape(ModelDescriptor model, TensorDescriptor tensor) {
        boolean ggmlFastestFirst = "ggml-fastest-first".equalsIgnoreCase(
                tensor.extensions().getOrDefault("gguf.dimension-order",
                        model.extensions().get("gguf.tensor.dimension-order")));
        if (!ggmlFastestFirst || tensor.shape().size() < 2) {
            return tensor.shape();
        }
        List<Long> reversed = new ArrayList<>(tensor.shape());
        java.util.Collections.reverse(reversed);
        return List.copyOf(reversed);
    }

    private static Map<String, String> metadata(ModelDescriptor model) {
        Map<String, String> values = new LinkedHashMap<>();
        values.put("aljabr.source.format", SOURCE_FORMAT);
        values.put("aljabr.source.architecture", model.architecture());
        model.metadata().forEach((key, value) ->
                values.put("aljabr.source.metadata." + key, value));
        model.extensions().forEach((key, value) ->
                values.put("aljabr.source.extension." + key, value));
        for (TensorDescriptor tensor : model.tensors()) {
            tensor.extensions().forEach((key, value) -> values.put(
                    "aljabr.source.tensor." + keyPart(tensor.name()) + "." + keyPart(key), value));
        }
        return Map.copyOf(values);
    }

    private static String keyPart(String value) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    private static boolean hasUnrepresentableLayers(ModelDescriptor model) {
        return model.architectureDescriptor().isPresent() || model.graphCollection().isPresent()
                || model.assetCollection().isPresent() || model.componentCollection().isPresent();
    }

    private static List<String> untransferredKeys(ModelDescriptor model) {
        List<String> keys = new ArrayList<>();
        model.architectureDescriptor().ifPresent(ignored -> keys.add("model:architectureDescriptor"));
        model.graphCollection().ifPresent(ignored -> keys.add("model:graphCollection"));
        model.assetCollection().ifPresent(ignored -> keys.add("model:assetCollection"));
        model.componentCollection().ifPresent(ignored -> keys.add("model:componentCollection"));
        return List.copyOf(keys);
    }

    private static Path filePath(URI source) throws IOException {
        if (!"file".equalsIgnoreCase(source.getScheme())) {
            throw new IOException("Only file: GGUF payload references are supported: " + source);
        }
        try {
            return Path.of(source).toAbsolutePath().normalize();
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid GGUF payload URI: " + source, invalid);
        }
    }

    private record TensorEncoding(String ggmlType, String safetensorsDtype, int scalarBytes, boolean dequantized) {
        static TensorEncoding raw(String dtype, int scalarBytes) {
            return new TensorEncoding(dtype, dtype, scalarBytes, false);
        }
        static TensorEncoding dequantized(String type, int blockBytes) {
            return new TensorEncoding(type, "F32", blockBytes, true);
        }
        long inputBytes(long elements) {
            return dequantized ? Math.multiplyExact(elements / 32L, scalarBytes)
                    : Math.multiplyExact(elements, scalarBytes);
        }
        long outputBytes(TensorDescriptor tensor) {
            return outputBytes(tensor.shape().stream().mapToLong(Long::longValue)
                    .reduce(1L, Math::multiplyExact));
        }
        long outputBytes(long elements) {
            return Math.multiplyExact(elements, dequantized ? Float.BYTES : scalarBytes);
        }
    }
}
