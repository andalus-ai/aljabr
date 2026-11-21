package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.core.GgufMetaValue;
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
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * First AMIR GGUF conversion path: Safetensors payloads to GGUF storage.
 *
 * <p>This executor intentionally performs no quantization or dequantization.
 * It accepts only scalar dtypes with an exact GGML storage equivalent and
 * transfers tensor bytes unchanged. Source payload ranges are resolved from
 * their declared offset basis and streamed by {@link StreamingGgufWriter}.
 */
public final class AmirGgufConversionExecutor implements ModelConversionExecutor {
    private static final String SOURCE_FORMAT = "safetensors";
    private static final String TARGET_FORMAT = "gguf";
    private static final String ALIGNMENT_OPTION = "gguf.alignment";

    @Override
    public boolean supports(String sourceFormat, String targetFormat) {
        return SOURCE_FORMAT.equalsIgnoreCase(Objects.requireNonNullElse(sourceFormat, ""))
                && TARGET_FORMAT.equalsIgnoreCase(Objects.requireNonNullElse(targetFormat, ""));
    }

    @Override
    public ConversionExecutionResult execute(
            ModelDescriptor model, ConversionPlan plan, ConversionRequest request) throws IOException {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(request, "request must not be null");
        validatePlan(model, plan, request);

        if (!"file".equalsIgnoreCase(request.output().getScheme())) {
            throw new IOException("GGUF executor requires a file: output URI: " + request.output());
        }
        int alignment = parseAlignment(request.options());
        Path output = Path.of(request.output()).toAbsolutePath().normalize();

        List<StreamingGgufWriter.TensorPayload> tensors = new ArrayList<>(model.tensors().size());
        long payloadBytes = 0L;
        for (TensorDescriptor tensor : model.tensors()) {
            PayloadReference payload = tensor.payload();
            if (payload == null) {
                throw new IOException("Tensor '" + tensor.name() + "' has no payload reference");
            }
            Path source = filePath(payload.source());
            long sourceOffset = resolveOffset(payload, source);
            GgmlType targetType = targetType(tensor.dataType());
            long expectedBytes = expectedBytes(tensor, targetType);
            if (expectedBytes != payload.length()) {
                throw new IOException("Tensor '" + tensor.name() + "' has " + payload.length()
                        + " source bytes but GGML " + targetType.label + " requires " + expectedBytes);
            }
            try {
                payloadBytes = Math.addExact(payloadBytes, expectedBytes);
            } catch (ArithmeticException overflow) {
                throw new IOException("Total GGUF payload size overflows long", overflow);
            }
            tensors.add(new StreamingGgufWriter.TensorPayload(
                    tensor.name(), toDimensions(tensor), targetType, source, sourceOffset, payload.length()));
        }

        Map<String, GgufMetaValue> metadata = mapMetadata(model);
        StreamingGgufWriter.WriteResult written = StreamingGgufWriter.write(
                output, metadata, tensors, alignment);
        List<String> losses = untransferredInformation(model);
        return new ConversionExecutionResult(
                model.sourceFormat(), TARGET_FORMAT, written.output().toUri(),
                tensors.size(), payloadBytes, plan.compatibility(), losses);
    }

    private static void validatePlan(
            ModelDescriptor model, ConversionPlan plan, ConversionRequest request) throws IOException {
        if (!plan.sourceIdentity().equals(model.identity())) {
            throw new IOException("Conversion plan belongs to a different model identity");
        }
        if (!plan.sourceFormat().equalsIgnoreCase(model.sourceFormat())
                || !SOURCE_FORMAT.equalsIgnoreCase(model.sourceFormat())) {
            throw new IOException("GGUF executor currently requires a Safetensors source descriptor");
        }
        if (!TARGET_FORMAT.equalsIgnoreCase(plan.targetFormat())) {
            throw new IOException("Conversion plan target is not GGUF: " + plan.targetFormat());
        }
        if (plan.compatibility().fidelity() == CompatibilityReport.ConversionFidelity.UNSUPPORTED) {
            throw new IOException("Conversion plan is UNSUPPORTED: " + plan.compatibility().diagnostics());
        }
        List<String> losses = untransferredInformation(model);
        if (!losses.isEmpty()
                && plan.compatibility().fidelity() != CompatibilityReport.ConversionFidelity.PARTIAL) {
            throw new IOException("Model contains information not represented by this GGUF path; "
                    + "preflight must report PARTIAL fidelity: " + losses);
        }
        if (plan.compatibility().fidelity() == CompatibilityReport.ConversionFidelity.PARTIAL
                && !request.acknowledgePartial()) {
            throw new IOException("GGUF conversion plan is PARTIAL; explicit acknowledgement is required");
        }
    }

    private static List<String> untransferredInformation(ModelDescriptor model) {
        List<String> losses = new ArrayList<>();
        model.extensions().keySet().forEach(key -> losses.add("model:" + key));
        model.tensors().forEach(tensor -> tensor.extensions().keySet()
                .forEach(key -> losses.add("tensor:" + tensor.name() + ":" + key)));
        model.graphCollection().ifPresent(graphs -> graphs.graphs()
                .forEach(graph -> losses.add("graph:" + graph.id())));
        model.assetCollection().ifPresent(assets -> assets.assets()
                .forEach(asset -> losses.add("asset:" + asset.id())));
        model.componentCollection().ifPresent(components -> components.components()
                .forEach(component -> losses.add("component:" + component.id())));
        return List.copyOf(losses);
    }

    private static Map<String, GgufMetaValue> mapMetadata(ModelDescriptor model) throws IOException {
        Map<String, GgufMetaValue> result = new LinkedHashMap<>();
        putMetadata(result, "general.architecture", model.architecture());
        for (Map.Entry<String, String> entry : model.metadata().entrySet()) {
            String key = entry.getKey();
            String targetKey = key.startsWith("gguf.")
                    ? key.substring("gguf.".length())
                    : "aljabr.source." + key;
            if (targetKey.equals("general.architecture")) {
                if (!model.architecture().equals(entry.getValue())) {
                    throw new IOException("Conflicting general.architecture metadata");
                }
                continue;
            }
            putMetadata(result, targetKey, entry.getValue());
        }
        return Map.copyOf(result);
    }

    private static void putMetadata(Map<String, GgufMetaValue> target, String key, String value)
            throws IOException {
        if (key == null || key.isBlank() || value == null) {
            throw new IOException("GGUF metadata keys and values must be non-null and non-blank");
        }
        if (target.putIfAbsent(key, GgufMetaValue.ofString(value)) != null) {
            throw new IOException("Duplicate GGUF metadata key after mapping: " + key);
        }
    }

    private static int parseAlignment(Map<String, String> options) throws IOException {
        for (String key : options.keySet()) {
            if (!ALIGNMENT_OPTION.equals(key)) {
                throw new IOException("Unsupported GGUF conversion option: " + key);
            }
        }
        String configured = options.get(ALIGNMENT_OPTION);
        if (configured == null) return 32;
        try {
            int alignment = Integer.parseInt(configured);
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0 || alignment > 4096) {
                throw new NumberFormatException("not a supported power-of-two alignment");
            }
            return alignment;
        } catch (NumberFormatException invalid) {
            throw new IOException("gguf.alignment must be a power of two in [1, 4096]", invalid);
        }
    }

    private static GgmlType targetType(String sourceType) throws IOException {
        return switch (sourceType.toUpperCase(Locale.ROOT)) {
            case "F32", "FLOAT32" -> GgmlType.F32;
            case "F16", "FLOAT16" -> GgmlType.F16;
            case "BF16", "BFLOAT16" -> GgmlType.BF16;
            case "F64", "FLOAT64" -> GgmlType.F64;
            case "I8", "INT8" -> GgmlType.I8;
            case "I16", "INT16" -> GgmlType.I16;
            case "I32", "INT32" -> GgmlType.I32;
            case "I64", "INT64" -> GgmlType.I64;
            default -> throw new IOException("Safetensors dtype '" + sourceType
                    + "' has no exact GGML storage mapping; implicit quantization is disabled");
        };
    }

    private static long expectedBytes(TensorDescriptor tensor, GgmlType type) throws IOException {
        try {
            long elements = 1;
            for (long dimension : tensor.shape()) {
                if (dimension <= 0) throw new IOException("Tensor dimensions must be positive: " + tensor.name());
                elements = Math.multiplyExact(elements, dimension);
            }
            return type.bytesFor(elements);
        } catch (ArithmeticException | IllegalArgumentException invalid) {
            throw new IOException("Invalid GGUF tensor shape for '" + tensor.name() + "'", invalid);
        }
    }

    private static long[] toDimensions(TensorDescriptor tensor) throws IOException {
        // AMIR represents scalars with an empty shape. GGUF tensor descriptors
        // use an explicit dimension vector, so encode a scalar as a one-element
        // vector of length one while preserving its single-element payload.
        if (tensor.shape().isEmpty()) {
            return new long[] {1L};
        }
        long[] dimensions = new long[tensor.shape().size()];
        for (int i = 0; i < dimensions.length; i++) dimensions[i] = tensor.shape().get(i);
        return dimensions;
    }

    private static Path filePath(URI source) throws IOException {
        if (!"file".equalsIgnoreCase(source.getScheme())) {
            throw new IOException("Only file: tensor payload containers are supported: " + source);
        }
        try {
            return Path.of(source).toAbsolutePath().normalize();
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid tensor payload URI: " + source, invalid);
        }
    }

    private static long resolveOffset(PayloadReference payload, Path source) throws IOException {
        if (payload.offsetBasis() == PayloadReference.OffsetBasis.CONTAINER) return payload.offset();
        try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
            if (channel.size() < Long.BYTES) {
                throw new IOException("Safetensors container is shorter than its header prefix: " + source);
            }
            ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            long position = 0;
            while (prefix.hasRemaining()) {
                int read = channel.read(prefix, position);
                if (read < 0) throw new IOException("Unexpected EOF reading Safetensors header: " + source);
                if (read == 0) continue;
                position += read;
            }
            prefix.flip();
            long headerLength = prefix.getLong();
            if (headerLength < 0) throw new IOException("Negative Safetensors header length: " + source);
            long dataOffset;
            try {
                dataOffset = Math.addExact(Long.BYTES, headerLength);
            } catch (ArithmeticException overflow) {
                throw new IOException("Safetensors data offset overflows: " + source, overflow);
            }
            if (dataOffset > channel.size()) {
                throw new IOException("Safetensors header extends beyond source file: " + source);
            }
            try {
                return Math.addExact(dataOffset, payload.offset());
            } catch (ArithmeticException overflow) {
                throw new IOException("Safetensors payload offset overflows: " + source, overflow);
            }
        }
    }
}
