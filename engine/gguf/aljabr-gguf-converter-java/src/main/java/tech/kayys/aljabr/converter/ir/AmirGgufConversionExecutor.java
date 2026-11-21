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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Executes AMIR conversions to GGUF without materializing tensor payloads.
 *
 * <p>Source metadata and opaque model/tensor extensions are carried as
 * namespaced GGUF string metadata. Typed graph, asset, and component layers
 * have no GGUF representation in this writer and therefore require an
 * explicitly acknowledged PARTIAL conversion plan.
 */
public final class AmirGgufConversionExecutor implements ModelConversionExecutor {

    private static final String TARGET_FORMAT = "gguf";

    @Override
    public boolean supports(String sourceFormat, String targetFormat) {
        return sourceFormat != null && !sourceFormat.isBlank()
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
            throw new IOException("GGUF executor requires a file: output URI: " + request.output());
        }
        if (!request.options().isEmpty()) {
            throw new IOException("Unsupported GGUF conversion options: " + request.options().keySet());
        }

        Path target = Path.of(request.output()).toAbsolutePath().normalize();
        Map<String, GgufMetaValue> metadata = metadata(model);
        List<StreamingGgufWriter.TensorPayload> tensors = new ArrayList<>(model.tensors().size());
        long payloadBytes = 0L;
        for (TensorDescriptor tensor : model.tensors()) {
            PayloadReference payload = tensor.payload();
            if (payload == null) {
                throw new IOException("Tensor '" + tensor.name() + "' has no payload reference");
            }
            Path source = filePath(payload.source());
            if (target.equals(source)) {
                throw new IOException("Output must not replace a source tensor container: " + source);
            }
            GgmlType type;
            try {
                type = GgmlType.fromLabel(tensor.dataType());
            } catch (IllegalArgumentException unsupported) {
                throw new IOException("Tensor '" + tensor.name() + "' uses unsupported GGUF dtype '"
                        + tensor.dataType() + "'", unsupported);
            }
            long offset = absoluteOffset(payload, source, model.sourceFormat());
            validatePayload(tensor, payload, source, offset, type);
            try {
                payloadBytes = Math.addExact(payloadBytes, payload.length());
            } catch (ArithmeticException overflow) {
                throw new IOException("Total tensor payload size overflows long", overflow);
            }
            tensors.add(new StreamingGgufWriter.TensorPayload(
                    tensor.name(),
                    tensor.shape().stream().mapToLong(Long::longValue).toArray(),
                    type,
                    source,
                    offset,
                    payload.length()));
        }

        StreamingGgufWriter.WriteResult written = StreamingGgufWriter.write(target, metadata, tensors);
        return new ConversionExecutionResult(
                model.sourceFormat(),
                TARGET_FORMAT,
                written.output().toUri(),
                tensors.size(),
                payloadBytes,
                plan.compatibility(),
                untransferredKeys(model));
    }

    private static void validatePlan(ModelDescriptor model, ConversionPlan plan, ConversionRequest request)
            throws IOException {
        if (!plan.sourceIdentity().equals(model.identity())) {
            throw new IOException("Conversion plan belongs to a different model identity");
        }
        if (!plan.sourceFormat().equalsIgnoreCase(model.sourceFormat())) {
            throw new IOException("Conversion plan source format does not match the AMIR descriptor");
        }
        if (!TARGET_FORMAT.equalsIgnoreCase(plan.targetFormat())) {
            throw new IOException("This executor only writes GGUF, not '" + plan.targetFormat() + "'");
        }
        CompatibilityReport.ConversionFidelity fidelity = plan.compatibility().fidelity();
        if (fidelity == CompatibilityReport.ConversionFidelity.UNSUPPORTED) {
            throw new IOException("Conversion plan is UNSUPPORTED: " + plan.compatibility().diagnostics());
        }
        boolean hasUnrepresentableLayers = model.architectureDescriptor().isPresent()
                || model.graphCollection().isPresent()
                || model.assetCollection().isPresent()
                || model.componentCollection().isPresent();
        if (hasUnrepresentableLayers && fidelity != CompatibilityReport.ConversionFidelity.PARTIAL) {
            throw new IOException("Typed architecture, graph, asset, or component layers require a PARTIAL plan");
        }
        if (fidelity == CompatibilityReport.ConversionFidelity.PARTIAL && !request.acknowledgePartial()) {
            throw new IOException("GGUF conversion is PARTIAL; explicit acknowledgement is required: "
                    + plan.compatibility().diagnostics());
        }
    }

    private static Map<String, GgufMetaValue> metadata(ModelDescriptor model) {
        Map<String, GgufMetaValue> result = new LinkedHashMap<>();
        result.put("general.architecture", GgufMetaValue.ofString(model.architecture()));
        result.put("general.name", GgufMetaValue.ofString(model.identity().id()));
        model.metadata().forEach((key, value) ->
                result.put("aljabr.source.metadata." + key, GgufMetaValue.ofString(value)));
        model.extensions().forEach((key, value) ->
                result.put("aljabr.source.extension." + key, GgufMetaValue.ofString(value)));
        for (TensorDescriptor tensor : model.tensors()) {
            tensor.extensions().forEach((key, value) ->
                    result.put("aljabr.source.tensor."
                                    + metadataKeyPart(tensor.name()) + "." + metadataKeyPart(key),
                            GgufMetaValue.ofString(value)));
        }
        return result;
    }

    private static String metadataKeyPart(String value) {
        return Base64.getUrlEncoder().withoutPadding()
                .encodeToString(value.getBytes(StandardCharsets.UTF_8));
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
            throw new IOException("Only file: payload references are supported: " + source);
        }
        try {
            return Path.of(source).toAbsolutePath().normalize();
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid payload file URI: " + source, invalid);
        }
    }

    private static long absoluteOffset(PayloadReference payload, Path source, String sourceFormat) throws IOException {
        if (payload.offsetBasis() == PayloadReference.OffsetBasis.CONTAINER) {
            return payload.offset();
        }
        if (!"safetensors".equalsIgnoreCase(sourceFormat)) {
            throw new IOException("DATA_SECTION offsets are currently supported only for Safetensors sources: "
                    + sourceFormat);
        }
        try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
            if (channel.size() < Long.BYTES) {
                throw new IOException("Safetensors container is shorter than its header prefix: " + source);
            }
            ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, prefix, 0L);
            prefix.flip();
            long headerLength = prefix.getLong();
            if (headerLength < 0L) {
                throw new IOException("Negative Safetensors header length: " + source);
            }
            try {
                long dataStart = Math.addExact(Long.BYTES, headerLength);
                if (dataStart > channel.size()) {
                    throw new IOException("Safetensors header extends beyond source file: " + source);
                }
                return Math.addExact(dataStart, payload.offset());
            } catch (ArithmeticException overflow) {
                throw new IOException("Safetensors payload offset overflows: " + source, overflow);
            }
        }
    }

    private static void validatePayload(
            TensorDescriptor tensor, PayloadReference payload, Path source, long absoluteOffset, GgmlType type)
            throws IOException {
        long elements = 1L;
        try {
            for (long dimension : tensor.shape()) {
                elements = Math.multiplyExact(elements, dimension);
            }
            if (tensor.shape().isEmpty() || tensor.shape().size() > 4
                    || tensor.shape().stream().anyMatch(dimension -> dimension <= 0L)) {
                throw new IOException("GGUF tensors require one to four positive dimensions: " + tensor.name());
            }
            if (type.blockSize <= 0 || type.typeSize <= 0 || elements % type.blockSize != 0L) {
                throw new IOException("Tensor '" + tensor.name() + "' cannot be represented by GGML type " + type.label);
            }
            long expectedBytes = Math.multiplyExact(elements / type.blockSize, type.typeSize);
            if (expectedBytes != payload.length()) {
                throw new IOException("Tensor '" + tensor.name() + "' declares " + payload.length()
                        + " bytes but GGUF dtype " + type.label + " and shape require " + expectedBytes);
            }
            long end = Math.addExact(absoluteOffset, payload.length());
            if (!Files.isRegularFile(source) || end > Files.size(source)) {
                throw new IOException("Tensor '" + tensor.name() + "' payload range exceeds source file: " + source);
            }
        } catch (ArithmeticException | IllegalArgumentException invalid) {
            throw new IOException("Invalid GGUF payload layout for tensor '" + tensor.name() + "'", invalid);
        }
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
}
