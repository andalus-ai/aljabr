package tech.kayys.aljabr.converter.ir;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.spi.model.ir.ModelDocument;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Inspects GGUF files into AMIR without copying or retaining tensor payloads.
 *
 * <p>Tensor payload references use absolute container offsets. GGUF tensor
 * dimensions are preserved in their on-disk order; the source ordering is
 * recorded explicitly rather than silently transposing dimensions.
 */
@ApplicationScoped
public final class GgufModelDocumentAdapter {

    private final ObjectMapper objectMapper;
    private final GGUFParser parser;

    @Inject
    public GgufModelDocumentAdapter(ObjectMapper objectMapper) {
        this(objectMapper, new GGUFParser());
    }

    GgufModelDocumentAdapter(ObjectMapper objectMapper, GGUFParser parser) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.parser = Objects.requireNonNull(parser, "parser must not be null");
    }

    /**
     * Reads the GGUF header, metadata and tensor inventory only.
     *
     * @param input GGUF file
     * @return metadata-first AMIR document with lazy source payload references
     */
    public ModelDocument inspect(Path input) throws IOException {
        Objects.requireNonNull(input, "input must not be null");
        Path source = input.toAbsolutePath().normalize();
        if (!Files.isRegularFile(source)) {
            throw new IOException("Input is not a GGUF file: " + source);
        }

        try (Arena arena = Arena.ofShared();
             GGUFReader reader = new GGUFReader(source, arena)) {
            GGUFModel model = parser.parse(reader.segment(), arena);
            return toDocument(source, Files.size(source), model);
        } catch (IllegalArgumentException | IllegalStateException exception) {
            throw new IOException("Unable to inspect GGUF model " + source + ": "
                    + exception.getMessage(), exception);
        }
    }

    private ModelDocument toDocument(Path source, long fileSize, GGUFModel model) throws IOException {
        if (model.dataStart() < 0 || model.dataStart() > fileSize) {
            throw new IOException("GGUF tensor data section starts outside the file: " + model.dataStart());
        }

        Map<String, Object> rawMetadata = model.metadata();
        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, String> extensions = new LinkedHashMap<>();

        String architecture = stringValue(rawMetadata.get("general.architecture"), "unknown");
        metadata.put("general.architecture", architecture);
        metadata.put("gguf.version", Integer.toString(model.version()));
        metadata.put("gguf.alignment", Integer.toString(alignment(rawMetadata)));

        for (Map.Entry<String, Object> entry : rawMetadata.entrySet()) {
            metadata.put("gguf." + entry.getKey(), json(entry.getValue()));
        }
        extensions.put("gguf.metadata.json", json(rawMetadata));
        extensions.put("gguf.tensor.dimension-order", "ggml-fastest-first");

        List<TensorDescriptor> tensors = new ArrayList<>(model.tensors().size());
        for (GGUFTensorInfo info : model.tensors()) {
            GgmlType type;
            try {
                type = GgmlType.fromId(info.typeId());
            } catch (IllegalArgumentException exception) {
                throw new IOException("Unsupported GGUF tensor type " + info.typeId()
                        + " for tensor " + info.name(), exception);
            }

            List<Long> shape = java.util.Arrays.stream(info.shape()).boxed().toList();
            long absoluteOffset;
            long payloadEnd;
            try {
                absoluteOffset = Math.addExact(model.dataStart(), info.offset());
                payloadEnd = Math.addExact(absoluteOffset, info.sizeInBytes());
            } catch (ArithmeticException exception) {
                throw new IOException("GGUF tensor payload range overflows for " + info.name(), exception);
            }
            if (info.offset() < 0 || info.sizeInBytes() < 0 || payloadEnd > fileSize) {
                throw new IOException("GGUF tensor payload is outside the file: " + info.name()
                        + " (offset=" + absoluteOffset + ", length=" + info.sizeInBytes()
                        + ", fileSize=" + fileSize + ")");
            }

            PayloadReference payload = new PayloadReference(
                    source.toUri(),
                    source.getFileName().toString(),
                    absoluteOffset,
                    info.sizeInBytes(),
                    PayloadReference.OffsetBasis.CONTAINER);

            String quantization = type.blockSize > 1 ? type.label : null;
            tensors.add(new TensorDescriptor(
                    info.name(),
                    type.label,
                    shape,
                    "unknown",
                    quantization,
                    payload,
                    Map.of("gguf.type-id", Integer.toString(info.typeId()),
                            "gguf.dimension-order", "ggml-fastest-first")));
        }

        return new ModelDocument(
                modelId(source),
                architecture,
                "gguf",
                metadata,
                extensions,
                List.copyOf(tensors));
    }

    private String json(Object value) throws IOException {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException exception) {
            throw new IOException("Unable to preserve GGUF metadata value", exception);
        }
    }

    private static String stringValue(Object value, String fallback) {
        return value instanceof String text && !text.isBlank() ? text : fallback;
    }

    private static int alignment(Map<String, Object> metadata) {
        Object value = metadata.getOrDefault("general.alignment", 32);
        return value instanceof Number number ? number.intValue() : 32;
    }

    private static String modelId(Path source) {
        String name = source.getFileName() == null ? "model" : source.getFileName().toString();
        int extension = name.lastIndexOf('.');
        return extension > 0 ? name.substring(0, extension) : name;
    }
}
