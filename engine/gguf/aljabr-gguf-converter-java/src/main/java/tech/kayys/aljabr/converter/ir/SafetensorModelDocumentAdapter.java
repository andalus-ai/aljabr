package tech.kayys.aljabr.converter.ir;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorHeader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardIndex;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensorInfo;
import tech.kayys.aljabr.spi.model.ir.ModelDocument;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Builds AMIR model metadata from a Safetensors checkpoint without mapping or
 * reading tensor payload bytes.
 *
 * <p>Supports a single file, a directory containing one Safetensors file, and
 * indexed multi-shard Hugging Face checkpoints. Payload references retain the
 * source shard and Safetensors data-section-relative offsets.
 */
@ApplicationScoped
public final class SafetensorModelDocumentAdapter {

    private static final String CONFIG_FILE = "config.json";

    private final SafetensorFFMLoader loader;
    private final ObjectMapper objectMapper;

    @Inject
    public SafetensorModelDocumentAdapter(SafetensorFFMLoader loader, ObjectMapper objectMapper) {
        this.loader = Objects.requireNonNull(loader, "loader must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
    }

    /**
     * Inspect a checkpoint and return its metadata-only AMIR document.
     *
     * @param input Safetensors file or checkpoint directory
     */
    public ModelDocument inspect(Path input) throws IOException {
        Objects.requireNonNull(input, "input must not be null");
        Path source = input.toAbsolutePath().normalize();
        if (Files.isRegularFile(source)) {
            return fromSingleFile(source);
        }
        if (!Files.isDirectory(source)) {
            throw new IOException("Input is not a Safetensors file or directory: " + source);
        }

        if (SafetensorShardIndex.isShardedModel(source)) {
            return fromShardedDirectory(source);
        }

        try (var files = Files.list(source)) {
            List<Path> candidates = files
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".safetensors"))
                    .sorted()
                    .toList();
            if (candidates.size() != 1) {
                throw new IOException("Expected exactly one .safetensors file in " + source
                        + " but found " + candidates.size());
            }
            return fromSingleFile(candidates.get(0));
        }
    }

    private ModelDocument fromSingleFile(Path file) throws IOException {
        SafetensorHeader header = loader.loadHeaderOnly(file);
        List<TensorDescriptor> tensors = descriptors(file, header);
        return document(file, header.fileMetadata(), tensors);
    }

    private ModelDocument fromShardedDirectory(Path directory) throws IOException {
        SafetensorShardIndex index = SafetensorShardIndex.load(
                SafetensorShardIndex.resolveIndexPath(directory), objectMapper);
        Map<String, SafetensorHeader> headers = new LinkedHashMap<>();
        for (String shardName : index.shardFileNames()) {
            headers.put(shardName, loader.loadHeaderOnly(index.shardPath(shardName)));
        }
        index.validateShardHeaders(headers);

        Map<String, String> metadata = mergeFileMetadata(headers);
        List<TensorDescriptor> tensors = new ArrayList<>(index.tensorCount());
        for (String shardName : index.shardFileNames()) {
            Path shard = index.shardPath(shardName);
            tensors.addAll(descriptors(shard, headers.get(shardName)));
        }
        return document(directory, metadata, List.copyOf(tensors));
    }

    private static List<TensorDescriptor> descriptors(Path source, SafetensorHeader header) {
        List<TensorDescriptor> result = new ArrayList<>(header.tensorCount());
        for (SafetensorTensorInfo info : header.tensors().values()) {
            List<Long> shape = java.util.Arrays.stream(info.shape()).boxed().toList();
            PayloadReference payload = new PayloadReference(
                    source.toAbsolutePath().normalize().toUri(),
                    source.getFileName().toString(),
                    info.dataBegin(),
                    info.byteLength(),
                    PayloadReference.OffsetBasis.DATA_SECTION);
            result.add(new TensorDescriptor(
                    info.name(),
                    info.dtype().jsonName(),
                    shape,
                    "unknown",
                    null,
                    payload));
        }
        return List.copyOf(result);
    }

    private ModelDocument document(Path identityPath, Map<String, String> fileMetadata,
            List<TensorDescriptor> tensors) throws IOException {
        Path root = Files.isDirectory(identityPath) ? identityPath : identityPath.getParent();
        Map<String, String> metadata = new LinkedHashMap<>();
        fileMetadata.forEach((key, value) -> metadata.put("safetensors." + key, value));
        Map<String, String> extensions = new LinkedHashMap<>();

        String architecture = "unknown";
        Path configPath = root == null ? null : root.resolve(CONFIG_FILE);
        if (configPath != null && Files.isRegularFile(configPath)) {
            String rawConfig = Files.readString(configPath);
            extensions.put("huggingface.config.json", rawConfig);
            JsonNode config = objectMapper.readTree(rawConfig);
            JsonNode modelType = config.get("model_type");
            if (modelType != null && modelType.isTextual() && !modelType.asText().isBlank()) {
                metadata.put("model_type", modelType.asText());
                architecture = modelType.asText();
            }
            JsonNode architectures = config.get("architectures");
            if (architectures != null && architectures.isArray() && !architectures.isEmpty()
                    && architectures.get(0).isTextual() && !architectures.get(0).asText().isBlank()) {
                architecture = architectures.get(0).asText();
                metadata.put("architectures", objectMapper.writeValueAsString(architectures));
            }
        }

        return new ModelDocument(
                modelId(identityPath),
                architecture,
                "safetensors",
                metadata,
                extensions,
                tensors);
    }

    private static String modelId(Path path) {
        Path name = path.getFileName();
        String value = name == null ? "model" : name.toString();
        if (Files.isRegularFile(path)) {
            int extension = value.lastIndexOf('.');
            if (extension > 0) {
                value = value.substring(0, extension);
            }
        }
        return value.isBlank() ? "model" : value;
    }

    private static Map<String, String> mergeFileMetadata(Map<String, SafetensorHeader> headers)
            throws IOException {
        Map<String, String> merged = new LinkedHashMap<>();
        for (Map.Entry<String, SafetensorHeader> shard : headers.entrySet()) {
            for (Map.Entry<String, String> entry : shard.getValue().fileMetadata().entrySet()) {
                String key = "safetensors." + entry.getKey();
                String previous = merged.putIfAbsent(key, entry.getValue());
                if (previous != null && !previous.equals(entry.getValue())) {
                    throw new IOException("Conflicting Safetensors metadata key '"
                            + entry.getKey() + "' across shards");
                }
            }
        }
        return merged;
    }
}
