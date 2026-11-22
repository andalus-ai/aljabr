package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Reverses the namespaced string metadata used to carry AMIR source attributes. */
final class AmirSourceMetadataCodec {
    private static final String METADATA_PREFIX = "aljabr.source.metadata.";
    private static final String EXTENSION_PREFIX = "aljabr.source.extension.";
    private static final String TENSOR_PREFIX = "aljabr.source.tensor.";

    private AmirSourceMetadataCodec() { }

    static Decoded decode(Map<String, String> stored, List<TensorDescriptor> tensors) {
        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, String> extensions = new LinkedHashMap<>();
        Map<String, Map<String, String>> tensorExtensions = new LinkedHashMap<>();
        Map<String, TensorDescriptor> tensorsByName = new LinkedHashMap<>();
        for (TensorDescriptor tensor : tensors) tensorsByName.put(tensor.name(), tensor);

        stored.forEach((key, value) -> {
            if (key.startsWith(METADATA_PREFIX)) {
                putUnique(metadata, key.substring(METADATA_PREFIX.length()), value, "source metadata");
            } else if (key.startsWith(EXTENSION_PREFIX)) {
                putUnique(extensions, key.substring(EXTENSION_PREFIX.length()), value, "source extension");
            } else if (key.startsWith(TENSOR_PREFIX)) {
                String[] parts = key.substring(TENSOR_PREFIX.length()).split("\\.", -1);
                if (parts.length != 2) throw new IllegalArgumentException("Malformed AMIR tensor extension key: " + key);
                String tensorName = decodePart(parts[0], key);
                String extensionName = decodePart(parts[1], key);
                if (!tensorsByName.containsKey(tensorName)) {
                    throw new IllegalArgumentException("AMIR tensor extension refers to missing tensor: " + tensorName);
                }
                putUnique(tensorExtensions.computeIfAbsent(tensorName, ignored -> new LinkedHashMap<>()),
                        extensionName, value, "tensor extension for " + tensorName);
            } else {
                metadata.put(key, value);
            }
        });

        List<TensorDescriptor> restored = new ArrayList<>(tensors.size());
        for (TensorDescriptor tensor : tensors) {
            Map<String, String> merged = new LinkedHashMap<>(tensor.extensions());
            tensorExtensions.getOrDefault(tensor.name(), Map.of()).forEach((key, value) ->
                    putUnique(merged, key, value, "tensor extension for " + tensor.name()));
            restored.add(new TensorDescriptor(tensor.name(), tensor.dataType(), tensor.shape(),
                    tensor.layout(), tensor.quantization(), tensor.payload(), merged, tensor.tensorType()));
        }
        return new Decoded(Map.copyOf(metadata), Map.copyOf(extensions), List.copyOf(restored));
    }

    private static String decodePart(String encoded, String sourceKey) {
        try {
            String decoded = new String(Base64.getUrlDecoder().decode(encoded), StandardCharsets.UTF_8);
            if (decoded.isBlank()) throw new IllegalArgumentException("Decoded key is blank");
            return decoded;
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Malformed AMIR tensor extension key: " + sourceKey, invalid);
        }
    }

    private static void putUnique(Map<String, String> target, String key, String value, String kind) {
        if (key.isBlank()) throw new IllegalArgumentException(kind + " key must not be blank");
        if (target.putIfAbsent(key, value) != null) {
            throw new IllegalArgumentException("Duplicate " + kind + " key: " + key);
        }
    }

    record Decoded(Map<String, String> metadata, Map<String, String> extensions,
                   List<TensorDescriptor> tensors) { }
}
