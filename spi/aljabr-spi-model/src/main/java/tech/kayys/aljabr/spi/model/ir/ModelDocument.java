package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Format-neutral, metadata-first representation of a model checkpoint.
 *
 * <p>Tensor payloads are deliberately not materialized here. A document can
 * describe a checkpoint using metadata and optional payload references, so
 * converters can inspect and plan a conversion without loading model weights
 * into the Java heap. Unknown source metadata belongs in {@code extensions}
 * and must be preserved by converters unless a target explicitly cannot
 * represent it.
 */
public record ModelDocument(
        String id,
        String architecture,
        String sourceFormat,
        Map<String, String> metadata,
        Map<String, String> extensions,
        List<TensorDescriptor> tensors) {

    public ModelDocument {
        id = requireText(id, "id");
        architecture = requireText(architecture, "architecture");
        sourceFormat = requireText(sourceFormat, "sourceFormat");
        metadata = immutableMap(metadata, "metadata");
        extensions = immutableMap(extensions, "extensions");
        tensors = List.copyOf(Objects.requireNonNull(tensors, "tensors must not be null"));

        Set<String> names = new HashSet<>();
        for (TensorDescriptor tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor descriptor must not be null");
            if (!names.add(tensor.name())) {
                throw new IllegalArgumentException("Duplicate tensor name: " + tensor.name());
            }
        }
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    private static Map<String, String> immutableMap(Map<String, String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
