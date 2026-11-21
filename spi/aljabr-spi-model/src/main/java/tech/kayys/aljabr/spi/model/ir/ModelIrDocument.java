package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Versioned AMIR interchange document.
 *
 * <p>The document envelope separates the schema version from the model
 * descriptor and provides a namespaced extension area for information that
 * must survive a read/write cycle even when the current implementation does
 * not interpret it.
 *
 * <p>Extension values are opaque strings. A codec may use them for canonical
 * text, encoded binary fragments, or serialized extension payloads, but must
 * preserve their exact values and keys.
 *
 * @param schemaVersion version of the AMIR document schema
 * @param model canonical model descriptor
 * @param extensions opaque namespaced document-level extensions
 */
public record ModelIrDocument(
        ModelIrSchemaVersion schemaVersion,
        ModelDescriptor model,
        Map<String, String> extensions) {

    public ModelIrDocument {
        Objects.requireNonNull(schemaVersion, "schemaVersion must not be null");
        Objects.requireNonNull(model, "model must not be null");
        extensions = immutableExtensions(extensions);
    }

    /** Creates a document using the current AMIR schema version. */
    public static ModelIrDocument current(ModelDescriptor model) {
        return new ModelIrDocument(ModelIrSchemaVersion.CURRENT, model, Map.of());
    }

    /** Returns a document with a replacement model and unchanged envelope metadata. */
    public ModelIrDocument withModel(ModelDescriptor replacement) {
        return new ModelIrDocument(schemaVersion, replacement, extensions);
    }

    /** Returns a document with one opaque extension added or replaced. */
    public ModelIrDocument withExtension(String key, String value) {
        if (key == null || key.isBlank()) throw new IllegalArgumentException("extension key must not be blank");
        Objects.requireNonNull(value, "extension value must not be null");
        Map<String, String> updated = new LinkedHashMap<>(extensions);
        updated.put(key, value);
        return new ModelIrDocument(schemaVersion, model, updated);
    }

    private static Map<String, String> immutableExtensions(Map<String, String> values) {
        Objects.requireNonNull(values, "extensions must not be null");
        if (values.entrySet().stream().anyMatch(entry ->
                entry.getKey() == null || entry.getKey().isBlank() || entry.getValue() == null)) {
            throw new IllegalArgumentException("extensions must contain non-blank keys and non-null values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
