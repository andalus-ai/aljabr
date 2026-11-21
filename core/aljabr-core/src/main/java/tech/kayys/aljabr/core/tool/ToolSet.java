package tech.kayys.aljabr.core.tool;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable, ordered collection of tool descriptions.
 *
 * <p>A {@code ToolSet} is a value object for presenting the same declarations
 * to different model adapters. It is not a runtime registry: it performs no
 * authorization, registration, dispatch, or execution.</p>
 */
public final class ToolSet {
    private final List<ToolDeclaration> declarations;
    private final Map<String, ToolDeclaration> byName;

    private ToolSet(Collection<? extends ToolDeclaration> source) {
        Objects.requireNonNull(source, "declarations");
        List<ToolDeclaration> ordered = new ArrayList<>(source.size());
        Map<String, ToolDeclaration> indexed = new LinkedHashMap<>();
        for (ToolDeclaration declaration : source) {
            ToolDeclaration snapshot = snapshot(Objects.requireNonNull(declaration, "declaration"));
            if (indexed.putIfAbsent(snapshot.name(), snapshot) != null) {
                throw new IllegalArgumentException("Duplicate tool declaration: " + snapshot.name());
            }
            ordered.add(snapshot);
        }
        this.declarations = List.copyOf(ordered);
        this.byName = Collections.unmodifiableMap(indexed);
    }

    /** Creates an immutable tool set, preserving declaration order. */
    public static ToolSet of(ToolDeclaration... declarations) {
        Objects.requireNonNull(declarations, "declarations");
        return new ToolSet(List.of(declarations));
    }

    /** Creates an immutable tool set from a collection, preserving iteration order. */
    public static ToolSet copyOf(Collection<? extends ToolDeclaration> declarations) {
        return new ToolSet(declarations);
    }

    public List<ToolDeclaration> declarations() {
        return declarations;
    }

    public int size() {
        return declarations.size();
    }

    public boolean isEmpty() {
        return declarations.isEmpty();
    }

    /** Returns the declaration for a name, or {@code null} when it is absent. */
    public ToolDeclaration find(String name) {
        return byName.get(Objects.requireNonNull(name, "name"));
    }

    /** Returns whether a declaration with this name is present. */
    public boolean contains(String name) {
        return byName.containsKey(Objects.requireNonNull(name, "name"));
    }

    private static ToolDeclaration snapshot(ToolDeclaration source) {
        String name = requireText(source.name(), "tool name");
        String description = requireText(source.description(), "tool description");
        Map<String, Object> schema = freezeMap(source.parameterSchema(), "parameterSchema");
        Object rootType = schema.get("type");
        if (!(rootType instanceof String type) || !type.equalsIgnoreCase("object")) {
            throw new IllegalArgumentException("Tool parameter schema root type must be object");
        }
        return new ImmutableToolDeclaration(name, description, schema);
    }

    private static Map<String, Object> freezeMap(Map<?, ?> source, String path) {
        Objects.requireNonNull(source, path);
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            if (!(entry.getKey() instanceof String key) || key.isBlank()) {
                throw new IllegalArgumentException(path + " keys must be non-blank strings");
            }
            copy.put(key, freeze(entry.getValue(), path + "." + key));
        }
        return Collections.unmodifiableMap(copy);
    }

    private static Object freeze(Object value, String path) {
        if (value == null || value instanceof String || value instanceof Boolean
                || value instanceof Number) return value;
        if (value instanceof Map<?, ?> map) return freezeMap(map, path);
        if (value instanceof Collection<?> collection) {
            List<Object> copy = new ArrayList<>(collection.size());
            int index = 0;
            for (Object item : collection) copy.add(freeze(item, path + "[" + index++ + "]"));
            return Collections.unmodifiableList(copy);
        }
        throw new IllegalArgumentException(path + " contains unsupported schema value: "
                + value.getClass().getName());
    }

    private static String requireText(String value, String label) {
        Objects.requireNonNull(value, label);
        if (value.isBlank()) throw new IllegalArgumentException(label + " must not be blank");
        return value;
    }

    private record ImmutableToolDeclaration(String name, String description,
            Map<String, Object> parameterSchema) implements ToolDeclaration { }
}
