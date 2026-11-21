package tech.kayys.aljabr.core.tool;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable, model-independent tool call extracted from model output.
 *
 * <p>The call is untrusted data. It does not resolve, authorize, dispatch, or
 * execute the named tool.</p>
 */
public record ToolCall(String name, Map<String, Object> arguments) {
    public ToolCall {
        Objects.requireNonNull(name, "name");
        if (name.isBlank()) throw new IllegalArgumentException("name must not be blank");
        arguments = freezeMap(Objects.requireNonNull(arguments, "arguments"), "arguments");
    }

    private static Map<String, Object> freezeMap(Map<?, ?> source, String path) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new IllegalArgumentException(path + " keys must be strings");
            }
            copy.put(key, freeze(entry.getValue(), path + "." + key));
        }
        return Collections.unmodifiableMap(copy);
    }

    private static Object freeze(Object value, String path) {
        if (value == null || value instanceof String || value instanceof Boolean || value instanceof Number) {
            return value;
        }
        if (value instanceof Map<?, ?> map) return freezeMap(map, path);
        if (value instanceof Collection<?> collection) {
            List<Object> copy = new ArrayList<>(collection.size());
            int index = 0;
            for (Object item : collection) copy.add(freeze(item, path + "[" + index++ + "]"));
            return Collections.unmodifiableList(copy);
        }
        throw new IllegalArgumentException(path + " contains unsupported argument value: "
                + value.getClass().getName());
    }
}
