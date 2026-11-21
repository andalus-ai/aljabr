package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.core.tool.ToolDeclaration;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/** Serializes model-independent tool declarations into FunctionGemma's native syntax. */
public final class FunctionGemmaToolDeclarationRenderer {
    private static final String ESCAPE = FunctionGemmaTokenizer.ESCAPE;
    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_.-]*");

    public String render(ToolDeclaration declaration) {
        Objects.requireNonNull(declaration, "declaration");
        String name = requireIdentifier(declaration.name(), "tool name");
        String description = requireText(declaration.description(), "tool description");
        Map<String, Object> schema = Objects.requireNonNull(
                declaration.parameterSchema(), "parameterSchema");
        if (!"OBJECT".equalsIgnoreCase(String.valueOf(schema.get("type")))) {
            throw new IllegalArgumentException("Tool parameter schema root must have OBJECT type");
        }
        return "declaration:" + name + "{description:" + escaped(description)
                + ",parameters:" + renderValue(schema, true) + "}";
    }

    private String renderValue(Object value, boolean schemaObject) {
        if (value == null) return "None";
        if (value instanceof String string) {
            return escaped(schemaObject && string.equalsIgnoreCase("object") ? "OBJECT"
                    : schemaObject && isTypeKeyword(string) ? string.toUpperCase() : string);
        }
        if (value instanceof Number || value instanceof Boolean) return value.toString();
        if (value instanceof List<?> list) {
            return "[" + String.join(",", list.stream()
                    .map(item -> renderValue(item, false)).toList()) + "]";
        }
        if (value instanceof Map<?, ?> map) {
            StringBuilder result = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!(entry.getKey() instanceof String key) || !IDENTIFIER.matcher(key).matches()) {
                    throw new IllegalArgumentException("Schema keys must be protocol identifiers");
                }
                if (!first) result.append(',');
                first = false;
                result.append(key).append(':');
                Object nested = entry.getValue();
                result.append(renderValue(nested, key.equals("type") || key.equals("properties")
                        || key.equals("items") || key.equals("additionalProperties")));
            }
            return result.append('}').toString();
        }
        throw new IllegalArgumentException("Unsupported schema value: " + value.getClass().getName());
    }

    private static boolean isTypeKeyword(String value) {
        return switch (value.toLowerCase()) {
            case "object", "array", "string", "integer", "number", "boolean", "null" -> true;
            default -> false;
        };
    }

    private static String escaped(String value) {
        if (value.contains(ESCAPE)) {
            throw new IllegalArgumentException("String values must not contain the FunctionGemma escape token");
        }
        return ESCAPE + value + ESCAPE;
    }

    private static String requireIdentifier(String value, String label) {
        Objects.requireNonNull(value, label);
        if (!IDENTIFIER.matcher(value).matches()) {
            throw new IllegalArgumentException(label + " must be a protocol identifier");
        }
        return value;
    }

    private static String requireText(String value, String label) {
        Objects.requireNonNull(value, label);
        if (value.isBlank()) throw new IllegalArgumentException(label + " must not be blank");
        return value;
    }
}
