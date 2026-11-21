package tech.kayys.aljabr.core.tool;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Validates untrusted tool arguments against the JSON-Schema-compatible subset
 * used by {@link ToolDeclaration}.
 *
 * <p>Supported keywords: type, properties, required, additionalProperties,
 * items, enum, const, minimum/maximum, minLength/maxLength, minItems/maxItems,
 * and pattern. This class validates data only; it never authorizes or executes
 * a tool.</p>
 */
public final class ToolSchemaValidator {
    private static final int MAX_DEPTH = 64;

    /** Validates both the call's tool name and its arguments against a declaration. */
    public ValidationResult validate(ToolCall call, ToolDeclaration declaration) {
        Objects.requireNonNull(call, "call");
        Objects.requireNonNull(declaration, "declaration");
        List<ValidationError> errors = new ArrayList<>();
        if (!call.name().equals(declaration.name())) {
            errors.add(new ValidationError("$tool", "expected tool '" + declaration.name()
                    + "' but call names '" + call.name() + "'"));
        }
        validate(call.arguments(), declaration.parameterSchema(), errors);
        return new ValidationResult(errors);
    }

    /** Validates an argument object against a parameter schema. */
    public ValidationResult validate(Map<String, ?> arguments, Map<String, ?> schema) {
        Objects.requireNonNull(arguments, "arguments");
        Objects.requireNonNull(schema, "schema");
        List<ValidationError> errors = new ArrayList<>();
        validate(arguments, schema, "$", errors, 0);
        return new ValidationResult(errors);
    }

    private void validate(Object value, Map<String, ?> schema, String path,
            List<ValidationError> errors, int depth) {
        if (depth > MAX_DEPTH) {
            errors.add(error(path, "maximum validation depth exceeded"));
            return;
        }
        if (schema.containsKey("type") && !matchesType(value, schema.get("type"))) {
            errors.add(error(path, "expected " + typeName(schema.get("type"))
                    + " but was " + actualType(value)));
            return;
        }
        if (schema.containsKey("enum")) {
            List<?> allowed = asList(schema.get("enum"), path + ".enum");
            if (allowed.stream().noneMatch(candidate -> jsonEquals(candidate, value))) {
                errors.add(error(path, "value is not one of the allowed enum values"));
            }
        }
        if (schema.containsKey("const") && !jsonEquals(schema.get("const"), value)) {
            errors.add(error(path, "value does not match the required constant"));
        }
        if (value instanceof Map<?, ?> object) validateObject(object, schema, path, errors, depth);
        else if (value instanceof Collection<?> array) validateArray(array, schema, path, errors, depth);
        else if (value instanceof String string) validateString(string, schema, path, errors);
        else if (value instanceof Number number) validateNumber(number, schema, path, errors);
    }

    private void validateObject(Map<?, ?> object, Map<String, ?> schema, String path,
            List<ValidationError> errors, int depth) {
        Map<String, ?> properties = schema.containsKey("properties")
                ? asMap(schema.get("properties"), path + ".properties") : Map.of();
        List<?> required = schema.containsKey("required")
                ? asList(schema.get("required"), path + ".required") : List.of();
        for (Object item : required) {
            if (!(item instanceof String name)) throw invalidSchema(path + ".required entries must be strings");
            if (!object.containsKey(name)) errors.add(error(child(path, name), "required property is missing"));
        }
        Object additional = schema.get("additionalProperties");
        if (schema.containsKey("additionalProperties")
                && !(additional instanceof Boolean) && !(additional instanceof Map<?, ?>)) {
            throw invalidSchema(path + ".additionalProperties must be a boolean or schema object");
        }
        for (Map.Entry<?, ?> entry : object.entrySet()) {
            if (!(entry.getKey() instanceof String name)) {
                errors.add(error(path, "object keys must be strings"));
            } else if (properties.containsKey(name)) {
                validate(entry.getValue(), asMap(properties.get(name), child(path, name)),
                        child(path, name), errors, depth + 1);
            } else if (Boolean.FALSE.equals(additional)) {
                errors.add(error(child(path, name), "unexpected property"));
            } else if (additional instanceof Map<?, ?> additionalSchema) {
                validate(entry.getValue(), castMap(additionalSchema, path + ".additionalProperties"),
                        child(path, name), errors, depth + 1);
            }
        }
    }

    private void validateArray(Collection<?> array, Map<String, ?> schema, String path,
            List<ValidationError> errors, int depth) {
        checkSize(array.size(), schema, "minItems", "maxItems", path, errors);
        if (schema.containsKey("items")) {
            Map<String, ?> itemSchema = asMap(schema.get("items"), path + ".items");
            int index = 0;
            for (Object item : array) validate(item, itemSchema, path + "[" + index++ + "]", errors, depth + 1);
        }
    }

    private void validateString(String value, Map<String, ?> schema, String path,
            List<ValidationError> errors) {
        checkSize(value.codePointCount(0, value.length()), schema, "minLength", "maxLength", path, errors);
        Object pattern = schema.get("pattern");
        if (pattern != null) {
            if (!(pattern instanceof String expression)) throw invalidSchema(path + ".pattern must be a string");
            try {
                if (!Pattern.compile(expression).matcher(value).find()) {
                    errors.add(error(path, "string does not match required pattern"));
                }
            } catch (PatternSyntaxException exception) {
                throw invalidSchema(path + ".pattern is not a valid regular expression");
            }
        }
    }

    private void validateNumber(Number value, Map<String, ?> schema, String path,
            List<ValidationError> errors) {
        BigDecimal number = decimal(value, path);
        checkBound(number, schema, "minimum", true, path, errors);
        checkBound(number, schema, "maximum", false, path, errors);
    }

    private void checkBound(BigDecimal value, Map<String, ?> schema, String key,
            boolean minimum, String path, List<ValidationError> errors) {
        if (!schema.containsKey(key)) return;
        BigDecimal bound = decimal(asNumber(schema.get(key), path + "." + key), path + "." + key);
        int comparison = value.compareTo(bound);
        if ((minimum && comparison < 0) || (!minimum && comparison > 0)) {
            errors.add(error(path, (minimum ? "must be >= " : "must be <= ") + bound));
        }
    }

    private void checkSize(int size, Map<String, ?> schema, String minKey, String maxKey,
            String path, List<ValidationError> errors) {
        if (schema.containsKey(minKey)) {
            int min = asInteger(schema.get(minKey), path + "." + minKey);
            if (size < min) errors.add(error(path, "size must be at least " + min));
        }
        if (schema.containsKey(maxKey)) {
            int max = asInteger(schema.get(maxKey), path + "." + maxKey);
            if (size > max) errors.add(error(path, "size must be at most " + max));
        }
    }

    private boolean matchesType(Object value, Object declaredType) {
        if (declaredType instanceof List<?> alternatives) {
            if (alternatives.isEmpty()) throw invalidSchema("type array must not be empty");
            return alternatives.stream().anyMatch(candidate -> matchesType(value, candidate));
        }
        if (!(declaredType instanceof String name)) {
            throw invalidSchema("schema type must be a string or non-empty array of strings");
        }
        return switch (name.toUpperCase()) {
            case "OBJECT" -> value instanceof Map<?, ?>;
            case "ARRAY" -> value instanceof Collection<?>;
            case "STRING" -> value instanceof String;
            case "BOOLEAN" -> value instanceof Boolean;
            case "INTEGER" -> isInteger(value);
            case "NUMBER" -> value instanceof Number;
            case "NULL" -> value == null;
            default -> throw invalidSchema("unsupported schema type: " + name);
        };
    }

    private static boolean isInteger(Object value) {
        if (!(value instanceof Number number)) return false;
        try { new BigDecimal(number.toString()).toBigIntegerExact(); return true; }
        catch (NumberFormatException | ArithmeticException exception) { return false; }
    }

    private static String typeName(Object type) {
        return type instanceof List<?> list ? list.toString() : String.valueOf(type).toUpperCase();
    }

    private static String actualType(Object value) {
        if (value == null) return "NULL";
        if (value instanceof Map<?, ?>) return "OBJECT";
        if (value instanceof Collection<?>) return "ARRAY";
        if (value instanceof String) return "STRING";
        if (value instanceof Boolean) return "BOOLEAN";
        if (value instanceof Number) return "NUMBER";
        return value.getClass().getSimpleName();
    }

    private static Map<String, ?> asMap(Object value, String path) {
        if (!(value instanceof Map<?, ?> map)) throw invalidSchema(path + " must be an object");
        return castMap(map, path);
    }

    private static Map<String, ?> castMap(Map<?, ?> map, String path) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String key)) throw invalidSchema(path + " keys must be strings");
            result.put(key, entry.getValue());
        }
        return result;
    }

    private static List<?> asList(Object value, String path) {
        if (!(value instanceof List<?> list)) throw invalidSchema(path + " must be an array");
        return list;
    }

    private static Number asNumber(Object value, String path) {
        if (!(value instanceof Number number)) throw invalidSchema(path + " must be numeric");
        return number;
    }

    private static int asInteger(Object value, String path) {
        try { return decimal(asNumber(value, path), path).intValueExact(); }
        catch (ArithmeticException exception) { throw invalidSchema(path + " must be an integer"); }
    }

    private static BigDecimal decimal(Number value, String path) {
        try { return value instanceof BigDecimal decimal ? decimal : new BigDecimal(value.toString()); }
        catch (NumberFormatException exception) { throw invalidSchema(path + " must be a finite number"); }
    }

    private static boolean jsonEquals(Object left, Object right) {
        if (left instanceof Number a && right instanceof Number b) {
            try { return new BigDecimal(a.toString()).compareTo(new BigDecimal(b.toString())) == 0; }
            catch (NumberFormatException exception) { return false; }
        }
        return Objects.equals(left, right);
    }

    private static String child(String path, String name) { return path + "." + name; }
    private static ValidationError error(String path, String message) { return new ValidationError(path, message); }
    private static IllegalArgumentException invalidSchema(String message) {
        return new IllegalArgumentException("Invalid tool schema: " + message);
    }

    /** Immutable validation result. */
    public record ValidationResult(List<ValidationError> errors) {
        public ValidationResult { errors = List.copyOf(Objects.requireNonNull(errors, "errors")); }
        public boolean valid() { return errors.isEmpty(); }
    }

    /** One path-addressed validation diagnostic. */
    public record ValidationError(String path, String message) {
        public ValidationError {
            Objects.requireNonNull(path, "path");
            Objects.requireNonNull(message, "message");
        }
        @Override public String toString() { return path + ": " + message; }
    }
}
