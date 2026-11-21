package tech.kayys.aljabr.models.functiongemma;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Immutable, typed Java representation of a FunctionGemma tool declaration.
 *
 * <p>The declaration renders the native FunctionGemma developer-turn syntax.
 * It describes a tool only: it does not register, authorize, or execute it.</p>
 */
public final class FunctionGemmaToolDeclaration {
    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_.-]*");
    private static final String ESCAPE = FunctionGemmaTokenizer.ESCAPE;

    private final String name;
    private final String description;
    private final Schema parameters;

    private FunctionGemmaToolDeclaration(String name, String description, Schema parameters) {
        this.name = requireIdentifier(name, "tool name");
        this.description = requireText(description, "tool description");
        this.parameters = Objects.requireNonNull(parameters, "parameters");
        if (parameters.type != Type.OBJECT) {
            throw new IllegalArgumentException("Tool parameters must have OBJECT type");
        }
    }

    public static FunctionGemmaToolDeclaration of(String name, String description, Schema parameters) {
        return new FunctionGemmaToolDeclaration(name, description, parameters);
    }

    public String name() { return name; }
    public String description() { return description; }

    /** Returns a JSON-Schema-like immutable map suitable for the argument validator. */
    public Map<String, Object> parameterSchema() {
        return parameters.toMap();
    }

    /** Renders one native FunctionGemma declaration without surrounding control tokens. */
    public String render() {
        return "declaration:" + name + "{description:" + escaped(description)
                + ",parameters:" + renderSchema(parameters) + "}";
    }

    private static String renderSchema(Schema schema) {
        List<String> fields = new ArrayList<>();
        fields.add("type:" + escaped(schema.type.name()));
        if (schema.description != null) fields.add("description:" + escaped(schema.description));
        if (!schema.properties.isEmpty()) {
            List<String> properties = new ArrayList<>();
            schema.properties.forEach((key, value) -> properties.add(key + ":" + renderSchema(value)));
            fields.add("properties:{" + String.join(",", properties) + "}");
        }
        if (!schema.required.isEmpty()) {
            fields.add("required:[" + schema.required.stream().map(FunctionGemmaToolDeclaration::escaped)
                    .reduce((a, b) -> a + "," + b).orElse("") + "]");
        }
        if (schema.items != null) fields.add("items:" + renderSchema(schema.items));
        if (schema.additionalProperties != null) {
            fields.add("additionalProperties:" + schema.additionalProperties);
        }
        if (!schema.enumValues.isEmpty()) {
            fields.add("enum:[" + schema.enumValues.stream().map(FunctionGemmaToolDeclaration::literal)
                    .reduce((a, b) -> a + "," + b).orElse("") + "]");
        }
        if (schema.hasConstant) fields.add("const:" + literal(schema.constant));
        if (schema.minimum != null) fields.add("minimum:" + schema.minimum.toPlainString());
        if (schema.maximum != null) fields.add("maximum:" + schema.maximum.toPlainString());
        if (schema.minLength != null) fields.add("minLength:" + schema.minLength);
        if (schema.maxLength != null) fields.add("maxLength:" + schema.maxLength);
        if (schema.minItems != null) fields.add("minItems:" + schema.minItems);
        if (schema.maxItems != null) fields.add("maxItems:" + schema.maxItems);
        if (schema.pattern != null) fields.add("pattern:" + escaped(schema.pattern));
        return "{" + String.join(",", fields) + "}";
    }

    private static String literal(Object value) {
        if (value == null) return "None";
        if (value instanceof String string) return escaped(string);
        if (value instanceof Boolean || value instanceof Number) return value.toString();
        throw new IllegalArgumentException("Schema literals must be strings, numbers, booleans, or null");
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
            throw new IllegalArgumentException(label + " must be a protocol identifier: " + value);
        }
        return value;
    }

    private static String requireText(String value, String label) {
        Objects.requireNonNull(value, label);
        if (value.isBlank()) throw new IllegalArgumentException(label + " must not be blank");
        return value;
    }

    public enum Type { OBJECT, ARRAY, STRING, INTEGER, NUMBER, BOOLEAN, NULL }

    /** Fluent schema builder for tool parameters and nested properties. */
    public static final class Schema {
        private final Type type;
        private String description;
        private final Map<String, Schema> properties = new LinkedHashMap<>();
        private final List<String> required = new ArrayList<>();
        private Schema items;
        private Boolean additionalProperties;
        private final List<Object> enumValues = new ArrayList<>();
        private Object constant;
        private boolean hasConstant;
        private BigDecimal minimum;
        private BigDecimal maximum;
        private Integer minLength;
        private Integer maxLength;
        private Integer minItems;
        private Integer maxItems;
        private String pattern;

        private Schema(Type type) { this.type = Objects.requireNonNull(type, "type"); }

        public static Schema object() { return new Schema(Type.OBJECT); }
        public static Schema array() { return new Schema(Type.ARRAY); }
        public static Schema string() { return new Schema(Type.STRING); }
        public static Schema integer() { return new Schema(Type.INTEGER); }
        public static Schema number() { return new Schema(Type.NUMBER); }
        public static Schema bool() { return new Schema(Type.BOOLEAN); }
        public static Schema nil() { return new Schema(Type.NULL); }

        public Schema description(String value) {
            this.description = requireText(value, "schema description");
            return this;
        }

        public Schema property(String name, Schema schema) {
            if (type != Type.OBJECT) throw new IllegalStateException("properties are only valid on OBJECT schemas");
            String key = requireIdentifier(name, "property name");
            if (properties.putIfAbsent(key, Objects.requireNonNull(schema, "schema")) != null) {
                throw new IllegalArgumentException("Duplicate property: " + key);
            }
            return this;
        }

        public Schema required(String... names) {
            if (type != Type.OBJECT) throw new IllegalStateException("required is only valid on OBJECT schemas");
            for (String name : names) {
                String key = requireIdentifier(name, "required property");
                if (!required.contains(key)) required.add(key);
            }
            return this;
        }

        public Schema items(Schema schema) {
            if (type != Type.ARRAY) throw new IllegalStateException("items is only valid on ARRAY schemas");
            this.items = Objects.requireNonNull(schema, "schema");
            return this;
        }

        public Schema additionalProperties(boolean allowed) {
            if (type != Type.OBJECT) throw new IllegalStateException("additionalProperties is only valid on OBJECT schemas");
            this.additionalProperties = allowed;
            return this;
        }

        public Schema enumValues(Object... values) {
            if (values.length == 0) throw new IllegalArgumentException("enum must contain at least one value");
            enumValues.clear();
            for (Object value : values) {
                validateLiteral(value);
                enumValues.add(value);
            }
            return this;
        }

        public Schema constant(Object value) {
            validateLiteral(value);
            this.constant = value;
            this.hasConstant = true;
            return this;
        }

        public Schema minimum(Number value) {
            this.minimum = decimal(value, "minimum");
            return this;
        }

        public Schema maximum(Number value) {
            this.maximum = decimal(value, "maximum");
            return this;
        }

        public Schema minLength(int value) { this.minLength = nonNegative(value, "minLength"); return this; }
        public Schema maxLength(int value) { this.maxLength = nonNegative(value, "maxLength"); return this; }
        public Schema minItems(int value) { this.minItems = nonNegative(value, "minItems"); return this; }
        public Schema maxItems(int value) { this.maxItems = nonNegative(value, "maxItems"); return this; }

        public Schema pattern(String value) {
            Objects.requireNonNull(value, "pattern");
            try {
                Pattern.compile(value);
            } catch (java.util.regex.PatternSyntaxException exception) {
                throw new IllegalArgumentException("Invalid schema pattern", exception);
            }
            this.pattern = value;
            return this;
        }

        private Map<String, Object> toMap() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("type", type.name());
            if (description != null) result.put("description", description);
            if (!properties.isEmpty()) {
                Map<String, Object> nested = new LinkedHashMap<>();
                properties.forEach((key, value) -> nested.put(key, value.toMap()));
                result.put("properties", Collections.unmodifiableMap(nested));
            }
            if (!required.isEmpty()) result.put("required", List.copyOf(required));
            if (items != null) result.put("items", items.toMap());
            if (additionalProperties != null) result.put("additionalProperties", additionalProperties);
            if (!enumValues.isEmpty()) result.put("enum", List.copyOf(enumValues));
            if (hasConstant) result.put("const", constant);
            if (minimum != null) result.put("minimum", minimum);
            if (maximum != null) result.put("maximum", maximum);
            if (minLength != null) result.put("minLength", minLength);
            if (maxLength != null) result.put("maxLength", maxLength);
            if (minItems != null) result.put("minItems", minItems);
            if (maxItems != null) result.put("maxItems", maxItems);
            if (pattern != null) result.put("pattern", pattern);
            return Collections.unmodifiableMap(result);
        }

        private static void validateLiteral(Object value) {
            if (value != null && !(value instanceof String) && !(value instanceof Number)
                    && !(value instanceof Boolean)) {
                throw new IllegalArgumentException("Schema literals must be strings, numbers, booleans, or null");
            }
            if (value instanceof String string) escaped(string);
            if (value instanceof Number number) decimal(number, "numeric schema literal");
        }

        private static BigDecimal decimal(Number value, String label) {
            Objects.requireNonNull(value, label);
            try {
                return new BigDecimal(value.toString());
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException(label + " must be finite", exception);
            }
        }

        private static int nonNegative(int value, String label) {
            if (value < 0) throw new IllegalArgumentException(label + " must be non-negative");
            return value;
        }
    }
}
