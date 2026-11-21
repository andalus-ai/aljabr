package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolSchemaValidatorTest {
    private final ToolSchemaValidator validator = new ToolSchemaValidator();

    @Test
    void validatesCallNameAndNestedArguments() {
        ToolDeclaration declaration = declaration("lookup", Map.of(
                "type", "object",
                "properties", Map.of(
                        "query", Map.of("type", "string", "minLength", 2),
                        "limit", Map.of("type", "integer", "minimum", 1, "maximum", 10),
                        "tags", Map.of("type", "array", "items", Map.of("type", "string"))),
                "required", List.of("query"),
                "additionalProperties", false));

        assertTrue(validator.validate(new ToolCall("lookup",
                Map.of("query", "hello", "limit", 5, "tags", List.of("a"))), declaration).valid());
        var invalid = validator.validate(new ToolCall("other",
                Map.of("query", "x", "limit", 11, "extra", true)), declaration);
        assertFalse(invalid.valid());
        assertTrue(invalid.errors().stream().anyMatch(e -> e.path().equals("$tool")));
        assertTrue(invalid.errors().stream().anyMatch(e -> e.path().equals("$.query")));
        assertTrue(invalid.errors().stream().anyMatch(e -> e.path().equals("$.limit")));
        assertTrue(invalid.errors().stream().anyMatch(e -> e.path().equals("$.extra")));
    }

    @Test
    void validatesAdditionalPropertiesWithSchemaAndNumericEnumEquality() {
        var schema = Map.<String, Object>of("type", "object",
                "additionalProperties", Map.of("type", "integer", "enum", List.of(7L)));
        assertTrue(validator.validate(Map.of("count", 7), schema).valid());
        assertFalse(validator.validate(Map.of("count", "7"), schema).valid());
    }

    @Test
    void reportsInvalidSchemaDefinitions() {
        assertThrows(IllegalArgumentException.class,
                () -> validator.validate(Map.of("value", 1), Map.of("type", "mystery")));
    }

    private static ToolDeclaration declaration(String name, Map<String, Object> schema) {
        return new ToolDeclaration() {
            @Override public String name() { return name; }
            @Override public String description() { return "Test declaration"; }
            @Override public Map<String, Object> parameterSchema() { return schema; }
        };
    }
}
