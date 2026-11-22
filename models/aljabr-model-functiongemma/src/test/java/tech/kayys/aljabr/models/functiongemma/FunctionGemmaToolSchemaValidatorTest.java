package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tool.ToolCall;
import tech.kayys.aljabr.core.tool.ToolDeclaration;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaToolSchemaValidatorTest {
    private final FunctionGemmaToolSchemaValidator validator = new FunctionGemmaToolSchemaValidator();

    @Test
    void validatesRequiredPropertiesAndNestedTypes() {
        Map<String, Object> schema = Map.of(
                "type", "OBJECT",
                "required", List.of("location", "units"),
                "additionalProperties", false,
                "properties", Map.of(
                        "location", Map.of("type", "STRING", "minLength", 2),
                        "units", Map.of("type", "STRING", "enum", List.of("celsius", "fahrenheit")),
                        "options", Map.of("type", "OBJECT", "properties", Map.of(
                                "days", Map.of("type", "INTEGER", "minimum", 1, "maximum", 7)))));
        var result = validator.validate(Map.of(
                "location", "Bandung",
                "units", "celsius",
                "options", Map.of("days", 3)), schema);
        assertTrue(result.valid(), result.errors().toString());
    }

    @Test
    void reportsUnexpectedAndWronglyTypedArguments() {
        Map<String, Object> schema = Map.of(
                "type", "OBJECT",
                "required", List.of("query"),
                "additionalProperties", false,
                "properties", Map.of("query", Map.of("type", "STRING")));
        var result = validator.validate(Map.of("extra", true, "query", 42), schema);
        assertFalse(result.valid());
        assertEquals(2, result.errors().size());
        assertTrue(result.errors().stream().anyMatch(e -> e.path().equals("$.query")));
        assertTrue(result.errors().stream().anyMatch(e -> e.path().equals("$.extra")));
    }

    @Test
    void reportsMissingRequiredArguments() {
        var schema = Map.<String, Object>of("type", "OBJECT", "required", List.of("query"));
        var result = validator.validate(Map.of(), schema);
        assertEquals("$.query: required property is missing", result.errors().getFirst().toString());
    }

    @Test
    void validatesArrayItemsAndNumericBounds() {
        Map<String, Object> schema = Map.of(
                "type", "OBJECT",
                "properties", Map.of("scores", Map.of(
                        "type", "ARRAY", "minItems", 2,
                        "items", Map.of("type", "NUMBER", "minimum", 0, "maximum", 1))));
        var result = validator.validate(Map.of("scores", List.of(0.5, 1.2)), schema);
        assertFalse(result.valid());
        assertTrue(result.errors().stream().anyMatch(e -> e.path().equals("$.scores[1]")));
    }

    @Test
    void validatesPatternAndEnumAndAllowsNullWhenDeclared() {
        Map<String, Object> schema = Map.of(
                "type", "OBJECT",
                "properties", Map.of(
                        "code", Map.of("type", "STRING", "pattern", "^[A-Z]{2}-[0-9]+$"),
                        "optional", Map.of("type", List.of("STRING", "NULL"))));
        assertTrue(validator.validate(Map.of("code", "ID-42"), schema).valid());
        assertFalse(validator.validate(Map.of("code", "bad", "optional", "x"), schema).valid());
    }

    @Test
    void acceptsIntegralDecimalAndNullForDeclaredNullableType() {
        var integerSchema = Map.<String, Object>of("type", "INTEGER");
        assertTrue(validator.validate(Map.of("count", new BigDecimal("3.0")),
                Map.of("type", "OBJECT", "properties", Map.of("count", integerSchema))).valid());

        Map<String, Object> arguments = new java.util.LinkedHashMap<>();
        arguments.put("optional", null);
        assertTrue(validator.validate(arguments, Map.of("type", "OBJECT", "properties",
                Map.of("optional", Map.of("type", List.of("STRING", "NULL"))))).valid());
    }

    @Test
    void acceptsParsedFunctionCallThroughCompatibilityFacade() {
        var call = new FunctionGemmaToolCallParser.FunctionCall("lookup", Map.of("id", 7L));
        var schema = Map.<String, Object>of("type", "OBJECT", "required", List.of("id"),
                "properties", Map.of("id", Map.of("type", "INTEGER", "const", 7)));
        assertTrue(validator.validate(call, schema).valid());
    }

    @Test
    void delegatesCoreToolCallAndDeclarationValidation() {
        ToolDeclaration declaration = new ToolDeclaration() {
            @Override public String name() { return "lookup"; }
            @Override public String description() { return "Look up a record"; }
            @Override public Map<String, Object> parameterSchema() {
                return Map.of("type", "OBJECT", "required", List.of("id"),
                        "properties", Map.of("id", Map.of("type", "INTEGER")));
            }
        };
        assertTrue(validator.validate(new ToolCall("lookup", Map.of("id", 7)), declaration).valid());
        var mismatch = validator.validate(new ToolCall("delete", Map.of("id", 7)), declaration);
        assertFalse(mismatch.valid());
        assertTrue(mismatch.errors().stream().anyMatch(error -> error.path().equals("$tool")));
    }

    @Test
    void comparesDecimalBoundsWithoutFloatingPointRounding() {
        var schema = Map.<String, Object>of("type", "NUMBER", "maximum", new BigDecimal("0.3"));
        assertFalse(validator.validate(Map.of("value", 0.31), Map.of(
                "type", "OBJECT", "properties", Map.of("value", schema))).valid());
    }
}
