package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;

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
    void reportsMissingUnexpectedAndWronglyTypedArguments() {
        Map<String, Object> schema = Map.of(
                "type", "OBJECT",
                "required", List.of("query"),
                "additionalProperties", false,
                "properties", Map.of("query", Map.of("type", "STRING")));
        var result = validator.validate(Map.of("extra", true, "query", 42), schema);
        assertFalse(result.valid());
        assertEquals(3, result.errors().size());
        assertTrue(result.errors().stream().anyMatch(e -> e.path().equals("$.query")));
        assertTrue(result.errors().stream().anyMatch(e -> e.path().equals("$.extra")));
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
    void acceptsParsedFunctionCallDirectly() {
        var call = new FunctionGemmaToolCallParser.FunctionCall("lookup", Map.of("id", 7L));
        var schema = Map.<String, Object>of("type", "OBJECT", "required", List.of("id"),
                "properties", Map.of("id", Map.of("type", "INTEGER", "const", 7)));
        assertTrue(validator.validate(call, schema).valid());
    }

    @Test
    void comparesDecimalBoundsWithoutFloatingPointRounding() {
        var schema = Map.<String, Object>of("type", "NUMBER", "maximum", new BigDecimal("0.3"));
        assertFalse(validator.validate(Map.of("value", 0.31), Map.of(
                "type", "OBJECT", "properties", Map.of("value", schema))).valid());
    }
}
