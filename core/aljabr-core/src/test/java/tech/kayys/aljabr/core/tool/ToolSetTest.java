package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolSetTest {

    @Test
    void preservesOrderAndSupportsNameLookup() {
        ToolSet tools = ToolSet.of(tool("first"), tool("second"));

        assertEquals(List.of("first", "second"),
                tools.declarations().stream().map(ToolDeclaration::name).toList());
        assertTrue(tools.contains("first"));
        assertEquals("second", tools.find("second").name());
        assertNull(tools.find("missing"));
    }

    @Test
    void rejectsDuplicateNames() {
        assertThrows(IllegalArgumentException.class,
                () -> ToolSet.of(tool("lookup"), tool("lookup")));
    }

    @Test
    void snapshotsMutableSchemasAndCollections() {
        Map<String, Object> property = new LinkedHashMap<>();
        property.put("type", "string");
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("value", property);
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        ToolDeclaration mutable = new ToolDeclaration() {
            @Override public String name() { return "snapshot"; }
            @Override public String description() { return "Snapshot test"; }
            @Override public Map<String, Object> parameterSchema() { return schema; }
        };

        ToolSet tools = ToolSet.copyOf(List.of(mutable));
        property.put("type", "integer");
        properties.put("other", Map.of("type", "boolean"));
        schema.put("additionalProperties", false);

        Map<?, ?> copiedProperties = (Map<?, ?>) tools.find("snapshot")
                .parameterSchema().get("properties");
        assertEquals(Map.of("type", "string"), copiedProperties.get("value"));
        assertFalse(copiedProperties.containsKey("other"));
        assertFalse(tools.find("snapshot").parameterSchema().containsKey("additionalProperties"));
        assertThrows(UnsupportedOperationException.class,
                () -> tools.declarations().add(tool("third")));
    }

    @Test
    void supportsNullAsAJsonSchemaLiteral() {
        ToolDeclaration declaration = new ToolDeclaration() {
            @Override public String name() { return "nullable"; }
            @Override public String description() { return "Nullable value"; }
            @Override public Map<String, Object> parameterSchema() {
                return Map.of("type", "object", "properties",
                        Map.of("value", Map.of("type", List.of("string", "null"),
                                "enum", java.util.Arrays.asList("yes", null))));
            }
        };
        ToolSet tools = ToolSet.of(declaration);
        List<?> values = (List<?>) ((Map<?, ?>) ((Map<?, ?>) tools.find("nullable")
                .parameterSchema().get("properties")).get("value")).get("enum");
        assertNull(values.get(1));
    }

    private static ToolDeclaration tool(String name) {
        return new ToolDeclaration() {
            @Override public String name() { return name; }
            @Override public String description() { return "Description for " + name; }
            @Override public Map<String, Object> parameterSchema() {
                return Map.of("type", "object", "properties", Map.of());
            }
        };
    }
}
