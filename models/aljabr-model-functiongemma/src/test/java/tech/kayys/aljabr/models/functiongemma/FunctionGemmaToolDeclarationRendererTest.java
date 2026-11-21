package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tool.ToolDeclaration;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaToolDeclarationRendererTest {

    @Test
    void rendersADeclarationThatDoesNotDependOnFunctionGemmaTypes() {
        Map<String, Object> location = new LinkedHashMap<>();
        location.put("type", "string");
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("location", location);
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        schema.put("required", List.of("location"));
        schema.put("additionalProperties", false);

        ToolDeclaration declaration = new ToolDeclaration() {
            @Override public String name() { return "get_weather"; }
            @Override public String description() { return "Get current weather"; }
            @Override public Map<String, Object> parameterSchema() { return schema; }
        };

        String rendered = new FunctionGemmaToolDeclarationRenderer().render(declaration);

        assertEquals("declaration:get_weather{description:<escape>Get current weather<escape>,"
                + "parameters:{type:<escape>OBJECT<escape>,properties:{location:{type:<escape>STRING<escape>}},"
                + "required:[<escape>location<escape>],additionalProperties:false}}", rendered);
    }

    @Test
    void templateAcceptsAnyImplementationOfTheSharedContract() {
        ToolDeclaration declaration = new ToolDeclaration() {
            @Override public String name() { return "lookup"; }
            @Override public String description() { return "Find a record"; }
            @Override public Map<String, Object> parameterSchema() {
                return Map.of("type", "object", "properties",
                        Map.of("id", Map.of("type", "integer")));
            }
        };

        String content = new FunctionGemmaChatTemplate().formatDeveloperContent("Use tools.", declaration);

        assertTrue(content.contains("declaration:lookup"));
        assertTrue(content.contains("type:<escape>INTEGER<escape>"));
    }

    @Test
    void rejectsInvalidGenericDeclarationBeforeRendering() {
        ToolDeclaration invalid = new ToolDeclaration() {
            @Override public String name() { return "bad name"; }
            @Override public String description() { return "Bad"; }
            @Override public Map<String, Object> parameterSchema() { return Map.of("type", "object"); }
        };
        assertThrows(IllegalArgumentException.class,
                () -> new FunctionGemmaToolDeclarationRenderer().render(invalid));
    }
    @Test
    void formatsSharedToolSetWithoutModelSpecificDeclarationTypes() {
        ToolDeclaration declaration = new ToolDeclaration() {
            @Override public String name() { return "search"; }
            @Override public String description() { return "Search records"; }
            @Override public Map<String, Object> parameterSchema() {
                return Map.of("type", "object", "properties", Map.of());
            }
        };
        tech.kayys.aljabr.core.tool.ToolSet tools =
                tech.kayys.aljabr.core.tool.ToolSet.of(declaration);

        String content = new FunctionGemmaChatTemplate().formatDeveloperContent("Search when needed.", tools);

        assertTrue(content.contains("declaration:search"));
        assertTrue(content.contains("Search records"));
    }

}
