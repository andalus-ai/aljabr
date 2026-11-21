package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaToolDeclarationTest {
    @Test
    void rendersTypedDeclarationUsingNativeEscapeTokens() {
        var parameters = FunctionGemmaToolDeclaration.Schema.object()
                .property("location", FunctionGemmaToolDeclaration.Schema.string()
                        .description("City or place").minLength(2))
                .property("units", FunctionGemmaToolDeclaration.Schema.string()
                        .enumValues("celsius", "fahrenheit"))
                .required("location", "units")
                .additionalProperties(false);
        var tool = FunctionGemmaToolDeclaration.of("get_weather", "Get current weather", parameters);

        assertEquals("declaration:get_weather{description:<escape>Get current weather<escape>,parameters:"
                + "{type:<escape>OBJECT<escape>,properties:{location:{type:<escape>STRING<escape>,"
                + "description:<escape>City or place<escape>,minLength:2},units:{type:<escape>STRING"
                + "<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]}},"
                + "required:[<escape>location<escape>,<escape>units<escape>],additionalProperties:false}}",
                tool.render());
        assertEquals(List.of("location", "units"), tool.parameterSchema().get("required"));
    }

    @Test
    void formatsTypedToolsInsideDeveloperContent() {
        var tool = FunctionGemmaToolDeclaration.of("lookup", "Find a record",
                FunctionGemmaToolDeclaration.Schema.object()
                        .property("id", FunctionGemmaToolDeclaration.Schema.integer())
                        .required("id"));
        var content = new FunctionGemmaChatTemplate().formatDeveloperContent("Use tools.", tool);
        assertTrue(content.startsWith("Use tools.\\n" + FunctionGemmaChatTemplate.START_FUNCTION_DECLARATION));
        assertTrue(content.contains("declaration:lookup"));
        assertTrue(content.endsWith(FunctionGemmaChatTemplate.END_FUNCTION_DECLARATION));
    }

    @Test
    void snapshotsBuilderSoLaterMutationDoesNotChangeDeclaration() {
        var builder = FunctionGemmaToolDeclaration.Schema.object()
                .property("query", FunctionGemmaToolDeclaration.Schema.string());
        var tool = FunctionGemmaToolDeclaration.of("search", "Search", builder);
        builder.property("late", FunctionGemmaToolDeclaration.Schema.bool());
        assertFalse(tool.parameterSchema().get("properties").toString().contains("late"));
    }

    @Test
    void supportsNestedObjectAndArraySchemasAndValidatorInterop() {
        var parameters = FunctionGemmaToolDeclaration.Schema.object()
                .property("options", FunctionGemmaToolDeclaration.Schema.object()
                        .property("days", FunctionGemmaToolDeclaration.Schema.integer().minimum(1).maximum(7))
                        .required("days"))
                .property("tags", FunctionGemmaToolDeclaration.Schema.array()
                        .items(FunctionGemmaToolDeclaration.Schema.string().minLength(1))
                        .minItems(1));
        var tool = FunctionGemmaToolDeclaration.of("search", "Search records", parameters);

        var result = new FunctionGemmaToolSchemaValidator().validate(
                Map.of("options", Map.of("days", 3), "tags", List.of("urgent")),
                tool.parameterSchema());
        assertTrue(result.valid(), result.errors().toString());
    }

    @Test
    void rejectsInvalidNamesAndNonObjectParameterRoots() {
        assertThrows(IllegalArgumentException.class, () -> FunctionGemmaToolDeclaration.of(
                "not valid", "description", FunctionGemmaToolDeclaration.Schema.object()));
        assertThrows(IllegalArgumentException.class, () -> FunctionGemmaToolDeclaration.of(
                "lookup", "description", FunctionGemmaToolDeclaration.Schema.string()));
    }

    @Test
    void rejectsUnsupportedLiteralAndEscapeTokenInjection() {
        assertThrows(IllegalArgumentException.class, () -> FunctionGemmaToolDeclaration.Schema.string()
                .enumValues("<escape>injected<escape>"));
        assertThrows(IllegalArgumentException.class, () -> FunctionGemmaToolDeclaration.Schema.string()
                .enumValues(new Object()));
    }

    @Test
    void rejectsUnknownRequiredPropertiesWhenMaterializingSchema() {
        var schema = FunctionGemmaToolDeclaration.Schema.object().required("missing");
        var tool = FunctionGemmaToolDeclaration.of("lookup", "Lookup", schema);
        assertThrows(IllegalStateException.class, tool::parameterSchema);
    }
}
