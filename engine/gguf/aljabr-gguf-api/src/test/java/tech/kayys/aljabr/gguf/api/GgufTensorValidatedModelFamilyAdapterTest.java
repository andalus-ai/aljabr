package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class GgufTensorValidatedModelFamilyAdapterTest {

    private static final String FAMILY = "example.family";

    @Test
    void matchesArchitectureAndVerifiedTensorContract() {
        GgufTensorValidatedModelFamilyAdapter adapter = adapter(
                Set.of("example_arch"),
                List.of(new GgufTensorRequirement(
                        "token_embd.weight", List.of(8L, 16L), Set.of(0, 2))));

        GgufModelFamilyInspection result = adapter.inspect(descriptor(
                "example_arch",
                List.of(new GgufTensorDescriptor("token_embd.weight", List.of(8L, 16L), 2))));

        assertEquals(GgufModelFamilyInspection.Status.MATCH, result.status());
        assertTrue(result.diagnostics().isEmpty());
    }

    @Test
    void rejectsMissingOrUnsupportedArchitectureBeforeTensorValidation() {
        GgufTensorValidatedModelFamilyAdapter adapter = adapter(
                Set.of("example_arch"), List.of(GgufTensorRequirement.named("weight")));

        GgufModelFamilyInspection missing = adapter.inspect(descriptor(null, List.of()));
        GgufModelFamilyInspection unsupported = adapter.inspect(descriptor("other_arch", List.of()));

        assertEquals(GgufModelFamilyInspection.Status.UNSUPPORTED_ARCHITECTURE, missing.status());
        assertEquals(GgufModelFamilyInspection.Status.UNSUPPORTED_ARCHITECTURE, unsupported.status());
        assertTrue(missing.diagnostics().getFirst().contains("architecture"));
    }

    @Test
    void reportsTensorContractFailuresAsIncompleteArtifact() {
        GgufTensorValidatedModelFamilyAdapter adapter = adapter(
                Set.of("example_arch"),
                List.of(
                        GgufTensorRequirement.named("missing.weight"),
                        GgufTensorRequirement.withShape("projection.weight", 4, 8)));

        GgufModelFamilyInspection result = adapter.inspect(descriptor(
                "example_arch",
                List.of(new GgufTensorDescriptor("projection.weight", List.of(8L, 4L), 0))));

        assertEquals(GgufModelFamilyInspection.Status.INCOMPLETE_ARTIFACT, result.status());
        assertEquals(2, result.diagnostics().size());
        assertTrue(result.diagnostics().getFirst().contains("Missing required tensor"));
        assertTrue(result.diagnostics().get(1).contains("dimensions"));
    }

    @Test
    void requiresNonEmptyArchitectureDeclaration() {
        GgufTensorValidatedModelFamilyAdapter adapter = adapter(Set.of(), List.of());
        assertThrows(IllegalStateException.class,
                () -> adapter.inspect(descriptor("example_arch", List.of())));
    }

    private static GgufModelDescriptor descriptor(
            String architecture, List<GgufTensorDescriptor> tensors) {
        return GgufModelDescriptor.inspectWithTensors(
                architecture == null ? Map.of() : Map.of("general.architecture", architecture),
                tensors);
    }

    private static GgufTensorValidatedModelFamilyAdapter adapter(
            Set<String> architectures, List<GgufTensorRequirement> requirements) {
        return new GgufTensorValidatedModelFamilyAdapter() {
            @Override
            public String familyId() {
                return FAMILY;
            }

            @Override
            public Set<String> supportedArchitectures() {
                return architectures;
            }

            @Override
            public List<GgufTensorRequirement> tensorRequirements() {
                return requirements;
            }
        };
    }
}
