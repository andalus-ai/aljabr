package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class GgufTensorRequirementMatcherTest {

    @Test
    void matchesRequiredNameShapeAndQuantizationType() {
        GgufTensorRequirement requirement = new GgufTensorRequirement(
                "blk.0.attn_q.weight", List.of(8L, 16L), Set.of(0, 2));

        GgufTensorRequirementMatch result = GgufTensorRequirementMatcher.match(
                List.of(requirement),
                List.of(new GgufTensorDescriptor("blk.0.attn_q.weight", List.of(8L, 16L), 2)));

        assertTrue(result.compatible());
        assertTrue(result.diagnostics().isEmpty());
    }

    @Test
    void reportsMissingTensorShapeAndTypeMismatchesTogether() {
        List<GgufTensorRequirement> requirements = List.of(
                GgufTensorRequirement.named("missing.weight"),
                GgufTensorRequirement.withShape("projection.weight", 4, 8),
                new GgufTensorRequirement("embedding.weight", List.of(), Set.of(7)));

        GgufTensorRequirementMatch result = GgufTensorRequirementMatcher.match(
                requirements,
                List.of(
                        new GgufTensorDescriptor("projection.weight", List.of(8L, 4L), 0),
                        new GgufTensorDescriptor("embedding.weight", List.of(16L, 16L), 2)));

        assertFalse(result.compatible());
        assertEquals(3, result.diagnostics().size());
        assertTrue(result.diagnostics().get(0).contains("Missing required tensor"));
        assertTrue(result.diagnostics().get(1).contains("dimensions"));
        assertTrue(result.diagnostics().get(2).contains("GGML type id"));
    }

    @Test
    void rejectsDuplicateRequirementsAndDuplicateInventoryEntries() {
        GgufTensorRequirement requirement = GgufTensorRequirement.named("weight");
        GgufTensorDescriptor tensor = new GgufTensorDescriptor("weight", List.of(2L, 2L), 0);

        assertThrows(IllegalArgumentException.class,
                () -> GgufTensorRequirementMatcher.match(
                        List.of(requirement, requirement), List.of(tensor)));
        assertThrows(IllegalArgumentException.class,
                () -> GgufTensorRequirementMatcher.match(
                        List.of(requirement), List.of(tensor, tensor)));
    }

    @Test
    void validatesRequirementAndDescriptorInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> new GgufTensorDescriptor("weight", List.of(0L, 2L), 0));
        assertThrows(IllegalArgumentException.class,
                () -> new GgufTensorRequirement("weight", List.of(), Set.of(-1)));
    }
}
