package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GgufModelFamilyInspectionTest {

    @Test
    void matchIsAnArtifactMatchNotAnExecutionClaim() {
        GgufModelFamilyInspection result = GgufModelFamilyInspection.match("org.example.family");

        assertTrue(result.matched());
        assertEquals(GgufModelFamilyInspection.Status.MATCH, result.status());
        assertTrue(result.diagnostics().isEmpty());
    }

    @Test
    void rejectedInspectionRequiresActionableDiagnostics() {
        GgufModelFamilyInspection result = GgufModelFamilyInspection.rejected(
                "org.example.family",
                GgufModelFamilyInspection.Status.INCOMPLETE_ARTIFACT,
                List.of("Missing required tensor: token_embd.weight"));

        assertFalse(result.matched());
        assertEquals(1, result.diagnostics().size());
    }

    @Test
    void diagnosticsAreDefensivelyCopied() {
        List<String> diagnostics = new ArrayList<>(List.of("Unsupported architecture"));
        GgufModelFamilyInspection result = GgufModelFamilyInspection.rejected(
                "org.example.family",
                GgufModelFamilyInspection.Status.UNSUPPORTED_ARCHITECTURE,
                diagnostics);

        diagnostics.add("mutated");
        assertEquals(List.of("Unsupported architecture"), result.diagnostics());
        assertThrows(UnsupportedOperationException.class,
                () -> result.diagnostics().add("mutated"));
    }

    @Test
    void rejectsInvalidResultShapes() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelFamilyInspection.match(" "));
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelFamilyInspection.rejected(
                        "family", GgufModelFamilyInspection.Status.MATCH, List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelFamilyInspection.rejected(
                        "family", GgufModelFamilyInspection.Status.UNSUPPORTED_ARTIFACT, List.of()));
    }
}
