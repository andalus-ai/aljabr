package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class GgufModelFamilyRegistryTest {

    private static final GgufModelDescriptor DESCRIPTOR =
            GgufModelDescriptor.inspect(Map.of("general.architecture", "sample"), List.of());

    @Test
    void resolvesOneMatchingAdapterInStableOrder() {
        GgufModelFamilyAdapter zeta = adapter("zeta", false);
        GgufModelFamilyAdapter alpha = adapter("alpha", true);
        GgufModelFamilyRegistry registry = new GgufModelFamilyRegistry(List.of(zeta, alpha));

        GgufModelFamilyRegistry.Resolution result = registry.resolve(DESCRIPTOR);

        assertEquals(GgufModelFamilyRegistry.Status.MATCH, result.status());
        assertEquals("alpha", result.adapter().orElseThrow().familyId());
        assertEquals(List.of("alpha", "zeta"),
                registry.adapters().stream().map(GgufModelFamilyAdapter::familyId).toList());
    }

    @Test
    void returnsNoMatchWhenAllAdaptersRejectArtifact() {
        GgufModelFamilyRegistry.Resolution result = new GgufModelFamilyRegistry(
                List.of(adapter("sample", false))).resolve(DESCRIPTOR);

        assertEquals(GgufModelFamilyRegistry.Status.NO_MATCH, result.status());
        assertTrue(result.adapter().isEmpty());
    }

    @Test
    void refusesAmbiguousFamilyMatches() {
        GgufModelFamilyRegistry.Resolution result = new GgufModelFamilyRegistry(
                List.of(adapter("one", true), adapter("two", true))).resolve(DESCRIPTOR);

        assertEquals(GgufModelFamilyRegistry.Status.AMBIGUOUS, result.status());
        assertTrue(result.adapter().isEmpty());
        assertFalse(result.diagnostics().isEmpty());
    }

    @Test
    void adapterFailurePreventsUnsafeUniqueMatch() {
        GgufModelFamilyAdapter broken = new GgufModelFamilyAdapter() {
            @Override public String familyId() { return "broken"; }
            @Override public GgufModelFamilyInspection inspect(GgufModelDescriptor descriptor) {
                throw new IllegalStateException("bad adapter");
            }
        };
        GgufModelFamilyRegistry.Resolution result = new GgufModelFamilyRegistry(
                List.of(adapter("match", true), broken)).resolve(DESCRIPTOR);

        assertEquals(GgufModelFamilyRegistry.Status.ADAPTER_FAILURE, result.status());
        assertTrue(result.adapter().isEmpty());
        assertTrue(result.diagnostics().getFirst().contains("bad adapter"));
    }

    @Test
    void rejectsDuplicateFamilyIds() {
        assertThrows(IllegalArgumentException.class,
                () -> new GgufModelFamilyRegistry(List.of(
                        adapter("duplicate", false), adapter("duplicate", true))));
    }

    private static GgufModelFamilyAdapter adapter(String familyId, boolean matches) {
        return new GgufModelFamilyAdapter() {
            @Override public String familyId() { return familyId; }
            @Override public GgufModelFamilyInspection inspect(GgufModelDescriptor descriptor) {
                return matches
                        ? GgufModelFamilyInspection.match(familyId)
                        : GgufModelFamilyInspection.rejected(
                                familyId,
                                GgufModelFamilyInspection.Status.UNSUPPORTED_ARCHITECTURE,
                                List.of("Architecture not supported"));
            }
        };
    }
}
