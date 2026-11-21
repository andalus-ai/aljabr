package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionFamilyProvider;
import tech.kayys.aljabr.diffusion.api.DiffusionModelDescriptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionFamilyRegistryTest {

    @Test
    void resolvesAnInstalledFamilyProvider() {
        DiffusionFamilyProvider provider = provider("stable-diffusion", "stable-diffusion-xl");
        DiffusionFamilyRegistry registry = new DiffusionFamilyRegistry(List.of(provider));

        assertSame(provider, registry.requireProvider(
                new DiffusionModelDescriptor("stable-diffusion", "stable-diffusion-xl")));
        assertEquals(List.of("stable-diffusion"), registry.familyIds().stream().toList());
    }

    @Test
    void rejectsUnknownFamilyOrArchitecture() {
        DiffusionFamilyRegistry registry = new DiffusionFamilyRegistry(
                List.of(provider("stable-diffusion", "stable-diffusion-xl")));

        assertThrows(UnsupportedDiffusionFamilyException.class, () ->
                registry.requireProvider(new DiffusionModelDescriptor("flux", "flux-dev")));
        assertThrows(UnsupportedDiffusionFamilyException.class, () ->
                registry.requireProvider(new DiffusionModelDescriptor("stable-diffusion", "unknown")));
    }

    @Test
    void rejectsDuplicateProvidersForSameFamily() {
        assertThrows(IllegalArgumentException.class, () -> new DiffusionFamilyRegistry(List.of(
                provider("flux", "flux-dev"),
                provider("flux", "flux-schnell"))));
    }

    private static DiffusionFamilyProvider provider(String family, String architecture) {
        return new DiffusionFamilyProvider() {
            @Override
            public String familyId() {
                return family;
            }

            @Override
            public boolean supports(DiffusionModelDescriptor descriptor) {
                return family.equals(descriptor.familyId())
                        && architecture.equals(descriptor.architecture());
            }
        };
    }
}
