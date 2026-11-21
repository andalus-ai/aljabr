package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionFamilyProvider;
import tech.kayys.aljabr.diffusion.api.DiffusionModelDescriptor;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable registry of optional diffusion-family providers.
 *
 * <p>Providers are injected by the application or discovered by an optional
 * integration. This class intentionally avoids static/global registration.</p>
 */
public final class DiffusionFamilyRegistry {

    private final Map<String, DiffusionFamilyProvider> providers;

    public DiffusionFamilyRegistry(Collection<? extends DiffusionFamilyProvider> providers) {
        Objects.requireNonNull(providers, "providers");
        Map<String, DiffusionFamilyProvider> indexed = new LinkedHashMap<>();
        for (DiffusionFamilyProvider provider : providers) {
            Objects.requireNonNull(provider, "provider");
            String familyId = requireFamilyId(provider.familyId());
            DiffusionFamilyProvider previous = indexed.putIfAbsent(familyId, provider);
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate diffusion family provider: " + familyId);
            }
        }
        this.providers = Map.copyOf(indexed);
    }

    public DiffusionFamilyProvider requireProvider(DiffusionModelDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        DiffusionFamilyProvider provider = providers.get(descriptor.familyId());
        if (provider == null || !provider.supports(descriptor)) {
            throw new UnsupportedDiffusionFamilyException(
                    "No installed diffusion provider supports family '%s' architecture '%s'"
                            .formatted(descriptor.familyId(), descriptor.architecture()));
        }
        return provider;
    }

    public Collection<String> familyIds() {
        return providers.keySet();
    }

    private static String requireFamilyId(String familyId) {
        Objects.requireNonNull(familyId, "familyId");
        if (familyId.isBlank()) {
            throw new IllegalArgumentException("provider familyId must not be blank");
        }
        return familyId;
    }
}
