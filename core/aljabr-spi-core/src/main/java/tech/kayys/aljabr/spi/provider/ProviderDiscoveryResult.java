package tech.kayys.aljabr.spi.provider;

import java.util.List;
import java.util.Objects;

/** Immutable result of a provider discovery pass. */
public record ProviderDiscoveryResult(
        List<AljabrProvider> providers,
        List<ProviderDiscoveryFailure> failures) {

    public ProviderDiscoveryResult {
        providers = List.copyOf(Objects.requireNonNull(providers, "providers"));
        failures = List.copyOf(Objects.requireNonNull(failures, "failures"));
    }
}
