package tech.kayys.aljabr.spi.testkit;

import java.util.List;
import java.util.Objects;

/**
 * Immutable outcome of checking the provider metadata and basic lifecycle contract.
 *
 * <p>This result deliberately does not claim that a provider's numerical operations,
 * hardware support, performance, or declared capabilities have been validated.</p>
 */
public record ProviderConformanceResult(String providerId, List<String> violations) {
    public ProviderConformanceResult {
        providerId = Objects.requireNonNull(providerId, "providerId");
        violations = List.copyOf(Objects.requireNonNull(violations, "violations"));
    }

    public boolean isConformant() {
        return violations.isEmpty();
    }
}
