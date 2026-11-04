package tech.kayys.aljabr.spi.provider;

/**
 * Describes one provider that could not be discovered or validated.
 *
 * @param providerType provider implementation class when known
 * @param message concise diagnostic suitable for logs and tooling
 * @param cause original discovery failure
 */
public record ProviderDiscoveryFailure(String providerType, String message, Throwable cause) {
    public ProviderDiscoveryFailure {
        providerType = providerType == null || providerType.isBlank() ? "unknown" : providerType;
        if (message == null || message.isBlank()) {
            throw new IllegalArgumentException("message must not be blank");
        }
    }
}
