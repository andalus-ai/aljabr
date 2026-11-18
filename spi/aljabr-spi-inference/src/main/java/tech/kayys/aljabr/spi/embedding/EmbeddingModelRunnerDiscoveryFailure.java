package tech.kayys.aljabr.spi.embedding;

/**
 * Diagnostic for an embedding runner that could not be discovered.
 *
 * @param providerType implementation class name, or {@code unknown} when the
 *        service declaration itself could not be enumerated
 * @param message concise discovery failure description
 * @param cause optional underlying failure
 */
public record EmbeddingModelRunnerDiscoveryFailure(
        String providerType,
        String message,
        Throwable cause) {
}
