package tech.kayys.aljabr.spi.embedding;

import java.util.List;

/**
 * Immutable result of one embedding-runner service discovery pass.
 *
 * @param runners uniquely identified runners discovered successfully
 * @param failures diagnostics for invalid or unavailable providers
 */
public record EmbeddingModelRunnerDiscoveryResult(
        List<EmbeddingModelRunner> runners,
        List<EmbeddingModelRunnerDiscoveryFailure> failures) {

    public EmbeddingModelRunnerDiscoveryResult {
        runners = List.copyOf(runners);
        failures = List.copyOf(failures);
    }
}
