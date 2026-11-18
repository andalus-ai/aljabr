package tech.kayys.aljabr.spi.embedding;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable, classloader-scoped index of discoverable embedding model runners.
 *
 * <p>The registry is a snapshot, not a process-wide singleton. Creating it
 * performs metadata-only service discovery; it does not initialize model
 * weights or compute embeddings.</p>
 */
public final class EmbeddingModelRunnerRegistry {
    private final Map<String, EmbeddingModelRunner> runners;
    private final java.util.List<EmbeddingModelRunnerDiscoveryFailure> discoveryFailures;

    private EmbeddingModelRunnerRegistry(EmbeddingModelRunnerDiscoveryResult discovery) {
        Map<String, EmbeddingModelRunner> indexed = new LinkedHashMap<>();
        for (EmbeddingModelRunner runner : discovery.runners()) {
            indexed.put(runner.modelFamily(), runner);
        }
        this.runners = Collections.unmodifiableMap(indexed);
        this.discoveryFailures = discovery.failures();
    }

    /**
     * Discovers runners visible to the supplied classloader.
     *
     * @param classLoader classloader defining the runner provider scope
     * @return immutable registry snapshot
     */
    public static EmbeddingModelRunnerRegistry discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");
        return new EmbeddingModelRunnerRegistry(
                EmbeddingModelRunnerDiscovery.discover(classLoader));
    }

    /** Returns the runner for an exact model-family identifier. */
    public Optional<EmbeddingModelRunner> find(String modelFamily) {
        Objects.requireNonNull(modelFamily, "modelFamily");
        return Optional.ofNullable(runners.get(modelFamily));
    }

    /** Returns runners in stable discovery order. */
    public Collection<EmbeddingModelRunner> runners() {
        return runners.values();
    }

    /** Returns diagnostics for providers excluded during discovery. */
    public java.util.List<EmbeddingModelRunnerDiscoveryFailure> discoveryFailures() {
        return discoveryFailures;
    }

    public int size() {
        return runners.size();
    }

    public boolean isEmpty() {
        return runners.isEmpty();
    }
}
