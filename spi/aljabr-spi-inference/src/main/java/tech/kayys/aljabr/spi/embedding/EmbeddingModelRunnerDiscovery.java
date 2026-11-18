package tech.kayys.aljabr.spi.embedding;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Discovers embedding model runners through Java's {@link ServiceLoader}.
 *
 * <p>Discovery constructs runners and reads their family identifiers only.
 * It does not load model weights, initialize tensor backends, or execute
 * inference. Runtime owners retain control over those lifecycle operations.</p>
 *
 * <p>Discovery is scoped to the supplied classloader and returns an immutable
 * snapshot. If multiple providers claim the same model family, all providers
 * claiming that family are excluded and reported as failures.</p>
 */
public final class EmbeddingModelRunnerDiscovery {
    private EmbeddingModelRunnerDiscovery() {
    }

    /**
     * Discovers runners visible to {@code classLoader}.
     *
     * @param classLoader classloader defining the runner provider scope
     * @return immutable runners and discovery diagnostics
     */
    public static EmbeddingModelRunnerDiscoveryResult discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");

        Map<String, EmbeddingModelRunner> runnersByFamily = new LinkedHashMap<>();
        Set<String> duplicateFamilies = new HashSet<>();
        List<EmbeddingModelRunnerDiscoveryFailure> failures = new ArrayList<>();

        Iterator<ServiceLoader.Provider<EmbeddingModelRunner>> providers =
                ServiceLoader.load(EmbeddingModelRunner.class, classLoader).stream().iterator();

        while (true) {
            ServiceLoader.Provider<EmbeddingModelRunner> providerReference;
            try {
                if (!providers.hasNext()) {
                    break;
                }
                providerReference = providers.next();
            } catch (ServiceConfigurationError error) {
                failures.add(failure("unknown",
                        "Unable to enumerate embedding runner service configuration", error));
                break;
            }

            String providerType = "unknown";
            try {
                providerType = providerReference.type().getName();
                EmbeddingModelRunner runner = providerReference.get();
                String family = Objects.requireNonNull(
                        runner.modelFamily(), "runner modelFamily");
                if (family.isBlank()) {
                    throw new IllegalArgumentException("runner modelFamily must not be blank");
                }

                if (duplicateFamilies.contains(family)) {
                    failures.add(failure(providerType,
                            "Duplicate embedding runner family '" + family + "'; implementation excluded",
                            null));
                    continue;
                }

                EmbeddingModelRunner previous = runnersByFamily.putIfAbsent(family, runner);
                if (previous != null) {
                    runnersByFamily.remove(family);
                    duplicateFamilies.add(family);
                    failures.add(failure(previous.getClass().getName(),
                            "Duplicate embedding runner family '" + family + "'; implementation excluded",
                            null));
                    failures.add(failure(providerType,
                            "Duplicate embedding runner family '" + family + "'; implementation excluded",
                            null));
                }
            } catch (ServiceConfigurationError | LinkageError | RuntimeException error) {
                failures.add(failure(providerType, "Embedding runner discovery failed", error));
            }
        }

        return new EmbeddingModelRunnerDiscoveryResult(
                new ArrayList<>(runnersByFamily.values()), failures);
    }

    private static EmbeddingModelRunnerDiscoveryFailure failure(
            String providerType, String message, Throwable cause) {
        return new EmbeddingModelRunnerDiscoveryFailure(providerType, message, cause);
    }
}
