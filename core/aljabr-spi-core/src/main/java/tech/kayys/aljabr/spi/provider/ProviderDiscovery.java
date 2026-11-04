package tech.kayys.aljabr.spi.provider;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Performs an isolated ServiceLoader discovery pass using an explicit classloader.
 *
 * <p>Discovery instantiates providers and reads their descriptors only. It never
 * initializes providers, enumerates devices, opens sessions, or loads accelerator
 * runtimes on their behalf. Runtime owners retain control over those operations.
 *
 * <p>This type has no process-wide registry or mutable static state. Each call
 * returns an immutable snapshot and diagnostics for providers that could not be
 * loaded or whose identifiers conflict.
 */
public final class ProviderDiscovery {
    private ProviderDiscovery() {
    }

    /**
     * Discovers {@link AljabrProvider} implementations visible to {@code classLoader}.
     *
     * <p>Provider construction and descriptor failures are isolated so that a
     * broken optional provider does not prevent other providers from being found.
     * If multiple implementations advertise the same provider id, every
     * implementation with that id is excluded from the result.
     */
    public static ProviderDiscoveryResult discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");

        List<ProviderDiscoveryFailure> failures = new ArrayList<>();
        Map<String, AljabrProvider> providersById = new LinkedHashMap<>();
        Set<String> duplicateIds = new HashSet<>();

        ServiceLoader<AljabrProvider> serviceLoader =
                ServiceLoader.load(AljabrProvider.class, classLoader);
        Iterator<ServiceLoader.Provider<AljabrProvider>> iterator =
                serviceLoader.stream().iterator();

        while (true) {
            ServiceLoader.Provider<AljabrProvider> providerReference;
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                providerReference = iterator.next();
            } catch (ServiceConfigurationError error) {
                failures.add(failure("unknown", "Unable to enumerate provider service configuration", error));
                // A malformed service declaration may leave the iterator in an
                // unrecoverable state. Stop enumeration rather than risk looping.
                break;
            }

            String providerType = "unknown";
            try {
                providerType = providerReference.type().getName();
                AljabrProvider provider = providerReference.get();
                ProviderDescriptor descriptor = Objects.requireNonNull(
                        provider.descriptor(), "provider descriptor");
                String providerId = descriptor.id();

                if (duplicateIds.contains(providerId)) {
                    failures.add(failure(providerType,
                            "Duplicate provider id '" + providerId + "'; implementation excluded", null));
                    continue;
                }

                AljabrProvider previous = providersById.putIfAbsent(providerId, provider);
                if (previous != null) {
                    providersById.remove(providerId);
                    duplicateIds.add(providerId);
                    failures.add(failure(previous.getClass().getName(),
                            "Duplicate provider id '" + providerId + "'; implementation excluded", null));
                    failures.add(failure(providerType,
                            "Duplicate provider id '" + providerId + "'; implementation excluded", null));
                }
            } catch (ServiceConfigurationError | LinkageError | RuntimeException error) {
                failures.add(failure(providerType, "Provider discovery failed", error));
            }
        }

        return new ProviderDiscoveryResult(new ArrayList<>(providersById.values()), failures);
    }

    private static ProviderDiscoveryFailure failure(
            String providerType, String message, Throwable cause) {
        return new ProviderDiscoveryFailure(providerType, message, cause);
    }
}
