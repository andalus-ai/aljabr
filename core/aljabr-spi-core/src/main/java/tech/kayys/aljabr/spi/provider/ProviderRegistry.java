package tech.kayys.aljabr.spi.provider;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable, classloader-scoped snapshot of discoverable Aljabr providers.
 *
 * <p>Discovery reads provider metadata only. It never calls
 * {@link AljabrProvider#initialize(ProviderContext)} and therefore does not
 * initialize native runtimes or claim that a device is available.</p>
 *
 * <p>Provider construction, descriptor validation, duplicate-id handling, and
 * per-provider diagnostics are delegated to {@link ProviderDiscovery}. This
 * registry is the canonical indexed view over that discovery result.</p>
 */
public final class ProviderRegistry {
    private final Map<String, AljabrProvider> providers;
    private final List<ProviderDiscoveryFailure> discoveryFailures;

    private ProviderRegistry(
            Map<String, AljabrProvider> providers,
            List<ProviderDiscoveryFailure> discoveryFailures) {
        this.providers = Collections.unmodifiableMap(new LinkedHashMap<>(providers));
        this.discoveryFailures = List.copyOf(discoveryFailures);
    }

    /**
     * Discovers providers visible to the supplied classloader without
     * initializing them. Broken providers are excluded while healthy providers
     * remain available; inspect {@link #discoveryFailures()} for diagnostics.
     */
    public static ProviderRegistry discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");
        ProviderDiscoveryResult result = ProviderDiscovery.discover(classLoader);
        return fromProviders(result.providers(), result.failures());
    }

    static ProviderRegistry fromProviders(Collection<? extends AljabrProvider> discovered) {
        return fromProviders(discovered, List.of());
    }

    private static ProviderRegistry fromProviders(
            Collection<? extends AljabrProvider> discovered,
            List<ProviderDiscoveryFailure> failures) {
        Objects.requireNonNull(discovered, "discovered");
        Objects.requireNonNull(failures, "failures");
        List<AljabrProvider> ordered = new ArrayList<>(discovered);
        ordered.sort(Comparator.comparing(provider -> {
            Objects.requireNonNull(provider, "provider");
            ProviderDescriptor descriptor = Objects.requireNonNull(provider.descriptor(),
                    "provider descriptor");
            return descriptor.id();
        }));
        Map<String, AljabrProvider> indexed = new LinkedHashMap<>();
        for (AljabrProvider provider : ordered) {
            ProviderDescriptor descriptor = Objects.requireNonNull(provider.descriptor(),
                    "provider descriptor");
            AljabrProvider previous = indexed.putIfAbsent(descriptor.id(), provider);
            if (previous != null) {
                throw new IllegalStateException("Duplicate Aljabr provider id: " + descriptor.id());
            }
        }
        return new ProviderRegistry(indexed, failures);
    }

    /** Returns the provider for an exact, namespaced provider id. */
    public Optional<AljabrProvider> find(String providerId) {
        Objects.requireNonNull(providerId, "providerId");
        return Optional.ofNullable(providers.get(providerId));
    }

    /** Returns an immutable provider collection in stable provider-id order. */
    public Collection<AljabrProvider> providers() {
        return providers.values();
    }

    /** Returns immutable diagnostics for providers excluded during discovery. */
    public List<ProviderDiscoveryFailure> discoveryFailures() {
        return discoveryFailures;
    }

    public int size() {
        return providers.size();
    }

    public boolean isEmpty() {
        return providers.isEmpty();
    }
}
