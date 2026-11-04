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
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;

/**
 * Immutable, classloader-scoped snapshot of discoverable Aljabr providers.
 *
 * <p>Discovery reads provider metadata only. It never calls {@link AljabrProvider#initialize(ProviderContext)}
 * and therefore does not initialize native runtimes or claim that a device is available.</p>
 */
public final class ProviderRegistry {
    private final Map<String, AljabrProvider> providers;

    private ProviderRegistry(Map<String, AljabrProvider> providers) {
        this.providers = Collections.unmodifiableMap(new LinkedHashMap<>(providers));
    }

    /** Discovers providers visible to the supplied classloader without initializing them. */
    public static ProviderRegistry discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");
        List<AljabrProvider> discovered = new ArrayList<>();
        try {
            ServiceLoader.load(AljabrProvider.class, classLoader).forEach(discovered::add);
        } catch (ServiceConfigurationError error) {
            throw new IllegalStateException("Unable to discover Aljabr providers", error);
        }
        return fromProviders(discovered);
    }

    static ProviderRegistry fromProviders(Collection<? extends AljabrProvider> discovered) {
        Objects.requireNonNull(discovered, "discovered");
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
        return new ProviderRegistry(indexed);
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

    public int size() {
        return providers.size();
    }

    public boolean isEmpty() {
        return providers.isEmpty();
    }
}
