package tech.kayys.aljabr.runtime.compute;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDiscovery;
import tech.kayys.aljabr.spi.provider.ProviderDiscoveryResult;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * Owns provider initialization and the device session backing a selected compute backend.
 *
 * <p>The tensor module deliberately has no dependency on the compute-provider SPI.
 * Applications that need provider discovery open this runtime and pass
 * {@link #backend()} to the explicit-backend TensorFactory overloads. The runtime
 * must remain open while tensors created with that backend are in use.</p>
 */
public final class ComputeRuntime implements AutoCloseable {
    public static final String PROVIDER_PROPERTY = "aljabr.compute.provider";
    private static final String ALJABR_VERSION = "0.1.0";

    private final List<AljabrProvider> ownedProviders;
    private final ProviderSession session;
    private final ComputeBackend backend;
    private final DeviceDescriptor device;
    private boolean closed;

    private ComputeRuntime(
            List<AljabrProvider> ownedProviders,
            ProviderSession session,
            ComputeBackend backend,
            DeviceDescriptor device) {
        this.ownedProviders = List.copyOf(ownedProviders);
        this.session = session;
        this.backend = backend;
        this.device = device;
    }

    /** Opens the best available provider, preferring GPU, accelerator, then CPU devices. */
    public static ComputeRuntime open() {
        String providerId = System.getProperty(PROVIDER_PROPERTY, "auto").trim();
        return open(providerId.isEmpty() ? "auto" : providerId, Map.of());
    }

    /**
     * Opens a provider selected by stable provider id, or uses {@code auto}.
     * Explicit provider selection never silently falls back to another provider.
     */
    public static ComputeRuntime open(String providerId) {
        return open(providerId, Map.of());
    }

    /** Opens a provider with immutable provider initialization configuration. */
    public static ComputeRuntime open(String providerId, Map<String, String> configuration) {
        Objects.requireNonNull(providerId, "providerId");
        Objects.requireNonNull(configuration, "configuration");
        String requested = providerId.trim();
        if (requested.isEmpty()) {
            throw new IllegalArgumentException("providerId must not be blank");
        }

        ProviderDiscoveryResult discovery = ProviderDiscovery.discover(
                Thread.currentThread().getContextClassLoader() == null
                        ? ComputeRuntime.class.getClassLoader()
                        : Thread.currentThread().getContextClassLoader());
        List<AljabrProvider> owned = new ArrayList<>(discovery.providers());
        List<Candidate> candidates = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        ProviderContext context = new ProviderContext(ALJABR_VERSION, configuration);

        try {
            for (AljabrProvider provider : owned) {
                if (!(provider instanceof ComputeProvider computeProvider)) {
                    continue;
                }
                String id = provider.descriptor().id();
                if (!"auto".equals(requested) && !requested.equals(id)) {
                    continue;
                }
                try {
                    computeProvider.initialize(context);
                    for (DeviceDescriptor device : computeProvider.devices()) {
                        candidates.add(new Candidate(computeProvider, device));
                    }
                } catch (ProviderException | RuntimeException | LinkageError failure) {
                    failures.add(id + ": " + failure.getMessage());
                    if (!"auto".equals(requested)) {
                        throw new IllegalStateException(
                                "Unable to initialize requested compute provider '" + requested + "'", failure);
                    }
                }
            }

            if (!"auto".equals(requested)
                    && candidates.stream().noneMatch(candidate ->
                            requested.equals(candidate.provider().descriptor().id()))) {
                throw new IllegalStateException("Requested compute provider is unavailable: " + requested
                        + diagnosticSuffix(failures));
            }

            candidates.sort(Comparator
                    .comparingInt((Candidate candidate) -> deviceRank(candidate.device().kind()))
                    .thenComparing(candidate -> candidate.provider().descriptor().id())
                    .thenComparing(candidate -> candidate.device().deviceId()));

            for (Candidate candidate : candidates) {
                ProviderSession openedSession = null;
                try {
                    openedSession = candidate.provider().openSession(candidate.device());
                    Map<String, ComputeBackend> backends = new LinkedHashMap<>(openedSession.computeBackends());
                    if (!backends.isEmpty()) {
                        ComputeBackend backend = backends.entrySet().stream()
                                .sorted(Map.Entry.comparingByKey())
                                .map(Map.Entry::getValue)
                                .findFirst()
                                .orElseThrow();
                        return new ComputeRuntime(owned, openedSession, backend, candidate.device());
                    }
                    openedSession.close();
                    openedSession = null;
                    failures.add(candidate.provider().descriptor().id()
                            + ": opened a session without compute backends");
                } catch (ProviderException | RuntimeException | LinkageError failure) {
                    if (openedSession != null) {
                        try {
                            openedSession.close();
                        } catch (ProviderException | RuntimeException closeFailure) {
                            failure.addSuppressed(closeFailure);
                        }
                    }
                    failures.add(candidate.provider().descriptor().id() + ": " + failure.getMessage());
                    if (!"auto".equals(requested)) {
                        throw new IllegalStateException(
                                "Unable to open requested compute provider '" + requested + "'", failure);
                    }
                }
            }

            throw new IllegalStateException("No usable Aljabr compute provider was found"
                    + diagnosticSuffix(failures));
        } catch (RuntimeException | Error failure) {
            closeProviders(owned, failure);
            throw failure;
        }
    }

    public ComputeBackend backend() {
        ensureOpen();
        return backend;
    }

    public DeviceDescriptor device() {
        ensureOpen();
        return device;
    }

    public String providerId() {
        ensureOpen();
        return device.providerId();
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException closeFailure = null;
        try {
            session.close();
        } catch (ProviderException failure) {
            closeFailure = new IllegalStateException("Unable to close compute provider session", failure);
        }
        for (AljabrProvider provider : ownedProviders) {
            try {
                provider.close();
            } catch (ProviderException | RuntimeException failure) {
                if (closeFailure == null) {
                    closeFailure = new IllegalStateException("Unable to close compute provider", failure);
                } else {
                    closeFailure.addSuppressed(failure);
                }
            }
        }
        if (closeFailure != null) {
            throw closeFailure;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Compute runtime is closed");
        }
    }

    private static int deviceRank(DeviceKind kind) {
        return switch (kind) {
            case GPU -> 0;
            case ACCELERATOR -> 1;
            case CPU -> 2;
            case OTHER -> 3;
        };
    }

    private static String diagnosticSuffix(List<String> failures) {
        return failures.isEmpty() ? "" : "; provider failures: " + String.join("; ", failures);
    }

    private static void closeProviders(List<AljabrProvider> providers, Throwable primary) {
        for (AljabrProvider provider : providers) {
            try {
                provider.close();
            } catch (ProviderException | RuntimeException closeFailure) {
                primary.addSuppressed(closeFailure);
            }
        }
    }

    private record Candidate(ComputeProvider provider, DeviceDescriptor device) {
    }
}
