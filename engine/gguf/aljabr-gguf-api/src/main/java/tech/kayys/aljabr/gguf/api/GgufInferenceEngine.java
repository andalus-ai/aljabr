package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
import tech.kayys.aljabr.execution.ModelExecutionContext;
import tech.kayys.aljabr.diagnostics.DiagnosticContext;
import tech.kayys.aljabr.diagnostics.ExecutionObserver;
import tech.kayys.aljabr.diagnostics.ExecutionScope;
import tech.kayys.aljabr.diagnostics.ProviderDecisionEvent;
import tech.kayys.aljabr.diagnostics.ProviderDecisionObserver;
import tech.kayys.aljabr.spi.embedding.EmbeddingRequest;
import tech.kayys.aljabr.spi.embedding.EmbeddingResponse;
import tech.kayys.aljabr.spi.exception.InferenceException;
import tech.kayys.aljabr.spi.inference.InferenceEngine;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;
import tech.kayys.aljabr.spi.inference.StreamingInferenceChunk;
import tech.kayys.aljabr.spi.model.HealthStatus;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.function.Supplier;
import java.util.concurrent.Callable;

/**
 * ServiceLoader-based GGUF inference engine implementing {@link InferenceEngine}.
 *
 * <p>Discovers backends via {@link java.util.ServiceLoader} so that:
 * <ul>
 *   <li>this module has no compile-time dependency on any concrete backend;</li>
 *   <li>"pure Java" is a structural property (omit the llama.cpp jar) rather
 *       than a runtime default;</li>
 *   <li>new backends need only a {@code META-INF/services} registration.</li>
 * </ul>
 */
public final class GgufInferenceEngine implements InferenceEngine, AutoCloseable {

    private final Map<String, GgufBackend> backends = new ConcurrentHashMap<>();
    private final List<GgufBackendProvider> providers;
    private final ProviderDecisionObserver decisionObserver;
    private final ExecutionObserver executionObserver;
    private final AtomicLong activeCount = new AtomicLong(0);
    private final AtomicLong totalCount = new AtomicLong(0);
    private final AtomicLong failedCount = new AtomicLong(0);
    private final AtomicLong decisionObserverFailureCount = new AtomicLong(0);
    private final AtomicLong executionObserverFailureCount = new AtomicLong(0);
    private volatile boolean initialized = false;

    public GgufInferenceEngine() {
        this(loadProviders());
    }

    public GgufInferenceEngine(List<GgufBackendProvider> providers) {
        this(providers, ProviderDecisionObserver.NOOP, ExecutionObserver.NOOP);
    }

    /**
     * Creates an engine with an explicit provider decision observer.
     * Observer runtime failures are isolated from provider selection.
     */
    public GgufInferenceEngine(
            List<GgufBackendProvider> providers,
            ProviderDecisionObserver decisionObserver) {
        this(providers, decisionObserver, ExecutionObserver.NOOP);
    }

    /**
     * Creates an engine with explicit provider-decision and execution observers.
     * Both observers are optional; runtime observer failures never alter inference.
     */
    public GgufInferenceEngine(
            List<GgufBackendProvider> providers,
            ProviderDecisionObserver decisionObserver,
            ExecutionObserver executionObserver) {
        List<GgufBackendProvider> sorted = new ArrayList<>(providers);
        sorted.sort(Comparator.comparingInt(GgufBackendProvider::priority).reversed());
        this.providers = List.copyOf(sorted);
        this.decisionObserver = Objects.requireNonNull(decisionObserver, "decisionObserver");
        this.executionObserver = Objects.requireNonNull(executionObserver, "executionObserver");
        this.initialized = true;
    }

    private static List<GgufBackendProvider> loadProviders() {
        List<GgufBackendProvider> found = new ArrayList<>();
        ServiceLoader.load(GgufBackendProvider.class, GgufInferenceEngine.class.getClassLoader())
                .forEach(found::add);
        return found;
    }

    public synchronized GgufBackend loadModel(String modelId, Path modelPath, Map<String, Object> config) throws Exception {
        Objects.requireNonNull(modelId, "modelId");
        if (modelId.isBlank()) {
            throw new IllegalArgumentException("modelId must not be blank");
        }
        if (backends.containsKey(modelId)) {
            throw new IllegalStateException(
                    "A GGUF backend is already registered for model ID '" + modelId
                            + "'; unload it before loading a replacement");
        }
        GgufBackendSelection selection = GgufBackendSelection.resolve(config);
        GgufBackendProvider provider = selectProvider(modelPath, selection);
        if (executionObserver == ExecutionObserver.NOOP) {
            GgufBackend backend = provider.create(modelPath, config);
            backends.put(modelId, backend);
            return backend;
        }
        return observeLifecycle("gguf.model.load", provider.id(), () -> {
            GgufBackend backend = provider.create(modelPath, config);
            backends.put(modelId, backend);
            return backend;
        });
    }

    private <T> T observeLifecycle(String operation, String provider, Callable<T> action) throws Exception {
        if (executionObserver == ExecutionObserver.NOOP) {
            return action.call();
        }
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId(UUID.randomUUID().toString())
                .operation(operation)
                .component("gguf")
                .provider(provider)
                .build();
        ExecutionScope scope = ExecutionScope.start(context, executionObserver);
        try (scope) {
            try {
                T result = action.call();
                scope.succeed();
                return result;
            } catch (Exception | Error failure) {
                scope.fail(failure);
                throw failure;
            }
        } finally {
            if (scope.observerFailure().isPresent()) {
                executionObserverFailureCount.incrementAndGet();
            }
        }
    }

    private void observeLifecycleUnchecked(String operation, String provider, Runnable action) {
        try {
            observeLifecycle(operation, provider, () -> {
                action.run();
                return null;
            });
        } catch (RuntimeException failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalStateException("GGUF lifecycle operation failed: " + operation, failure);
        }
    }

    /** Number of provider-observer runtime failures isolated by this engine. */
    public long providerObserverFailureCount() {
        return decisionObserverFailureCount.get();
    }

    /** Number of execution-observer runtime failures isolated by this engine. */
    public long executionObserverFailureCount() {
        return executionObserverFailureCount.get();
    }

    public synchronized void unloadModel(String modelId) {
        GgufBackend removed = backends.remove(modelId);
        if (removed != null) {
            if (executionObserver == ExecutionObserver.NOOP) {
                removed.close();
            } else {
                observeLifecycleUnchecked("gguf.model.unload", removed.name(), removed::close);
            }
        }
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        return Uni.createFrom().item(() -> execute(request.getModel(), request))
                .runSubscriptionOn(Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Uni<InferenceResponse> executeAsync(String modelId, InferenceRequest request) {
        return Uni.createFrom().item(() -> execute(modelId, request))
                .runSubscriptionOn(Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public InferenceResponse execute(String modelId, InferenceRequest request) {
        activeCount.incrementAndGet();
        totalCount.incrementAndGet();
        try {
            return executeObserved(backendFor(modelId), request, null);
        } catch (Exception error) {
            failedCount.incrementAndGet();
            if (error instanceof InferenceException inferenceException) {
                throw inferenceException;
            }
            throw new InferenceException("GGUF inference failed: " + error.getMessage(), error);
        } finally {
            activeCount.decrementAndGet();
        }
    }

    public InferenceResponse execute(InferenceRequest request) {
        return execute(request.getModel(), request);
    }

    private InferenceResponse executeObserved(
            GgufBackend backend,
            InferenceRequest request,
            ModelExecutionContext modelContext) {
        if (executionObserver == ExecutionObserver.NOOP) {
            return modelContext == null
                    ? backend.execute(request)
                    : backend.execute(request, modelContext);
        }
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId(UUID.randomUUID().toString())
                .operation("gguf.infer")
                .component("gguf")
                .provider(backend.name())
                .build();
        ExecutionScope scope = ExecutionScope.start(context, executionObserver);
        try (scope) {
            try {
                InferenceResponse response = modelContext == null
                        ? backend.execute(request)
                        : backend.execute(request, modelContext);
                scope.succeed();
                return response;
            } catch (RuntimeException | Error failure) {
                scope.fail(failure);
                throw failure;
            }
        } finally {
            if (scope.observerFailure().isPresent()) {
                executionObserverFailureCount.incrementAndGet();
            }
        }
    }

    /**
     * Execute a request with its model execution context.
     *
     * <p>The context is passed unchanged to the selected backend. The engine
     * does not allocate, mutate, or close any cache storage exposed by it.</p>
     */
    public InferenceResponse execute(
            String modelId, InferenceRequest request, ModelExecutionContext context) {
        Objects.requireNonNull(context, "context");
        activeCount.incrementAndGet();
        totalCount.incrementAndGet();
        try {
            return executeObserved(backendFor(modelId), request, context);
        } catch (Exception error) {
            failedCount.incrementAndGet();
            if (error instanceof InferenceException inferenceException) {
                throw inferenceException;
            }
            throw new InferenceException("GGUF inference failed: " + error.getMessage(), error);
        } finally {
            activeCount.decrementAndGet();
        }
    }

    /** Execute asynchronously while preserving the supplied execution context. */
    public Uni<InferenceResponse> executeAsync(
            String modelId, InferenceRequest request, ModelExecutionContext context) {
        Objects.requireNonNull(context, "context");
        return Uni.createFrom().item(() -> execute(modelId, request, context))
                .runSubscriptionOn(Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return streamExecute(request.getModel(), request);
    }

    @Override
    public Multi<StreamingInferenceChunk> streamExecute(String modelId, InferenceRequest request) {
        return Multi.createFrom().deferred(() -> {
            activeCount.incrementAndGet();
            totalCount.incrementAndGet();
            try {
                GgufBackend backend = backendFor(modelId);
                return observeStream(backend, () -> backend.stream(request));
            } catch (Exception error) {
                failedCount.incrementAndGet();
                activeCount.decrementAndGet();
                return Multi.createFrom().failure(error);
            } catch (Error fatal) {
                activeCount.decrementAndGet();
                throw fatal;
            }
        });
    }

    /** Stream inference while preserving the supplied model execution context. */
    public Multi<StreamingInferenceChunk> streamExecute(
            String modelId, InferenceRequest request, ModelExecutionContext context) {
        Objects.requireNonNull(context, "context");
        return Multi.createFrom().deferred(() -> {
            activeCount.incrementAndGet();
            totalCount.incrementAndGet();
            try {
                GgufBackend backend = backendFor(modelId);
                return observeStream(backend, () -> backend.stream(request, context));
            } catch (Exception error) {
                failedCount.incrementAndGet();
                activeCount.decrementAndGet();
                return Multi.createFrom().failure(error);
            }
        });
    }

    /**
     * Attaches one lifecycle scope to each stream subscription. The supplier is
     * invoked from the deferred subscription path, never when the Multi is
     * assembled. Completion, failure and cancellation have distinct outcomes.
     */
    private Multi<StreamingInferenceChunk> observeStream(
            GgufBackend backend,
            Supplier<Multi<StreamingInferenceChunk>> streamSupplier) {
        ExecutionScope scope = null;
        if (executionObserver != ExecutionObserver.NOOP) {
            DiagnosticContext context = DiagnosticContext.builder()
                    .correlationId(UUID.randomUUID().toString())
                    .operation("gguf.stream")
                    .component("gguf")
                    .provider(backend.name())
                    .build();
            scope = ExecutionScope.start(context, executionObserver);
        }
        final ExecutionScope activeScope = scope;
        try {
            Multi<StreamingInferenceChunk> stream =
                    Objects.requireNonNull(streamSupplier.get(), "backend stream");
            if (activeScope == null) {
                return stream
                        .onFailure().invoke(failure -> failedCount.incrementAndGet())
                        .onTermination().invoke(() -> activeCount.decrementAndGet());
            }
            return stream
                    .onFailure().invoke(failure -> {
                        failedCount.incrementAndGet();
                        activeScope.fail(failure);
                    })
                    .onCompletion().invoke(activeScope::succeed)
                    .onCancellation().invoke(activeScope::close)
                    .onTermination().invoke(() -> {
                        if (activeScope.observerFailure().isPresent()) {
                            executionObserverFailureCount.incrementAndGet();
                        }
                        activeCount.decrementAndGet();
                    });
        } catch (RuntimeException failure) {
            failedCount.incrementAndGet();
            try {
                if (activeScope != null) {
                    activeScope.fail(failure);
                    if (activeScope.observerFailure().isPresent()) {
                        executionObserverFailureCount.incrementAndGet();
                    }
                }
            } finally {
                activeCount.decrementAndGet();
            }
            return Multi.createFrom().failure(failure);
        }
    }

    @Override
    public Uni<EmbeddingResponse> executeEmbedding(String modelId, EmbeddingRequest request) {
        return Uni.createFrom().deferred(() -> {
            activeCount.incrementAndGet();
            totalCount.incrementAndGet();
            try {
                GgufBackend backend = backendFor(modelId);
                if (executionObserver == ExecutionObserver.NOOP) {
                    return backend.embed(request)
                            .onFailure().invoke(failure -> failedCount.incrementAndGet())
                            .onTermination().invoke(() -> activeCount.decrementAndGet());
                }
                return observeEmbedding(backend, () -> backend.embed(request));
            } catch (Exception error) {
                failedCount.incrementAndGet();
                activeCount.decrementAndGet();
                return Uni.createFrom().failure(error);
            } catch (Error fatal) {
                activeCount.decrementAndGet();
                throw fatal;
            }
        });
    }

    /**
     * Attaches one lifecycle scope to each embedding subscription. The backend
     * supplier is invoked only after subscription to the deferred Uni.
     */
    private Uni<EmbeddingResponse> observeEmbedding(
            GgufBackend backend,
            Supplier<Uni<EmbeddingResponse>> embeddingSupplier) {
        ExecutionScope scope = null;
        if (executionObserver != ExecutionObserver.NOOP) {
            DiagnosticContext context = DiagnosticContext.builder()
                    .correlationId(UUID.randomUUID().toString())
                    .operation("gguf.embed")
                    .component("gguf")
                    .provider(backend.name())
                    .build();
            scope = ExecutionScope.start(context, executionObserver);
        }
        final ExecutionScope activeScope = scope;
        try {
            Uni<EmbeddingResponse> embedding =
                    Objects.requireNonNull(embeddingSupplier.get(), "backend embedding");
            if (activeScope == null) {
                return embedding
                        .onFailure().invoke(failure -> failedCount.incrementAndGet())
                        .onTermination().invoke(() -> activeCount.decrementAndGet());
            }
            return embedding
                    .onItem().invoke(item -> activeScope.succeed())
                    .onFailure().invoke(failure -> {
                        failedCount.incrementAndGet();
                        activeScope.fail(failure);
                    })
                    .onCancellation().invoke(activeScope::close)
                    .onTermination().invoke(() -> {
                        if (activeScope.observerFailure().isPresent()) {
                            executionObserverFailureCount.incrementAndGet();
                        }
                        activeCount.decrementAndGet();
                    });
        } catch (RuntimeException failure) {
            failedCount.incrementAndGet();
            try {
                activeScope.fail(failure);
                if (activeScope.observerFailure().isPresent()) {
                    executionObserverFailureCount.incrementAndGet();
                }
            } finally {
                activeCount.decrementAndGet();
            }
            return Uni.createFrom().failure(failure);
        } catch (Error fatal) {
            activeScope.fail(fatal);
            if (activeScope.observerFailure().isPresent()) {
                executionObserverFailureCount.incrementAndGet();
            }
            throw fatal;
        }
    }

    @Override
    public Uni<String> submitAsyncJob(InferenceRequest request) {
        String jobId = UUID.randomUUID().toString();
        return Uni.createFrom().item(jobId);
    }

    @Override
    public void initialize() {
        this.initialized = true;
    }

    @Override
    public boolean isHealthy() {
        return initialized && !backends.isEmpty();
    }

    @Override
    public HealthStatus health() {
        return isHealthy() ? HealthStatus.healthy() : HealthStatus.unhealthy("Engine not healthy");
    }

    @Override
    public EngineStats getStats() {
        return new EngineStats(
                activeCount.get(),
                totalCount.get(),
                failedCount.get(),
                0.0d,
                isHealthy() ? "HEALTHY" : "UNHEALTHY"
        );
    }

    @Override
    public void shutdown() {
        close();
    }

    @Override
    public synchronized void close() {
        // Detach the complete set before invoking provider code. This makes
        // shutdown idempotent and prevents a failing close from leaving models
        // registered or causing already-attempted backends to be closed twice.
        List<GgufBackend> closing = new ArrayList<>(backends.values());
        backends.clear();
        initialized = false;

        Throwable firstFailure = null;
        for (GgufBackend backend : closing) {
            try {
                if (executionObserver == ExecutionObserver.NOOP) {
                    backend.close();
                } else {
                    observeLifecycleUnchecked("gguf.model.unload", backend.name(), backend::close);
                }
            } catch (RuntimeException | Error failure) {
                if (firstFailure == null) {
                    firstFailure = failure;
                } else if (firstFailure != failure) {
                    firstFailure.addSuppressed(failure);
                }
            }
        }

        if (firstFailure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (firstFailure instanceof Error fatalFailure) {
            throw fatalFailure;
        }
    }

    private GgufBackend backendFor(String modelId) {
        GgufBackend backend = backends.get(modelId);
        if (backend == null) {
            if (backends.size() == 1) {
                return backends.values().iterator().next();
            }
            throw new InferenceException(
                    "No GGUF model loaded with id: " + modelId + ". Loaded models: " + backends.keySet());
        }
        return backend;
    }

    /**
     * Measures the provider's cheap model-readiness probe independently from
     * the later backend construction/load lifecycle. Provider decisions remain
     * on the dedicated decision observer.
     */
    private GgufBackendCapability probeProvider(
            GgufBackendProvider provider, Path modelPath) throws Exception {
        if (executionObserver == ExecutionObserver.NOOP) {
            return provider.probe(modelPath);
        }
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId(UUID.randomUUID().toString())
                .operation("gguf.provider.probe")
                .component("gguf")
                .provider(provider.id())
                .build();
        ExecutionScope scope = ExecutionScope.start(context, executionObserver);
        try (scope) {
            try {
                GgufBackendCapability capability = probeProvider(provider, modelPath);
                scope.succeed();
                return capability;
            } catch (Exception | Error failure) {
                scope.fail(failure);
                throw failure;
            }
        } finally {
            if (scope.observerFailure().isPresent()) {
                executionObserverFailureCount.incrementAndGet();
            }
        }
    }

    private GgufBackendProvider selectProvider(Path modelPath, GgufBackendSelection selection) {
        if (selection.explicit()) {
            for (GgufBackendProvider provider : providers) {
                if (!matches(provider, selection.normalizedValue())) {
                    continue;
                }
                if (!provider.isAvailable()) {
                    observe(provider, ProviderDecisionEvent.Decision.UNAVAILABLE,
                            "explicitly requested provider is unavailable",
                            ProviderDecisionEvent.SelectionMode.EXPLICIT);
                    throw new InferenceException(
                            "Requested GGUF backend '" + selection.requestedValue()
                                    + "' is not available. Available: " + availableIds());
                }
                observe(provider, ProviderDecisionEvent.Decision.SELECTED,
                        "explicit provider selection; readiness probe intentionally bypassed",
                        ProviderDecisionEvent.SelectionMode.EXPLICIT);
                return provider;
            }
            observe(selection.normalizedValue(), 0, ProviderDecisionEvent.Decision.REJECTED,
                    "no registered provider matches the explicit request",
                    ProviderDecisionEvent.SelectionMode.EXPLICIT);
            throw new InferenceException(
                    "Requested GGUF backend '" + selection.requestedValue()
                            + "' is not available. Available: " + availableIds());
        }

        List<String> readiness = new ArrayList<>();
        int skipped = 0;
        for (GgufBackendProvider provider : providers) {
            if (!provider.isAvailable()) {
                readiness.add(provider.id() + ": unavailable");
                observe(provider, ProviderDecisionEvent.Decision.UNAVAILABLE,
                        "provider reports unavailable", ProviderDecisionEvent.SelectionMode.AUTO);
                skipped++;
                continue;
            }
            try {
                GgufBackendCapability capability = provider.probe(modelPath);
                if (capability != null && capability.generationReady()) {
                    String reason = skipped == 0
                            ? "provider is generation-ready and has the highest available priority"
                            : "provider is generation-ready; selected after " + skipped
                                    + " higher-priority candidate(s) were skipped";
                    observe(provider, ProviderDecisionEvent.Decision.SELECTED,
                            reason, ProviderDecisionEvent.SelectionMode.AUTO);
                    return provider;
                }
                String reason = capability == null
                        ? "probe returned no capability"
                        : capability.reason() == null || capability.reason().isBlank()
                                ? "generation is not ready"
                                : capability.reason();
                readiness.add(provider.id() + ": " + reason);
                observe(provider, ProviderDecisionEvent.Decision.NOT_READY,
                        "provider is not generation-ready", ProviderDecisionEvent.SelectionMode.AUTO);
                skipped++;
            } catch (Exception error) {
                readiness.add(provider.id() + ": probe failed (" + error.getClass().getSimpleName() + ")");
                observe(provider, ProviderDecisionEvent.Decision.PROBE_FAILED,
                        "probe threw " + error.getClass().getSimpleName(),
                        ProviderDecisionEvent.SelectionMode.AUTO);
                skipped++;
            }
        }

        throw new InferenceException(
                "No GGUF backend reports generation readiness for model: " + modelPath
                        + ". Provider readiness: " + String.join("; ", readiness)
                        + ". Select a provider explicitly only for deliberate testing, or install a compatible backend.");
    }

    private void observe(
            GgufBackendProvider provider,
            ProviderDecisionEvent.Decision decision,
            String reason,
            ProviderDecisionEvent.SelectionMode mode) {
        observe(provider.id(), provider.priority(), decision, reason, mode);
    }

    private void observe(
            String providerId,
            int priority,
            ProviderDecisionEvent.Decision decision,
            String reason,
            ProviderDecisionEvent.SelectionMode mode) {
        if (decisionObserver == ProviderDecisionObserver.NOOP) {
            return;
        }
        try {
            decisionObserver.onDecision(new ProviderDecisionEvent(
                    providerId, decision, reason, priority, mode, java.time.Instant.now()));
        } catch (RuntimeException ignored) {
            // Provider diagnostics must never change backend selection behavior.
            decisionObserverFailureCount.incrementAndGet();
        }
    }

    private static boolean matches(GgufBackendProvider provider, String normalizedValue) {
        return provider.id().equalsIgnoreCase(normalizedValue) || provider.aliases().contains(normalizedValue);
    }

    private String availableIds() {
        return providers.stream()
                .filter(GgufBackendProvider::isAvailable)
                .map(GgufBackendProvider::id)
                .collect(Collectors.joining(", "));
    }
}
