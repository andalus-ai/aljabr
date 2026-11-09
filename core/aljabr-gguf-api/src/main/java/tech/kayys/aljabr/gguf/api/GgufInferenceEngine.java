package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
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
    private final AtomicLong activeCount = new AtomicLong(0);
    private final AtomicLong totalCount = new AtomicLong(0);
    private final AtomicLong failedCount = new AtomicLong(0);
    private volatile boolean initialized = false;

    public GgufInferenceEngine() {
        this(loadProviders());
    }

    public GgufInferenceEngine(List<GgufBackendProvider> providers) {
        List<GgufBackendProvider> sorted = new ArrayList<>(providers);
        sorted.sort(Comparator.comparingInt(GgufBackendProvider::priority).reversed());
        this.providers = List.copyOf(sorted);
        this.initialized = true;
    }

    private static List<GgufBackendProvider> loadProviders() {
        List<GgufBackendProvider> found = new ArrayList<>();
        ServiceLoader.load(GgufBackendProvider.class, GgufInferenceEngine.class.getClassLoader())
                .forEach(found::add);
        return found;
    }

    public synchronized GgufBackend loadModel(String modelId, Path modelPath, Map<String, Object> config) throws Exception {
        GgufBackendSelection selection = GgufBackendSelection.resolve(config);
        GgufBackendProvider provider = selectProvider(modelPath, selection);
        GgufBackend backend = provider.create(modelPath, config);
        backends.put(modelId, backend);
        return backend;
    }

    public synchronized void unloadModel(String modelId) {
        GgufBackend removed = backends.remove(modelId);
        if (removed != null) {
            removed.close();
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
            return backendFor(modelId).execute(request);
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
                return backendFor(modelId).stream(request)
                        .onFailure().invoke(failure -> failedCount.incrementAndGet())
                        .onTermination().invoke(() -> activeCount.decrementAndGet());
            } catch (Exception error) {
                failedCount.incrementAndGet();
                activeCount.decrementAndGet();
                return Multi.createFrom().failure(error);
            }
        });
    }

    @Override
    public Uni<EmbeddingResponse> executeEmbedding(String modelId, EmbeddingRequest request) {
        return Uni.createFrom().deferred(() -> {
            activeCount.incrementAndGet();
            totalCount.incrementAndGet();
            try {
                return backendFor(modelId).embed(request)
                        .onFailure().invoke(failure -> failedCount.incrementAndGet())
                        .onTermination().invoke(() -> activeCount.decrementAndGet());
            } catch (Exception error) {
                failedCount.incrementAndGet();
                activeCount.decrementAndGet();
                return Uni.createFrom().failure(error);
            }
        });
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
    public void close() {
        backends.values().forEach(GgufBackend::close);
        backends.clear();
        initialized = false;
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

    private GgufBackendProvider selectProvider(Path modelPath, GgufBackendSelection selection) {
        if (selection.explicit()) {
            return providers.stream()
                    .filter(p -> matches(p, selection.normalizedValue()))
                    .filter(GgufBackendProvider::isAvailable)
                    .findFirst()
                    .orElseThrow(() -> new InferenceException(
                            "Requested GGUF backend '" + selection.requestedValue()
                                    + "' is not available. Available: " + availableIds()));
        }

        List<String> readiness = new ArrayList<>();
        for (GgufBackendProvider provider : providers) {
            if (!provider.isAvailable()) {
                readiness.add(provider.id() + ": unavailable");
                continue;
            }
            try {
                GgufBackendCapability capability = provider.probe(modelPath);
                if (capability != null && capability.generationReady()) {
                    return provider;
                }
                String reason = capability == null
                        ? "probe returned no capability"
                        : capability.reason() == null || capability.reason().isBlank()
                                ? "generation is not ready"
                                : capability.reason();
                readiness.add(provider.id() + ": " + reason);
            } catch (Exception error) {
                readiness.add(provider.id() + ": probe failed (" + error.getClass().getSimpleName()
                        + (error.getMessage() == null ? "" : ": " + error.getMessage()) + ")");
            }
        }

        throw new InferenceException(
                "No GGUF backend reports generation readiness for model: " + modelPath
                        + ". Provider readiness: " + String.join("; ", readiness)
                        + ". Select a provider explicitly only for deliberate testing, or install a compatible backend.");
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
