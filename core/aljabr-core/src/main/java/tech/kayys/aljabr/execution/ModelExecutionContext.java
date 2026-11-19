package tech.kayys.aljabr.execution;

import tech.kayys.aljabr.cache.KVCacheStorage;

import java.util.Objects;
import java.util.Optional;

/**
 * Immutable, request-scoped inputs for a model execution.
 *
 * <p>The context may expose an Aljabr-managed KV storage implementation when
 * the model execution path uses the shared cache contract. Runtimes that own
 * their cache state (for example, an ONNX or LiteRT execution provider) may
 * leave that binding absent and retain their state in their native execution
 * context.</p>
 *
 * <p>This context is a non-owning view: closing the context is not supported
 * and it never closes, clears, advances, or otherwise manages the supplied
 * storage. Cache lifecycle and admission policy remain with the caller.</p>
 */
public final class ModelExecutionContext {

    private final String modelId;
    private final String requestId;
    private final KVCacheStorage kvCacheStorage;

    /**
     * Creates an execution context.
     *
     * @param modelId stable model identifier for this execution
     * @param requestId request identifier for this execution
     * @param kvCacheStorage optional Aljabr-managed KV storage; {@code null}
     *                       when the runtime manages cache state itself or
     *                       the model does not use a KV cache
     */
    public ModelExecutionContext(
            String modelId,
            String requestId,
            KVCacheStorage kvCacheStorage) {
        this.modelId = requireText(modelId, "modelId");
        this.requestId = requireText(requestId, "requestId");
        this.kvCacheStorage = kvCacheStorage;
    }

    /** Creates a context without an Aljabr-managed KV storage binding. */
    public static ModelExecutionContext withoutKVCache(String modelId, String requestId) {
        return new ModelExecutionContext(modelId, requestId, null);
    }

    /** Creates a context bound to caller-managed, format-neutral KV storage. */
    public static ModelExecutionContext withKVCacheStorage(
            String modelId, String requestId, KVCacheStorage storage) {
        return new ModelExecutionContext(
                modelId, requestId, Objects.requireNonNull(storage, "storage"));
    }

    public String modelId() {
        return modelId;
    }

    public String requestId() {
        return requestId;
    }

    /**
     * Returns the shared KV storage binding when this execution uses
     * Aljabr-managed cache storage.
     *
     * <p>The returned storage remains owned by the caller. This context does
     * not imply sequence commitment, cache allocation, eviction, or cleanup.</p>
     */
    public Optional<KVCacheStorage> kvCacheStorage() {
        return Optional.ofNullable(kvCacheStorage);
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
