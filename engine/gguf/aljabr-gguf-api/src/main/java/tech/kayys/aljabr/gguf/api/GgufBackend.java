package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.aljabr.execution.ModelExecutionContext;
import tech.kayys.aljabr.spi.embedding.EmbeddingRequest;
import tech.kayys.aljabr.spi.embedding.EmbeddingResponse;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;
import tech.kayys.aljabr.spi.inference.StreamingInferenceChunk;

import java.util.Map;

/**
 * Per-model execution contract implemented by every GGUF backend.
 *
 * <p>Backends are stateful (they hold file handles, model weights, caches, or
 * native memory): they are created once per model load and released via
 * {@link #close()} when the model is unloaded.</p>
 *
 * <p>Implementations should override {@link #stream(InferenceRequest)} and
 * {@link #embed(EmbeddingRequest)} when the underlying runtime supports those
 * operations. The defaults preserve compatibility for synchronous
 * generation-only backends.</p>
 */
public interface GgufBackend extends AutoCloseable {

    /** Short stable identifier, e.g. {@code "java"} or {@code "llamacpp"}. */
    String name();

    /** Diagnostic key/value pairs surfaced in metadata. */
    default Map<String, Object> metadata() {
        return Map.of("backend", name());
    }

    /** Execute inference request against this backend. */
    InferenceResponse execute(InferenceRequest request);

    /**
     * Execute inference with request-scoped model execution state.
     *
     * <p>The default preserves compatibility with existing backends. Backends
     * using Aljabr-managed KV storage may override this method and read the
     * non-owning storage binding from the context.</p>
     */
    default InferenceResponse execute(
            InferenceRequest request, ModelExecutionContext context) {
        requireNoExternalKVCache(context);
        return execute(request);
    }

    /**
     * Rejects caller-managed KV cache storage when using default delegation.
     * Backends supporting external storage must override the context-aware methods.
     */
    default void requireNoExternalKVCache(ModelExecutionContext context) {
        java.util.Objects.requireNonNull(context, "context");
        if (context.kvCacheStorage().isPresent()) {
            throw new UnsupportedOperationException(
                    "GGUF backend '" + name() + "' does not support external KV cache storage; "
                            + "backends consuming context with KV cache storage must override execute/stream with ModelExecutionContext");
        }
    }

    /**
     * Stream inference output. The default adapts a synchronous response into
     * one final chunk; backends with token streaming should override this.
     */
    default Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().item(() -> {
            InferenceResponse response = execute(request);
            String content = response == null || response.getContent() == null ? "" : response.getContent();
            return StreamingInferenceChunk.finalChunk(request.getRequestId(), 0, content);
        });
    }

    /**
     * Stream inference with request-scoped model execution state.
     *
     * <p>The default preserves compatibility with existing backends. A backend
     * that consumes the context during streaming should override this method.</p>
     */
    default Multi<StreamingInferenceChunk> stream(
            InferenceRequest request, ModelExecutionContext context) {
        requireNoExternalKVCache(context);
        return stream(request);
    }

    /**
     * Generate embeddings. Backends that do not implement embeddings fail
     * explicitly rather than returning fabricated vectors.
     */
    default Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
        return Uni.createFrom().failure(
                new UnsupportedOperationException("Embeddings are not supported by GGUF backend: " + name()));
    }

    @Override
    void close();
}
