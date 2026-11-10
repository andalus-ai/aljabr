package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
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
