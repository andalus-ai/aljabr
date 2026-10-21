package tech.kayys.aljabr.gguf.api;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * SPI for a GGUF execution backend.
 *
 * <p>Implementations are discovered at runtime via {@link java.util.ServiceLoader}:
 * a backend module registers itself with a
 * {@code META-INF/services/tech.kayys.aljabr.gguf.api.GgufBackendProvider}
 * file listing its provider class.</p>
 */
public interface GgufBackendProvider {

    /** Stable id used for explicit selection, e.g. {@code "java"} or {@code "llamacpp"}. */
    String id();

    /** Other tokens (case-insensitive) that should resolve to this provider. */
    default Set<String> aliases() {
        return Set.of();
    }

    /**
     * Whether this backend can run at all in this JVM/deployment — e.g. the
     * llama.cpp provider checks that its native bridge actually resolves.
     */
    boolean isAvailable();

    /**
     * Cheap, header-only readiness check for one specific model file. Must
     * not fully load tensor data.
     */
    GgufBackendCapability probe(Path modelPath) throws Exception;

    /** Fully instantiate a backend for this model. */
    GgufBackend create(Path modelPath, Map<String, Object> config) throws Exception;

    default GgufBackend create(Path modelPath) throws Exception {
        return create(modelPath, Map.of());
    }

    /**
     * Tie-breaker when more than one available provider reports
     * {@code generationReady} for the same model in AUTO mode. Higher wins.
     */
    default int priority() {
        return 0;
    }
}
