package tech.kayys.aljabr.gguf.api;

import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

import java.util.Map;

/**
 * Execution contract that every GGUF backend must honour.
 *
 * <p>Backends are stateful (they hold file handles / native memory): they are
 * created once per model load and released via {@link #close} when
 * the model is unloaded.</p>
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

    @Override
    void close();
}
