package tech.kayys.aljabr.gguf.llamacppbackend;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.gguf.api.GgufBackend;
import tech.kayys.aljabr.gguf.llamacpp.LlamaCppRunner;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

/**
 * GGUF backend powered by llama.cpp native bindings.
 *
 * <p>Delegates to {@link LlamaCppRunner}, which owns the FFM bindings into
 * {@code libllama.dylib / libllama.so}. The runner is fully initialised by
 * {@link LlamaCppGgufBackendProvider} before this backend is returned.</p>
 */
final class LlamaCppGgufBackend implements GgufBackend {

    private static final Logger log = Logger.getLogger(LlamaCppGgufBackend.class);

    private final LlamaCppRunner runner;

    LlamaCppGgufBackend(LlamaCppRunner runner) {
        this.runner = runner;
    }

    @Override
    public String name() {
        return "llamacpp";
    }

    @Override
    public InferenceResponse execute(InferenceRequest request) {
        try {
            return runner.infer(request);
        } catch (Exception e) {
            throw new tech.kayys.aljabr.spi.exception.InferenceException("Llama.cpp GGUF inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        try {
            runner.close();
        } catch (Exception e) {
            log.warnf(e, "Error closing LlamaCppRunner");
        }
    }
}
