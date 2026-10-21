package tech.kayys.aljabr.spi.model;

/**
 * Functional interface for executing multimodal requests.
 * Allows decoupling high-level builders from AljabrSdk or specific engine
 * implementations.
 * @author bhangun
 */
@FunctionalInterface
public interface MultimodalProcessor {
    /**
     * Process a multimodal request.
     *
     * @param request the request
     * @return the response
     */
    MultimodalResponse processMultimodal(MultimodalRequest request);
}
