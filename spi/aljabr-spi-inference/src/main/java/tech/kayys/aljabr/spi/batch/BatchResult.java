package tech.kayys.aljabr.spi.batch;

import tech.kayys.aljabr.error.ErrorPayload;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

/**
 * Represents an individual result within a batch execution.
 */
public record BatchResult(
        String requestId,
        InferenceResponse response,
        ErrorPayload error) {

    public boolean succeeded() {
        return error == null && response != null;
    }
}
