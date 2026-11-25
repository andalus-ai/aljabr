package tech.kayys.aljabr.diffusion.tokenizer;

import java.util.Objects;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/**
 * Tokenized prompt result. Token IDs are defensively copied on input and
 * output so callers cannot mutate the prepared representation accidentally.
 */
public final class PromptTokenizationData implements DiffusionConditioningData {

    private final String inputId;
    private final long[] tokenIds;

    public PromptTokenizationData(String inputId, long[] tokenIds) {
        this.inputId = Objects.requireNonNull(inputId, "inputId");
        if (inputId.isBlank()) {
            throw new IllegalArgumentException("inputId must not be blank");
        }
        this.tokenIds = Objects.requireNonNull(tokenIds, "tokenIds").clone();
    }

    @Override
    public String inputId() {
        return inputId;
    }

    @Override
    public DiffusionConditioningInput.Modality modality() {
        return DiffusionConditioningInput.Modality.PROMPT;
    }

    public long[] tokenIds() {
        return tokenIds.clone();
    }

    /** Token IDs are heap-owned; no external resource needs releasing. */
    @Override
    public void close() {
        // Intentionally empty.
    }
}
