package tech.kayys.aljabr.diffusion.api;

import java.util.Objects;
import java.util.Set;

/**
 * Extension point for preparing request conditioning inputs.
 *
 * <p>Implementations are modality/source-specific adapters. For example,
 * separate integrations may decode image references, sample video frames,
 * or tokenize/encode prompts. The API does not require any particular
 * preprocessing library or execution backend.</p>
 *
 * <p>{@link #supports(DiffusionConditioningInput)} must be side-effect free.
 * The returned data is caller-owned and must remain valid until closed.</p>
 */
public interface DiffusionConditioningPreprocessor {

    /** Modalities this adapter is able to prepare. */
    Set<DiffusionConditioningInput.Modality> modalities();

    /**
     * Whether this adapter can prepare the supplied input, including its
     * source scheme or metadata when relevant.
     */
    default boolean supports(DiffusionConditioningInput input) {
        return input != null && modalities().contains(input.modality());
    }

    /**
     * Prepare one input and return an owned, closeable representation.
     *
     * @throws DiffusionConditioningPreparationException if preparation fails
     */
    DiffusionConditioningData prepare(
            DiffusionConditioningInput input,
            DiffusionConditioningPreparationContext context)
            throws DiffusionConditioningPreparationException;

    /**
     * Stable diagnostic identifier for this adapter.
     */
    default String id() {
        return getClass().getName();
    }
}
