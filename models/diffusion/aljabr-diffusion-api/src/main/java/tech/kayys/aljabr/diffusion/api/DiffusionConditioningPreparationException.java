package tech.kayys.aljabr.diffusion.api;

/**
 * Failure to prepare a caller-supplied conditioning input.
 */
public class DiffusionConditioningPreparationException extends RuntimeException {

    private final String inputId;
    private final DiffusionConditioningInput.Modality modality;

    public DiffusionConditioningPreparationException(
            DiffusionConditioningInput input, String message, Throwable cause) {
        super(message, cause);
        this.inputId = input == null ? null : input.id();
        this.modality = input == null ? null : input.modality();
    }

    public DiffusionConditioningPreparationException(
            DiffusionConditioningInput input, String message) {
        this(input, message, null);
    }

    public String inputId() {
        return inputId;
    }

    public DiffusionConditioningInput.Modality modality() {
        return modality;
    }
}
