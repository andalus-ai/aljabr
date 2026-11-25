package tech.kayys.aljabr.diffusion.tokenizer;

import java.util.Objects;
import java.util.Set;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreprocessor;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

/**
 * Reusable prompt-to-token-ID adapter over Aljabr's tokenizer SPI.
 *
 * <p>Prompt source resolution is injected separately, keeping this adapter
 * independent of URI schemes, prompt stores, and model-family conventions.
 * It produces token IDs only; text-encoder execution and embedding creation
 * remain separate capabilities.</p>
 */
public final class TokenizerPromptConditioningPreprocessor
        implements DiffusionConditioningPreprocessor {

    private static final Set<DiffusionConditioningInput.Modality> MODALITIES =
            Set.of(DiffusionConditioningInput.Modality.PROMPT);

    private final Tokenizer tokenizer;
    private final PromptTextResolver textResolver;
    private final PromptTokenizationOptions options;

    public TokenizerPromptConditioningPreprocessor(
            Tokenizer tokenizer,
            PromptTextResolver textResolver) {
        this(tokenizer, textResolver, PromptTokenizationOptions.defaults());
    }

    public TokenizerPromptConditioningPreprocessor(
            Tokenizer tokenizer,
            PromptTextResolver textResolver,
            PromptTokenizationOptions options) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.textResolver = Objects.requireNonNull(textResolver, "textResolver");
        this.options = Objects.requireNonNull(options, "options");
    }

    @Override
    public String id() {
        return "aljabr.tokenizer.prompt";
    }

    @Override
    public Set<DiffusionConditioningInput.Modality> modalities() {
        return MODALITIES;
    }

    @Override
    public DiffusionConditioningData prepare(
            DiffusionConditioningInput input,
            DiffusionConditioningPreparationContext context) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(context, "context");
        if (!supports(input)) {
            throw new DiffusionConditioningPreparationException(
                    input, "Prompt tokenizer cannot prepare modality " + input.modality());
        }

        try {
            String prompt = Objects.requireNonNull(
                    textResolver.resolve(input), "resolved prompt text");
            if (prompt.isBlank()) {
                throw new IllegalArgumentException("resolved prompt text must not be blank");
            }
            long[] tokenIds = Objects.requireNonNull(
                    tokenizer.encode(prompt, EncodeOptions.builder()
                            .addBos(options.addBos())
                            .addEos(options.addEos())
                            .build()),
                    "tokenizer returned null token IDs");
            return new PromptTokenizationData(input.id(), tokenIds);
        } catch (Exception failure) {
            if (failure instanceof DiffusionConditioningPreparationException preparationFailure) {
                throw preparationFailure;
            }
            throw new DiffusionConditioningPreparationException(
                    input, "Failed to tokenize prompt input '" + input.id() + "'", failure);
        }
    }
}
