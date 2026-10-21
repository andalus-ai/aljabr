package tech.kayys.aljabr.models.gemma4;

import tech.kayys.aljabr.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelPromptTraits;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.util.Locale;
import java.util.Set;

/**
 * Gemma 4 text runtime policy for prompt/control-token and attention behavior.
 */
public final class Gemma4RuntimeProfile {

    public static final Set<String> GEMMA4_CONTROL_TOKEN_TEXTS = Set.of("<|channel>", "<channel|>", "<|think|>", "<|turn>", "<turn|>");

    private Gemma4RuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        boolean nativeBf16Matvec = isGemma4Text(config);
        boolean perLayerInputs = perLayerInputPath(config);
        ModelPromptTraits prompt = prompt(config);
        return ModelRuntimeTraits.builder()
                .nativeBf16Matvec(nativeBf16Matvec)
                .geluGatedFfn(nativeBf16Matvec)
                .perLayerInputEmbedding(nativeBf16Matvec && perLayerInputs)
                .perLayerInputPath(perLayerInputs)
                .prompt(prompt)
                .attention(nativeBf16Matvec
                        ? new ModelRuntimeTraits.AttentionRuntimeTraits(true, true, true, true, true, true, true, 0, false, false, false)
                        : ModelAttentionTraitsPolicy.generic(config, perLayerInputs))
                .build();
    }

    public static ModelPromptTraits prompt(ModelConfig config) {
        boolean isGemma4 = isGemma4Text(config);
        // When no config is supplied (e.g. during capability-contract probing) default to
        // TURN_AWARE so callers can rely on the trait without a concrete config object.
        // When a real config is present and identifies as gemma4, use NEVER (the Gemma4
        // native behaviour). For any other config fall back to DEFAULT.
        ModelRuntimeTraits.PromptBosPolicy bosPolicy;
        if (config == null) {
            bosPolicy = ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE;
        } else if (isGemma4) {
            bosPolicy = ModelRuntimeTraits.PromptBosPolicy.NEVER;
        } else {
            bosPolicy = ModelRuntimeTraits.PromptBosPolicy.DEFAULT;
        }
        return new ModelPromptTraits(
                bosPolicy,
                isGemma4 ? GEMMA4_CONTROL_TOKEN_TEXTS : Set.of(),
                isGemma4,
                isGemma4,
                false,
                ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT,
                java.util.Collections.<String>emptySet(),
                false);
    }

    static boolean isGemma4Text(ModelConfig config) {
        return normalizedModelType(config).startsWith("gemma4");
    }

    static boolean perLayerInputPath(ModelConfig config) {
        // Gemma 4 unified 12B may expose vocab_size_per_layer_input without
        // shipping PLE tensors. The hidden size is the executable PLE signal.
        return config != null && config.hiddenSizePerLayerInput() > 0;
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }
}
