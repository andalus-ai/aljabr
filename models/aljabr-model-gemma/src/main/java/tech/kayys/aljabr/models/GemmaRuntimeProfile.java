package tech.kayys.tafkir.models;

import tech.kayys.aljabr.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelPromptTraits;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits.AttentionRuntimeTraits;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import java.util.Locale;

/**
 * Legacy Gemma runtime policy, including the Gemma4-compatible direct path.
 */
public final class GemmaRuntimeProfile {

    private static final boolean GEMMA3_TEXT = false;
    private static final boolean QWEN_TEXT = false;
    private static final boolean GEMMA_FAMILY = true;

    private GemmaRuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        boolean nativeBf16Matvec = isGemma4Text(config);
        boolean perLayerInputs = perLayerInputPath(config);
        ModelPromptTraits prompt = prompt(config);
        return ModelRuntimeTraits.builder()
                .nativeBf16Matvec(nativeBf16Matvec)
                .perLayerInputPath(perLayerInputs)
                .prompt(prompt)
                .attention(nativeBf16Matvec
                        ? new AttentionRuntimeTraits(true, true, true, true, true, true, true, 0, false, false, false)
                        : ModelAttentionTraitsPolicy.generic(config, perLayerInputs))
                .build();
    }

    public static ModelPromptTraits prompt(ModelConfig config) {
        boolean nativeBf16Matvec = isGemma4Text(config);
        PromptBosPolicy promptBosPolicy = nativeBf16Matvec ? PromptBosPolicy.NEVER : (GEMMA3_TEXT || GEMMA_FAMILY ? PromptBosPolicy.TURN_AWARE : PromptBosPolicy.DEFAULT);
        return new ModelPromptTraits(
                promptBosPolicy,
                nativeBf16Matvec ? java.util.Set.of("<|channel>", "<channel|>", "<|think|>") : java.util.Collections.<String>emptySet(),
                nativeBf16Matvec,
                nativeBf16Matvec,
                nativeBf16Matvec,
                QWEN_TEXT ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." : ModelPromptTraits.DEFAULT_SYSTEM_PROMPT,
                nativeBf16Matvec ? java.util.Set.of("<|turn>") : (GEMMA3_TEXT || GEMMA_FAMILY ? java.util.Set.of("<start_of_turn>") : java.util.Collections.<String>emptySet()),
                GEMMA3_TEXT || QWEN_TEXT);
    }

    static boolean isGemma4Text(ModelConfig config) {
        return normalizedModelType(config).startsWith("gemma4");
    }

    static boolean perLayerInputPath(ModelConfig config) {
        return config != null
                && (config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0);
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }
}
