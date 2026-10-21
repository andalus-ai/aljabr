package tech.kayys.tafkir.models;

import tech.kayys.aljabr.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelPromptTraits;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

/**
 * Phi-specific runtime policy for prompt and packed-QKV attention behavior.
 */
public final class PhiRuntimeProfile {

    public static ModelRuntimeTraits text(ModelConfig config) {
        return ModelRuntimeTraits.builder()
                .prompt(prompt())
                .attention(new ModelRuntimeTraits.AttentionRuntimeTraits(false, false, false, false, false, false, false, 0, false, ModelAttentionTraitsPolicy.isLargeAttentionMatvecCandidate(config, false, false), true))
                .build();
    }

    public static ModelPromptTraits prompt() {
        return new ModelPromptTraits(
                ModelRuntimeTraits.PromptBosPolicy.DEFAULT,
                java.util.Collections.<String>emptySet(),
                false,
                false,
                false,
                ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT,
                java.util.Collections.<String>emptySet(),
                false);
    }
}
