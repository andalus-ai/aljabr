package tech.kayys.aljabr.diffusion.tokenizer;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/**
 * Resolves an opaque prompt input reference to its text.
 *
 * <p>Applications can provide resolvers for inline text, application-managed
 * prompt stores, or other source schemes without teaching the tokenizer
 * adapter about storage or transport.</p>
 */
@FunctionalInterface
public interface PromptTextResolver {

    String resolve(DiffusionConditioningInput input) throws Exception;
}
