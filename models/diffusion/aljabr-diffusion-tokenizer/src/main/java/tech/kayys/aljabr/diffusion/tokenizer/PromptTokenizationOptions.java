package tech.kayys.aljabr.diffusion.tokenizer;

/** Immutable tokenizer boundary options for prompt conditioning. */
public record PromptTokenizationOptions(boolean addBos, boolean addEos) {

    public static PromptTokenizationOptions defaults() {
        return new PromptTokenizationOptions(false, false);
    }
}
