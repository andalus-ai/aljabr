package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.models.gemma3.Gemma3TextFamily;

import java.util.List;

/**
 * FunctionGemma uses the Gemma 3 text decoder architecture with a
 * FunctionGemma-specific tokenizer and conversation protocol.
 */
public final class FunctionGemmaArchitecture extends Gemma3TextFamily {

    @Override
    public String id() {
        return "functiongemma";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma3ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("functiongemma");
    }
}
