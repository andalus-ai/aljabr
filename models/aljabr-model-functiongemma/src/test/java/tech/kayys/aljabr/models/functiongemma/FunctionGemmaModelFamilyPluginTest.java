package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaModelFamilyPluginTest {

    @Test
    void familyPluginSatisfiesSharedContract() {
        List<ModelFamilyContractViolation> violations =
                ModelFamilyContractValidator.validate(new FunctionGemmaModelFamilyPlugin());

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary).toList());
    }

    @Test
    void registersGemma3DecoderUnderFunctionGemmaIdentity() {
        FunctionGemmaModelFamilyPlugin plugin = new FunctionGemmaModelFamilyPlugin();

        assertEquals("functiongemma", plugin.descriptor().id());
        assertEquals(List.of("functiongemma"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Gemma3ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("google/functiongemma-270m-it",
                plugin.descriptor().metadata().get("default_checkpoint"));
        assertEquals(List.of("functiongemma-gemma-bpe"), plugin.tokenizerDescriptors().stream()
                .map(d -> d.id()).toList());
        assertEquals("functiongemma", plugin.architectureAdapters().getFirst().id());
        assertTrue(plugin.architectureAdapters().getFirst() instanceof FunctionGemmaArchitecture);
    }

    @Test
    void formatsDeveloperUserAssistantAndFunctionResponseTurns() {
        FunctionGemmaChatTemplate template = new FunctionGemmaChatTemplate();

        String prompt = template.format(List.of(
                Message.system("You can call the available functions."),
                Message.user("Turn on the light."),
                Message.assistant("<start_function_call>call:set_light{enabled:true}<end_function_call>"),
                new Message(Message.Role.TOOL, "done")));

        assertTrue(template.supports("functiongemma"));
        assertFalse(template.supports("gemma3"));
        assertTrue(prompt.startsWith("<start_of_turn>developer\n"));
        assertTrue(prompt.contains("<start_of_turn>user\nTurn on the light.<end_of_turn>"));
        assertTrue(prompt.contains("<start_function_call>call:set_light{enabled:true}<end_function_call>"));
        assertTrue(prompt.contains("<start_function_response>done<end_function_response>"));
        assertTrue(prompt.endsWith("<start_of_turn>model\n"));
    }
}
