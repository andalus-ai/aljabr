package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;
import tech.kayys.aljabr.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

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
    void registersGemma3DecoderAndHuggingFaceBpeTokenizer() {
        FunctionGemmaModelFamilyPlugin plugin = new FunctionGemmaModelFamilyPlugin();

        assertEquals("functiongemma", plugin.descriptor().id());
        assertEquals(List.of("functiongemma"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Gemma3ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("google/functiongemma-270m-it",
                plugin.descriptor().metadata().get("default_checkpoint"));
        assertEquals(List.of("functiongemma-gemma-bpe"), plugin.tokenizerDescriptors().stream()
                .map(d -> d.id()).toList());
        assertEquals(ModelTokenizerKind.HUGGING_FACE_BPE,
                plugin.tokenizerDescriptors().getFirst().kind());
        assertEquals("functiongemma", plugin.architectureAdapters().getFirst().id());
        assertTrue(plugin.architectureAdapters().getFirst() instanceof FunctionGemmaArchitecture);
    }

    @Test
    void recognizesTheUpstreamGemma3TextCheckpointByFunctionGemmaTokenizerProtocol() throws Exception {
        Path fixture = Path.of(Objects.requireNonNull(
                getClass().getResource("/model-family-fixtures/functiongemma")).toURI());
        FunctionGemmaModelFamilyPlugin plugin = new FunctionGemmaModelFamilyPlugin();

        assertTrue(plugin.matchesCheckpoint(fixture));

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);
        assertEquals("functiongemma", registry.pluginForCheckpoint(fixture).orElseThrow().descriptor().id());
        assertEquals("functiongemma", registry.pluginsForCheckpoint(fixture).getFirst().descriptor().id());
    }

    @Test
    void doesNotClaimOrdinaryGemma3TextCheckpointWithoutFunctionTokens() throws Exception {
        Path fixture = Path.of(Objects.requireNonNull(
                getClass().getResource("/model-family-fixtures/gemma3_text")).toURI());
        assertFalse(new FunctionGemmaModelFamilyPlugin().matchesCheckpoint(fixture));
    }

    @Test
    void functionGemmaArchitectureReusesGemma3TextWeightLayoutAndRuntimeTraits() {
        FunctionGemmaArchitecture architecture = new FunctionGemmaArchitecture();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.layers.17.self_attn.q_proj.weight", architecture.layerQueryWeight(17));
        assertEquals("model.layers.17.self_attn.k_proj.weight", architecture.layerKeyWeight(17));
        assertEquals("model.layers.17.self_attn.v_proj.weight", architecture.layerValueWeight(17));
        assertEquals("model.layers.17.self_attn.o_proj.weight", architecture.layerOutputWeight(17));
        assertEquals("model.layers.17.mlp.gate_proj.weight", architecture.layerFfnGateWeight(17));
        assertEquals("model.layers.17.mlp.up_proj.weight", architecture.layerFfnUpWeight(17));
        assertEquals("model.layers.17.mlp.down_proj.weight", architecture.layerFfnDownWeight(17));
        assertTrue(architecture.hasTiedEmbeddings());
        assertEquals((float) Math.sqrt(640), architecture.embeddingScaleFactor(640), 0.0001f);
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);
        assertTrue(traits.gemma3Text());
        assertTrue(traits.attention().splitHalfRope());
    }

    @Test
    void formatsDeveloperUserAssistantAndStandaloneFunctionResponse() {
        FunctionGemmaChatTemplate template = new FunctionGemmaChatTemplate();

        String developer = template.formatDeveloperContent(
                "You can call the available functions.",
                List.of("set_light{enabled:<escape>BOOLEAN<escape>}"));
        String prompt = template.format(List.of(
                Message.system(developer),
                Message.user("Turn on the light."),
                Message.assistant("<start_function_call>call:set_light{enabled:true}<end_function_call>"),
                new Message(Message.Role.TOOL, "response:set_light{ok:true}")));

        assertTrue(template.supports("functiongemma"));
        assertFalse(template.supports("gemma3"));
        assertTrue(prompt.startsWith("<start_of_turn>developer\n"));
        assertTrue(prompt.contains("<start_function_declaration>set_light{enabled:<escape>BOOLEAN<escape>}<end_function_declaration>"));
        assertTrue(prompt.contains("<start_of_turn>user\nTurn on the light.<end_of_turn>"));
        assertTrue(prompt.contains("<start_function_call>call:set_light{enabled:true}<end_function_call>"));
        assertTrue(prompt.contains("<start_function_response>response:set_light{ok:true}<end_function_response>"));
        assertFalse(prompt.contains("<start_of_turn>model\n<start_function_response>"));
        assertTrue(prompt.endsWith("<start_of_turn>model\n"));
    }
}
