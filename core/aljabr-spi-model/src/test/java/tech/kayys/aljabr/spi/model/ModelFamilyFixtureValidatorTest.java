package tech.kayys.aljabr.spi.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyFixtureValidatorTest {

    @TempDir
    Path tempDir;

    @Test
    void validFixtureMatchesDescriptorTokenizerAndDirectAdapter() throws IOException {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "fixture_model",
                  "architectures": ["FixtureForCausalLM"]
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");

        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(validPlugin(), tempDir);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "fixture should satisfy descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void invalidFixtureReportsMachineReadableDriftCodes() throws IOException {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "other_model",
                  "architectures": ["OtherForCausalLM"]
                }
                """);

        Set<String> codes = ModelFamilyFixtureValidator.validate(validPlugin(), tempDir).stream()
                .map(ModelFamilyContractViolation::code)
                .collect(Collectors.toSet());

        assertTrue(codes.contains("fixture_model_type_unclaimed"));
        assertTrue(codes.contains("fixture_architecture_unclaimed"));
        assertTrue(codes.contains("fixture_tokenizer_files_unmatched"));
        assertTrue(codes.contains("fixture_architecture_adapter_unmatched"));
    }

    private static ModelFamilyPlugin validPlugin() {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "fixture_family",
                        "Fixture Family",
                        List.of("fixture_model"),
                        List.of("FixtureForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/fixture"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new FixtureArchitecture());
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("fixture-hf-bpe"));
            }
        };
    }

    private static final class FixtureArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "fixture";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("FixtureForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("fixture_model");
        }

        @Override
        public String embedTokensWeight() {
            return "embed.weight";
        }

        @Override
        public String finalNormWeight() {
            return "norm.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "q.weight";
        }

        @Override
        public String layerKeyWeight(int i) {
            return "k.weight";
        }

        @Override
        public String layerValueWeight(int i) {
            return "v.weight";
        }

        @Override
        public String layerOutputWeight(int i) {
            return "o.weight";
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "attn_norm.weight";
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "gate.weight";
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "up.weight";
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "down.weight";
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "ffn_norm.weight";
        }
    }
}
