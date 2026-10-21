package tech.kayys.aljabr.spi.model;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelTokenizerDescriptorTest {

    @Test
    void wordPieceDescriptorFindsNestedVocabFile() throws IOException {
        Path dir = Files.createTempDirectory("aljabr-tokenizer-descriptor");
        Path tokenizerDir = dir.resolve("tokenizer");
        Path vocab = tokenizerDir.resolve("vocab.txt");
        try {
            Files.createDirectories(tokenizerDir);
            Files.writeString(vocab, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n", StandardCharsets.UTF_8);

            ModelTokenizerDescriptor descriptor = ModelTokenizerDescriptor.wordPiece("bert-wordpiece");

            assertTrue(descriptor.firstExistingFileGroup(dir).isPresent());
            assertEquals(List.of(vocab), descriptor.firstExistingFileGroup(dir).orElseThrow());
        } finally {
            Files.deleteIfExists(vocab);
            Files.deleteIfExists(tokenizerDir);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void descriptorNormalizesFamilyIdsAndModelTypeClaims() {
        ModelFamilyDescriptor descriptor = new ModelFamilyDescriptor(
                " QWEN ",
                "  Qwen Family  ",
                List.of(" Qwen2 ", "qwen2", "QWEN3"),
                List.of(" Qwen2ForCausalLM ", "Qwen2ForCausalLM"),
                List.of(ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TOKENIZER),
                Map.of());

        assertEquals("qwen", descriptor.id());
        assertEquals("Qwen Family", descriptor.displayName());
        assertEquals(List.of("qwen2", "qwen3"), descriptor.modelTypes());
        assertEquals(List.of("Qwen2ForCausalLM"), descriptor.architectureClassNames());
        assertEquals(List.of(ModelFamilyCapability.TOKENIZER), descriptor.capabilities());
    }

    @Test
    void supportReportRequiresDirectCapabilityAndArchitectureAdapter() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "ready-family",
                        "Ready Family",
                        List.of("ready"),
                        List.of("ReadyForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of());
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(ModelFamilyDirectSupport.READY, report.directSafetensorStatus());
        assertTrue(report.directSafetensorReady());
        assertEquals(List.of("stub"), report.architectureAdapterIds());
    }

    @Test
    void supportReportTreatsExperimentalDirectPathAsNotProductionReady() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "experimental-direct-family",
                        "Experimental Direct Family",
                        List.of("experimental-direct"),
                        List.of("ExperimentalDirectForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("direct_safetensor", "experimental_guarded_path"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(ModelFamilyDirectSupport.EXPERIMENTAL, report.directSafetensorStatus());
        assertTrue(!report.directSafetensorReady());
        assertEquals("experimental_guarded_path", report.directSafetensorReason());
    }

    @Test
    void supportReportHonorsPendingMetadataEvenWhenAdapterExists() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "pending-direct-family",
                        "Pending Direct Family",
                        List.of("pending-direct"),
                        List.of("PendingDirectForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("direct_safetensor", "pending_custom_kernel"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(ModelFamilyDirectSupport.PENDING, report.directSafetensorStatus());
        assertTrue(!report.directSafetensorReady());
        assertEquals("pending_custom_kernel", report.directSafetensorReason());
    }

    @Test
    void supportReportCollectsPartialDirectSafetensorCaveats() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "partial-direct-family",
                        "Partial Direct Family",
                        List.of("partial-direct"),
                        List.of("PartialDirectForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "direct_safetensor", "experimental_text_path",
                                "direct_safetensor_scope", "text_only",
                                "moe_direct_safetensor", "pending_packed_expert_runtime",
                                "multimodal_direct_safetensor", "pending_projector_runtime"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(ModelFamilyDirectSupport.EXPERIMENTAL, report.directSafetensorStatus());
        assertEquals(Map.of(
                "moe", "pending_packed_expert_runtime",
                "multimodal", "pending_projector_runtime"), report.directSafetensorCaveats());
        assertEquals(
                "experimental:experimental_text_path;caveats=moe:pending_packed_expert_runtime,multimodal:pending_projector_runtime",
                report.shortDirectSafetensorSummary());
    }

    @Test
    void supportReportKeepsPendingReasonWhenFamilyIsMetadataOnly() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "pending-family",
                        "Pending Family",
                        List.of("pending"),
                        List.of("PendingForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                        Map.of(
                                "bundle_profile", "experimental",
                                "direct_safetensor", "pending_custom_kernel"));
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(ModelFamilyBundleProfile.EXPERIMENTAL, report.bundleProfile());
        assertTrue(!report.defaultBundle());
        assertEquals(ModelFamilyDirectSupport.PENDING, report.directSafetensorStatus());
        assertEquals("pending_custom_kernel", report.directSafetensorReason());
    }

    @Test
    void supportReportToleratesBrokenAdapterAndTokenizerLists() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "broken-extension-family",
                        "Broken Extension Family",
                        List.of("broken"),
                        List.of("BrokenForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of());
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                throw new IllegalStateException("adapter discovery failed");
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                throw new IllegalStateException("tokenizer discovery failed");
            }
        };

        ModelFamilySupportReport report = plugin.supportReport();

        assertEquals(List.of(), report.architectureAdapterIds());
        assertEquals(List.of(), report.tokenizerProfileIds());
        assertEquals(List.of(), report.tokenizerKinds());
        assertEquals(ModelFamilyDirectSupport.DECLARED_NO_ADAPTER, report.directSafetensorStatus());
    }

    private static final class StubArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "stub";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("ReadyForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("ready");
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
