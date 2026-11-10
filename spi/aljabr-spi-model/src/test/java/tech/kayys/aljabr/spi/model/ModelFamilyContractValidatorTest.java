package tech.kayys.aljabr.spi.model;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyContractValidatorTest {

    @Test
    void validModelFamilyPluginHasNoContractViolations() {
        ModelFamilyPlugin plugin = validPlugin("contract_valid", "contract_model");

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertTrue(violations.isEmpty(), () -> violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .collect(Collectors.joining("\n")));
    }

    @Test
    void invalidModelFamilyPluginReportsMachineReadableCodes() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "Bad Family",
                        "Bad Family",
                        List.of("bad type"),
                        List.of(),
                        List.of(ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.MULTIMODAL,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "research",
                                "moe_direct_safetensor", "ready_now"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(new ModelTokenizerDescriptor(
                        "bad tokenizer",
                        ModelTokenizerKind.HUGGING_FACE_BPE,
                        List.of(List.of("../tokenizer.json")),
                        Map.of()));
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("invalid_family_id"));
        assertTrue(codes.contains("invalid_model_type"));
        assertTrue(codes.contains("unknown_bundle_profile"));
        assertTrue(codes.contains("multimodal_without_modality"));
        assertTrue(codes.contains("invalid_tokenizer_id"));
        assertTrue(codes.contains("tokenizer_unsafe_file"));
        assertTrue(codes.contains("direct_safetensor_without_adapter"));
        assertTrue(codes.contains("invalid_scoped_direct_safetensor_reason"));
    }

    @Test
    void directSafetensorAdapterMustMatchDescriptorClaims() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "unclaimed_adapter",
                        "Unclaimed Adapter",
                        List.of("claimed_type"),
                        List.of("ClaimedForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/claimed"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("architecture_adapter_unclaimed"));
    }

    @Test
    void pendingTokenizerMetadataAllowsDocumentedDescriptorlessTokenizer() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "pending_tokenizer",
                        "Pending Tokenizer",
                        List.of("pending_tokenizer"),
                        List.of(),
                        List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                        Map.of(
                                "bundle_profile", "experimental",
                                "origin", "3rdparty/transformers/src/transformers/models/pending_tokenizer",
                                "tokenizer_metadata_status", "pending",
                                "tokenizer_metadata_pending_reason",
                                "tokenizer descriptor pending upstream tokenizer fixture"));
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertFalse(codes.contains("tokenizer_capability_without_descriptor"));
        assertFalse(codes.contains("tokenizer_metadata_pending_reason_missing"));
    }

    @Test
    void tokenizerMetadataChecklistReportsDrift() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "ready_tokenizer_without_descriptor",
                        "Ready Tokenizer Without Descriptor",
                        List.of("ready_tokenizer_without_descriptor"),
                        List.of(),
                        List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/ready_tokenizer",
                                "tokenizer_metadata_status", "ready",
                                "tokenizer_metadata_pending_reason", "stale pending reason"));
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("tokenizer_metadata_ready_without_descriptor"));
        assertTrue(codes.contains("tokenizer_metadata_pending_reason_without_pending_status"));
    }

    @Test
    void unifiedRuntimeRequirementsMustBeDiscoverableAndClaimed() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "runtime_requirement_drift",
                        "Runtime Requirement Drift",
                        List.of("claimed_runtime_type"),
                        List.of(),
                        List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/runtime_requirement"));
            }

            @Override
            public List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
                return List.of(new ModelFamilyUnifiedRuntimeRequirement(
                        "unclaimed_runtime_type",
                        List.of(),
                        true,
                        "requires a detached runtime",
                        Map.of()));
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("unified_runtime_requirement_missing_metadata"));
        assertTrue(codes.contains("unified_runtime_requirement_unclaimed_model_type"));
        assertTrue(codes.contains("unified_runtime_requirement_missing_modalities"));
    }

    @Test
    void originMetadataReportsUnsafePathShapes() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "bad_origin_shape",
                        "Bad Origin Shape",
                        List.of("bad_origin_shape"),
                        List.of(),
                        List.of(ModelFamilyCapability.CAUSAL_LM),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "tmp/models/bad origin"));
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("origin_contains_whitespace"));
        assertTrue(codes.contains("unexpected_origin_path"));
    }

    @Test
    void supportReportFailureIsContractViolation() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "broken_report_contract",
                        "Broken Report Contract",
                        List.of("broken_report_contract"),
                        List.of("ContractForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/contract"));
            }

            @Override
            public ModelFamilySupportReport supportReport() {
                throw new IllegalStateException("support report failed");
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("support_report_unavailable"));
    }

    @Test
    void supportReportMustStayConsistentWithPluginSurface() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "support_report_drift",
                        "Support Report Drift",
                        List.of("support_report_drift"),
                        List.of("ContractForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/contract"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("support-report-drift-bpe"));
            }

            @Override
            public ModelFamilySupportReport supportReport() {
                return new ModelFamilySupportReport(
                        "drifted_report",
                        "Drifted Report",
                        List.of("other_model_type"),
                        List.of("OtherForCausalLM"),
                        List.of("other-adapter"),
                        List.of("other-tokenizer"),
                        List.of(ModelTokenizerKind.WORD_PIECE),
                        ModelFamilyBundleProfile.CORE,
                        List.of(ModelFamilyCapability.EMBEDDING),
                        ModelFamilyDirectSupport.PENDING,
                        "pending_custom_path",
                        Map.of("moe", "pending_expert_runtime"),
                        Map.of());
            }
        };

        Set<String> codes = codes(ModelFamilyContractValidator.validate(plugin));

        assertTrue(codes.contains("support_report_id_mismatch"));
        assertTrue(codes.contains("support_report_display_name_mismatch"));
        assertTrue(codes.contains("support_report_model_types_mismatch"));
        assertTrue(codes.contains("support_report_architectures_mismatch"));
        assertTrue(codes.contains("support_report_architecture_adapters_mismatch"));
        assertTrue(codes.contains("support_report_tokenizers_mismatch"));
        assertTrue(codes.contains("support_report_tokenizer_kinds_mismatch"));
        assertTrue(codes.contains("support_report_capabilities_mismatch"));
        assertTrue(codes.contains("support_report_bundle_profile_mismatch"));
        assertTrue(codes.contains("support_report_direct_safetensor_mismatch"));
    }

    @Test
    void validateAllReportsDuplicateModelTypeClaims() {
        ModelFamilyPlugin first = validPlugin("contract_one", "shared_contract_type");
        ModelFamilyPlugin second = validPlugin("contract_two", "SHARED_CONTRACT_TYPE");

        Set<String> codes = codes(ModelFamilyContractValidator.validateAll(List.of(first, second)));

        assertTrue(codes.contains("duplicate_model_type_claim"));
    }

    @Test
    void capabilityMatrixEntryFlattensSupportReportForAutomation() {
        ModelFamilyCapabilityMatrixEntry entry = ModelFamilyCapabilityMatrixEntry.from(validPlugin(
                "matrix_family", "matrix_model").supportReport());

        assertTrue(entry.causalLm());
        assertTrue(entry.tokenizer());
        assertTrue(!entry.moe());
        assertTrue(entry.architectureAdapterPresent());
        assertTrue(entry.directSafetensorReady());
        assertTrue(entry.compactSummary().contains("matrix_family[optional]"));
        assertTrue(entry.compactSummary().contains("safetensor=ready"));
        assertTrue(entry.compactSummary().contains("adapters=1"));
    }

    private static ModelFamilyPlugin validPlugin(String id, String modelType) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        id,
                        id,
                        List.of(modelType),
                        List.of("ContractForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/contract"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe(id + "-bpe"));
            }
        };
    }

    private static Set<String> codes(List<ModelFamilyContractViolation> violations) {
        return violations.stream()
                .map(ModelFamilyContractViolation::code)
                .collect(Collectors.toSet());
    }

    private static final class StubArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "contract-stub";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("ContractForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("contract_model");
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
