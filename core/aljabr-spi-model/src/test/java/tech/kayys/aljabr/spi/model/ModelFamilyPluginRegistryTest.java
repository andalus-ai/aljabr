package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelFamilyPluginRegistryTest {

    @TempDir
    Path tempDir;

    @Test
    void selectsTokenizerProfilesByModelType() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "unit-test-family",
                        "Unit Test Family",
                        List.of("unit_model"),
                        List.of("UnitForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of());
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("unit-hf-bpe"));
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            List<ModelTokenizerDescriptor> descriptors = registry.tokenizerDescriptorsFor("unit_model", "Other");

            assertEquals(1, descriptors.size());
            assertEquals("unit-hf-bpe", descriptors.get(0).id());
            assertEquals(ModelTokenizerKind.HUGGING_FACE_BPE, descriptors.get(0).kind());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void runtimeManifestCarriesUnifiedRuntimeRequirementsFromMetadata() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "unified-runtime-family",
                        "Unified Runtime Family",
                        List.of("unified_runtime_model"),
                        List.of("UnifiedRuntimeForConditionalGeneration"),
                        List.of(ModelFamilyCapability.MULTIMODAL),
                        Map.of(
                                "unified_model_type", "unified_runtime_model",
                                "unified_runtime_required_modalities", "text,image",
                                "unified_runtime_reason", "needs external unified runtime"));
            }
        };

        ModelFamilyRuntimeManifest manifest = plugin.runtimeManifest();
        ModelFamilyUnifiedRuntimeRequirement requirement = manifest.unifiedRuntimeRequirements().getFirst();

        assertTrue(manifest.requiresUnifiedRuntime());
        assertEquals("unified_runtime_model", requirement.modelType());
        assertEquals(List.of("text", "image"), requirement.requiredInputModalities());
        assertTrue(requirement.productionReadyRequired());
        assertEquals("needs external unified runtime", requirement.reason());
    }

    @Test
    void resolvesModelFamilyByModelTypeWithTokenizerDescriptors() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "resolution-family",
                        "Resolution Family",
                        List.of("resolution_model"),
                        List.of("ResolutionForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.sentencePieceBpe("resolution-spm"));
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyResolution resolution = registry.resolve("resolution_model", null);

            assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
            assertTrue(resolution.resolved());
            assertEquals(List.of("resolution-family"), resolution.familyIds());
            assertEquals("resolution-family", resolution.primaryFamilyId().orElseThrow());
            assertEquals(List.of("resolution-family"), resolution.supportReports().stream()
                    .map(ModelFamilySupportReport::id)
                    .toList());
            assertEquals("resolution-family", resolution.primarySupportReport().orElseThrow().id());
            assertEquals("resolution-family", resolution.primaryRuntimeManifest().orElseThrow().familyId());
            assertEquals(List.of("resolution-spm"), resolution.runtimeManifests().getFirst()
                    .tokenizerProfileIds());
            assertEquals(List.of("resolution-spm"), resolution.tokenizerDescriptors().stream()
                    .map(ModelTokenizerDescriptor::id)
                    .toList());
            assertTrue(!resolution.requiresAttention());
            assertTrue(resolution.problemCodes().isEmpty());
            assertTrue(resolution.remediationHints().isEmpty());
            assertTrue(resolution.summary().contains("resolved model_type=resolution_model"));
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void resolvesModelFamilyByArchitectureClassName() {
        ModelFamilyPlugin plugin = metadataOnlyPlugin("arch-resolution-family", "arch_resolution_model",
                "ArchResolutionForCausalLM");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyResolution resolution = registry.resolve(null, "ArchResolutionForCausalLM");

            assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
            assertEquals(List.of("arch-resolution-family"), resolution.familyIds());
            assertTrue(resolution.summary().contains("architecture=ArchResolutionForCausalLM"));
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void exposesRuntimeManifestForResolvedFamilies() throws Exception {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "runtime-manifest-family",
                        "Runtime Manifest Family",
                        List.of("runtime_manifest"),
                        List.of("RuntimeManifestForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.CHAT_TEMPLATE,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "core",
                                "origin", "3rdparty/transformers/src/transformers/models/runtime_manifest",
                                "chat_template_ids", "runtime-chat,runtime-json"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        "runtime-manifest-adapter",
                        "runtime_manifest",
                        "RuntimeManifestForCausalLM"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("runtime-hf-bpe"));
            }
        };
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "architectures": ["RuntimeManifestForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            List<ModelFamilyRuntimeManifest> manifests = registry.runtimeManifestsFor("runtime_manifest", null);
            ModelFamilyRuntimeManifest manifest = manifests.getFirst();

            assertEquals(1, manifests.size());
            assertEquals("runtime-manifest-family", manifest.familyId());
            assertEquals(List.of("runtime_manifest"), manifest.modelTypes());
            assertEquals(List.of("runtime-manifest-adapter"), manifest.architectureAdapterIds());
            assertEquals(List.of("runtime-hf-bpe"), manifest.tokenizerProfileIds());
            assertEquals(List.of("runtime-chat", "runtime-json"), manifest.chatTemplateIds());
            assertEquals(ModelFamilyBundleProfile.CORE, manifest.bundleProfile());
            assertEquals(ModelFamilyDirectSupport.READY, manifest.directSafetensorStatus());
            assertTrue(manifest.tokenizerReady());
            assertTrue(manifest.chatTemplateReady());
            assertTrue(manifest.directSafetensorReady());
            assertEquals("runtime-manifest-family",
                    registry.runtimeManifest("runtime-manifest-family").orElseThrow().familyId());
            assertEquals("runtime-manifest-family",
                    registry.resolve("runtime_manifest", null).primaryRuntimeManifest().orElseThrow().familyId());
            ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(
                    registry.resolve("runtime_manifest", null));
            assertTrue(compatibility.compatible());
            assertTrue(compatibility.problemCodes().isEmpty());
            assertEquals("runtime-manifest-adapter", compatibility.selectedArchitectureAdapterId());
            assertEquals("model_type", compatibility.selectedArchitectureAdapterBy());
            assertEquals("runtime-manifest-family", registry
                    .directSafetensorCompatibilityForPlugin("runtime-manifest-family")
                    .orElseThrow()
                    .modelFamily()
                    .primaryFamilyId()
                    .orElseThrow());
            assertTrue(registry.directSafetensorCompatibilities().stream()
                    .anyMatch(candidate -> candidate.modelFamily().familyIds()
                            .contains("runtime-manifest-family")
                            && candidate.compatible()));
            ModelFamilyRuntimeCompatibilitySummary summary = registry.directSafetensorCompatibilitySummary();
            assertTrue(summary.familyCount() >= 1);
            assertTrue(summary.compatibleFamilyIds().contains("runtime-manifest-family"));
            assertEquals(0, summary.problemCounts().getOrDefault(
                    "model_family_architecture_adapters_missing", 0));
            assertEquals(List.of("runtime-manifest-family"), registry
                    .directSafetensorCompatibilitySummaryForFamilies(List.of("runtime-manifest-family"))
                    .compatibleFamilyIds());
            assertTrue(registry
                    .directSafetensorCompatibilitySummaryForFamilies(List.of("missing-runtime-family"))
                    .empty());
            assertTrue(plugin.runtimeTraits(config).nativeBf16Matvec());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void directSafetensorCompatibilityReportsPendingQuantizedLoaderFromModelConfig() throws Exception {
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "quantized_runtime",
                  "architectures": ["QuantizedRuntimeForCausalLM"],
                  "quantization_config": {
                    "format": "mobile",
                    "container": "compressed_tensors",
                    "loader_scope": "metadata_only_pending_mobile_quant_loader"
                  }
                }
                """);
        ModelFamilyPlugin plugin = directReadyPlugin(
                "quantized-runtime-family",
                "quantized_runtime",
                "QuantizedRuntimeForCausalLM");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(
                    registry.resolve("quantized_runtime", "QuantizedRuntimeForCausalLM"),
                    tempDir);

            assertTrue(!compatibility.compatible());
            assertEquals("quantized-runtime-family-adapter", compatibility.selectedArchitectureAdapterId());
            assertTrue(compatibility.problemCodes().contains(
                    ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING));
            assertTrue(compatibility.problemCodes().contains(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING));
            assertTrue(compatibility.problemCodes().stream()
                    .noneMatch("model_family_architecture_adapter_unmatched"::equals));
            assertTrue(compatibility.remediationHints().stream()
                    .anyMatch(hint -> hint.contains("mobile quantized weights in compressed_tensors")));
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void quantizedLoaderProfileInfersMobileQatSupportedFromUpstreamConfigShape() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4",
                  "quantization_config": {
                    "quant_method": "gemma",
                    "num_bits": 4
                  },
                  "text_config": {
                    "model_type": "gemma4_text"
                  },
                  "vision_config": {
                    "model_type": "gemma4_vision"
                  },
                  "audio_config": {
                    "model_type": "gemma4_audio"
                  }
                }
                """);

        ModelFamilyQuantizedLoaderProfile profile = ModelFamilyQuantizedLoaderProfile.fromModelDir(tempDir);

        assertTrue(profile.mobileQatSupported());
        assertTrue(profile.inferredFromConfig());
        assertEquals("mobile", profile.format());
        assertEquals("transformers", profile.container());
        assertEquals("metadata_only_pending_mobile_quant_loader", profile.loaderScope());
        assertTrue(profile.problemCodes().contains(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING));
    }

    @Test
    void quantizedLoaderProfileDoesNotInferMobileQatSupportedWithoutMultimodalTowers() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4",
                  "quantization_config": {
                    "quant_method": "gemma",
                    "num_bits": 4
                  },
                  "text_config": {
                    "model_type": "gemma4_text"
                  }
                }
                """);

        ModelFamilyQuantizedLoaderProfile profile = ModelFamilyQuantizedLoaderProfile.fromModelDir(tempDir);

        assertEquals(null, profile);
    }

    @Test
    void directSafetensorCompatibilityReportsGenericPendingQuantizedLoaderForUnknownFormat() throws Exception {
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "quantized_runtime",
                  "architectures": ["QuantizedRuntimeForCausalLM"],
                  "quantization_config": {
                    "format": "future_format",
                    "container": "future_container",
                    "loader_scope": "metadata_only_pending_future_quant_loader"
                  }
                }
                """);
        ModelFamilyPlugin plugin = directReadyPlugin(
                "quantized-runtime-family",
                "quantized_runtime",
                "QuantizedRuntimeForCausalLM");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(
                    registry.resolve("quantized_runtime", "QuantizedRuntimeForCausalLM"),
                    tempDir);

            assertTrue(!compatibility.compatible());
            assertEquals(List.of(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING),
                    compatibility.problemCodes());
            assertTrue(compatibility.remediationHints().stream()
                    .anyMatch(hint -> hint.contains("future_format quantized weights in future_container")));
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void directSafetensorCompatibilityDoesNotBlockSupportedQuantizedLoaderScope() throws Exception {
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "quantized_runtime",
                  "architectures": ["QuantizedRuntimeForCausalLM"],
                  "quantization_config": {
                    "format": "q4_0",
                    "container": "gguf",
                    "loader_scope": "ready_q4_0_weight_loader"
                  }
                }
                """);
        ModelFamilyPlugin plugin = directReadyPlugin(
                "quantized-runtime-family",
                "quantized_runtime",
                "QuantizedRuntimeForCausalLM");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(
                    registry.resolve("quantized_runtime", "QuantizedRuntimeForCausalLM"),
                    tempDir);

            assertTrue(compatibility.compatible());
            assertEquals(List.of(), compatibility.problemCodes());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void pluginRuntimeTraitsDelegateToMatchingArchitectureAdapter() throws Exception {
        ModelRuntimeTraits adapterTraits = ModelRuntimeTraits.builder()
                .qwenText()
                .perLayerInputEmbedding()
                .build();
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "runtime-traits-family",
                        "Runtime Traits Family",
                        List.of("runtime_traits"),
                        List.of("RuntimeTraitsForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        "runtime-traits-adapter",
                        "runtime_traits",
                        "RuntimeTraitsForCausalLM",
                        adapterTraits));
            }
        };
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "runtime_traits",
                  "architectures": ["RuntimeTraitsForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelRuntimeTraits traits = plugin.runtimeTraits(config);

            assertTrue(traits.qwenText());
            assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void resolvesModelFamilyFromParsedConfig() throws Exception {
        ModelFamilyPlugin plugin = metadataOnlyPlugin("config-resolution-family", "config_resolution",
                "ConfigResolutionForCausalLM");
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "config_resolution",
                  "architectures": ["ConfigResolutionForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            ModelFamilyResolution resolution = registry.resolve(config);

            assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
            assertEquals("config_resolution", resolution.modelType());
            assertEquals("ConfigResolutionForCausalLM", resolution.architectureClassName());
            assertEquals(List.of("config-resolution-family"), resolution.familyIds());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void isolatedRegistryDoesNotMutateGlobalRegistry() {
        ModelFamilyPlugin plugin = metadataOnlyPlugin("isolated-family", "isolated_model",
                "IsolatedForCausalLM");
        ModelFamilyPluginRegistry isolated = ModelFamilyPluginRegistry.create();

        isolated.register(plugin);

        assertEquals(ModelFamilyResolution.Status.RESOLVED,
                isolated.resolve("isolated_model", null).status());
        assertEquals(ModelFamilyResolution.Status.NOT_FOUND,
                ModelFamilyPluginRegistry.global().resolve("isolated_model", null).status());
        assertTrue(ModelFamilyPluginRegistry.global().plugin("isolated-family").isEmpty());
    }

    @Test
    void selectsArchitectureAdaptersByResolvedFamily() throws Exception {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "adapter-resolution-family",
                        "Adapter Resolution Family",
                        List.of("adapter_resolution"),
                        List.of("AdapterResolutionForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        "adapter-resolution",
                        "adapter_resolution",
                        "AdapterResolutionForCausalLM"));
            }
        };
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "adapter_resolution",
                  "architectures": ["AdapterResolutionForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            assertEquals(List.of("adapter-resolution"), registry.architectureAdaptersFor(config).stream()
                    .map(ModelArchitecture::id)
                    .toList());
            assertEquals(List.of("adapter-resolution"), registry.architectureAdaptersFor(
                    "adapter_resolution",
                    "OtherForCausalLM")
                    .stream()
                    .map(ModelArchitecture::id)
                    .toList());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void reportsAmbiguousModelFamilyResolution() {
        ModelFamilyPlugin first = metadataOnlyPlugin("resolution-claim-one", "shared_resolution");
        ModelFamilyPlugin second = metadataOnlyPlugin("resolution-claim-two", "shared_resolution");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(first);
        registry.register(second);
        try {
            ModelFamilyResolution resolution = registry.resolveModelType("shared_resolution");

            assertEquals(ModelFamilyResolution.Status.AMBIGUOUS, resolution.status());
            assertTrue(resolution.ambiguous());
            assertEquals(List.of("resolution-claim-one", "resolution-claim-two"), resolution.familyIds());
            assertTrue(resolution.requiresAttention());
            assertEquals(List.of("model_family_ambiguous"), resolution.problemCodes());
            assertTrue(resolution.remediationHints().stream()
                    .anyMatch(hint -> hint.contains("overlapping model-family plugins")));
            assertTrue(resolution.summary().contains("ambiguous model_type=shared_resolution"));
        } finally {
            registry.unregister(first.id());
            registry.unregister(second.id());
        }
    }

    @Test
    void reportsMissingModelFamilyResolution() {
        ModelFamilyResolution resolution = ModelFamilyPluginRegistry.global()
                .resolve("missing_resolution_model", "MissingResolutionForCausalLM");

        assertEquals(ModelFamilyResolution.Status.NOT_FOUND, resolution.status());
        assertTrue(resolution.notFound());
        assertEquals(List.of(), resolution.familyIds());
        assertEquals(List.of(), resolution.tokenizerDescriptors());
        assertTrue(resolution.requiresAttention());
        assertEquals(List.of("model_family_not_found"), resolution.problemCodes());
        assertTrue(resolution.remediationHints().stream()
                .anyMatch(hint -> hint.contains("missing_resolution_model")));
        assertTrue(resolution.summary().contains("no model family matched"));
    }

    @Test
    void skipsBrokenAdapterAndTokenizerDiscovery() {
        ModelFamilyPlugin healthy = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "healthy-discovery",
                        "Healthy Discovery",
                        List.of("healthy_discovery"),
                        List.of("HealthyDiscoveryForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of());
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("healthy-hf-bpe"));
            }
        };
        ModelFamilyPlugin broken = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "broken-discovery",
                        "Broken Discovery",
                        List.of("broken_discovery"),
                        List.of("BrokenDiscoveryForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER,
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

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(healthy);
        registry.register(broken);
        try {
            List<ModelArchitecture> adapters = registry.architectureAdapters();
            List<ModelTokenizerDescriptor> healthyTokenizers = registry.tokenizerDescriptorsFor("healthy_discovery",
                    null);
            List<ModelTokenizerDescriptor> brokenTokenizers = registry.tokenizerDescriptorsFor("broken_discovery",
                    null);

            assertTrue(adapters.stream().anyMatch(adapter -> adapter.id().equals("stub")));
            assertEquals(List.of("healthy-hf-bpe"), healthyTokenizers.stream()
                    .map(ModelTokenizerDescriptor::id)
                    .toList());
            assertEquals(List.of(), brokenTokenizers);
        } finally {
            registry.unregister(healthy.id());
            registry.unregister(broken.id());
        }
    }

    @Test
    void skipsBrokenDescriptorDiscovery() {
        ModelFamilyPlugin healthy = metadataOnlyPlugin("healthy-descriptor", "healthy_descriptor");
        ModelFamilyPlugin broken = new ModelFamilyPlugin() {
            @Override
            public String id() {
                return "model-family/broken-descriptor";
            }

            @Override
            public ModelFamilyDescriptor descriptor() {
                throw new IllegalStateException("descriptor discovery failed");
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(healthy);
        registry.register(broken);
        try {
            assertTrue(registry.plugin("healthy-descriptor").isPresent());
            assertTrue(registry.plugin("broken-descriptor").isPresent());
            assertTrue(registry.pluginsForModelType("healthy_descriptor").stream()
                    .anyMatch(plugin -> plugin.id().equals("model-family/healthy-descriptor")));
            assertEquals(List.of(), registry.pluginsForModelType("broken_descriptor"));
            assertTrue(registry.descriptors().stream()
                    .anyMatch(descriptor -> descriptor.id().equals("healthy-descriptor")));
            assertTrue(registry.descriptors().stream()
                    .noneMatch(descriptor -> descriptor.id().equals("broken-descriptor")));
            assertTrue(registry.modelTypeClaims().containsKey("healthy_descriptor"));
            assertTrue(!registry.modelTypeClaims().containsKey("broken_descriptor"));
            assertTrue(registry.modelTypeConflicts().isEmpty());
        } finally {
            registry.unregister(healthy.id());
            registry.unregister(broken.id());
        }
    }

    @Test
    void toleratesBrokenPluginIdAndOrder() {
        ModelFamilyPlugin broken = new ModelFamilyPlugin() {
            @Override
            public String id() {
                throw new IllegalStateException("id discovery failed");
            }

            @Override
            public int order() {
                throw new IllegalStateException("order discovery failed");
            }

            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "broken_identity",
                        "Broken Identity",
                        List.of("broken_identity"),
                        List.of(),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "optional"));
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(broken);
        try {
            assertTrue(registry.all().contains(broken));
            assertTrue(registry.plugin("broken_identity").isPresent());
            assertTrue(registry.contractViolations().stream()
                    .anyMatch(violation -> violation.familyId().equals("broken_identity")
                            && violation.code().equals("plugin_id_mismatch")));

            registry.unregister("broken_identity");

            assertTrue(!registry.all().contains(broken));
        } finally {
            registry.unregister("broken_identity");
        }
    }

    @Test
    void reportsDuplicateModelTypeClaims() {
        ModelFamilyPlugin first = metadataOnlyPlugin("claim-one", "shared_type");
        ModelFamilyPlugin second = metadataOnlyPlugin("claim-two", "SHARED_TYPE");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(first);
        registry.register(second);
        try {
            List<ModelFamilyClaimConflict> conflicts = registry.modelTypeConflicts();

            assertEquals(1, conflicts.size());
            assertEquals("model_type", conflicts.get(0).claimType());
            assertEquals("shared_type", conflicts.get(0).claim());
            assertEquals(List.of("claim-one", "claim-two"), conflicts.get(0).familyIds());
        } finally {
            registry.unregister(first.id());
            registry.unregister(second.id());
        }
    }

    @Test
    void looksUpAndUnregistersByShortOrFullFamilyId() {
        ModelFamilyPlugin plugin = metadataOnlyPlugin("lookup-family", "lookup_type");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(plugin);
        try {
            assertTrue(registry.plugin("lookup-family").isPresent());
            assertTrue(registry.plugin("model-family/lookup-family").isPresent());

            registry.unregister("lookup-family");

            assertTrue(registry.plugin("lookup-family").isEmpty());
        } finally {
            registry.unregister(plugin.id());
        }
    }

    @Test
    void filtersSupportReportsByBundleProfile() {
        ModelFamilyPlugin core = profiledPlugin("profile-core", "core_type", "core");
        ModelFamilyPlugin metadata = profiledPlugin("profile-metadata", "metadata_type", "metadata_only");

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(core);
        registry.register(metadata);
        try {
            List<ModelFamilySupportReport> coreReports = registry
                    .supportReportsForProfile(ModelFamilyBundleProfile.CORE);
            List<ModelFamilySupportReport> metadataReports = registry
                    .supportReportsForProfile(ModelFamilyBundleProfile.METADATA_ONLY);

            assertTrue(coreReports.stream().anyMatch(report -> report.id().equals("profile-core")));
            assertTrue(metadataReports.stream().anyMatch(report -> report.id().equals("profile-metadata")));
        } finally {
            registry.unregister(core.id());
            registry.unregister(metadata.id());
        }
    }

    @Test
    void skipsBrokenSupportReports() {
        ModelFamilyPlugin healthy = profiledPlugin("healthy-report", "healthy_type", "core");
        ModelFamilyPlugin broken = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "broken-report",
                        "Broken Report",
                        List.of("broken_report"),
                        List.of("BrokenReportForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public ModelFamilySupportReport supportReport() {
                throw new IllegalStateException("support report failed");
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(healthy);
        registry.register(broken);
        try {
            List<ModelFamilySupportReport> reports = registry.supportReportsForProfile(ModelFamilyBundleProfile.CORE);
            List<ModelFamilyCapabilityMatrixEntry> matrix = registry.capabilityMatrix();
            List<ModelFamilyContractViolation> violations = registry.contractViolations();

            assertTrue(reports.stream().anyMatch(report -> report.id().equals("healthy-report")));
            assertTrue(reports.stream().noneMatch(report -> report.id().equals("broken-report")));
            assertTrue(matrix.stream().anyMatch(entry -> entry.id().equals("healthy-report")));
            assertTrue(matrix.stream().noneMatch(entry -> entry.id().equals("broken-report")));
            assertTrue(violations.stream().anyMatch(violation -> violation.familyId().equals("broken-report")
                    && violation.code().equals("support_report_unavailable")));
        } finally {
            registry.unregister(healthy.id());
            registry.unregister(broken.id());
        }
    }

    private static ModelFamilyPlugin metadataOnlyPlugin(String id, String modelType) {
        return profiledPlugin(id, modelType, List.of(), "optional");
    }

    private static ModelFamilyPlugin metadataOnlyPlugin(String id, String modelType, String architectureClassName) {
        return profiledPlugin(id, modelType, List.of(architectureClassName), "optional");
    }

    private static ModelFamilyPlugin profiledPlugin(String id, String modelType, String profile) {
        return profiledPlugin(id, modelType, List.of(), profile);
    }

    private static ModelFamilyPlugin profiledPlugin(
            String id,
            String modelType,
            List<String> architectureClassNames,
            String profile) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        id,
                        id,
                        List.of(modelType),
                        architectureClassNames,
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", profile));
            }
        };
    }

    private static ModelFamilyPlugin directReadyPlugin(String id, String modelType, String architectureClassName) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        id,
                        id,
                        List.of(modelType),
                        List.of(architectureClassName),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.TOKENIZER,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        id + "-adapter",
                        modelType,
                        architectureClassName));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe(id + "-tokenizer"));
            }
        };
    }

    private static final class StubArchitecture implements ModelArchitecture {
        private final String id;
        private final String modelType;
        private final String architectureClassName;
        private final ModelRuntimeTraits runtimeTraits;

        private StubArchitecture() {
            this("stub", "ready", "ReadyForCausalLM");
        }

        private StubArchitecture(String id, String modelType, String architectureClassName) {
            this(id, modelType, architectureClassName, null);
        }

        private StubArchitecture(String id, String modelType, String architectureClassName,
                ModelRuntimeTraits runtimeTraits) {
            this.id = id;
            this.modelType = modelType;
            this.architectureClassName = architectureClassName;
            this.runtimeTraits = runtimeTraits;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of(architectureClassName);
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of(modelType);
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

        @Override
        public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
            return runtimeTraits == null ? ModelArchitecture.super.runtimeTraits(config) : runtimeTraits;
        }
    }
}
