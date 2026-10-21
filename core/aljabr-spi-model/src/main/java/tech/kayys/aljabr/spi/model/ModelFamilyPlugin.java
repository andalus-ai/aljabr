package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Map;
/**
 * 
 * Core interface for kayys module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public interface ModelFamilyPlugin {

    default String id() {
        return "model-family/" + descriptor().id();
    }

    default int order() {
        return 0;
    }

    ModelFamilyDescriptor descriptor();

    default List<ModelArchitecture> architectureAdapters() {
        return List.of();
    }

    default List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of();
    }

    default List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
        Map<String, String> meta = descriptor().metadata();
        if (meta.containsKey("unified_model_type")) {
            String modalities = meta.getOrDefault("unified_runtime_required_modalities", "");
            List<String> mod = modalities.isEmpty() ? List.of() : List.of(modalities.split(","));
            return List.of(new ModelFamilyUnifiedRuntimeRequirement(
                    meta.get("unified_model_type"),
                    mod,
                    Boolean.parseBoolean(meta.getOrDefault("unified_runtime_production_ready", "true")),
                    meta.getOrDefault("unified_runtime_reason", ""),
                    Map.of()));
        }
        return List.of();
    }

    default ModelFamilySupportReport supportReport() {
        ModelFamilyDescriptor desc = descriptor();

        List<ModelArchitecture> adapters;
        try {
            adapters = architectureAdapters();
        } catch (Exception e) {
            adapters = List.of();
        }

        List<ModelTokenizerDescriptor> tokenizers;
        try {
            tokenizers = tokenizerDescriptors();
        } catch (Exception e) {
            tokenizers = List.of();
        }

        List<String> archClassNames = adapters.stream()
                .flatMap(a -> a.supportedArchClassNames().stream())
                .distinct()
                .toList();
        List<String> adapterIds = adapters.stream()
                .map(ModelArchitecture::id)
                .toList();

        List<String> tokenizerIds = tokenizers.stream()
                .map(ModelTokenizerDescriptor::id)
                .toList();
        List<ModelTokenizerKind> tokenizerKinds = tokenizers.stream()
                .map(ModelTokenizerDescriptor::kind)
                .distinct()
                .toList();

        String profileKey = desc.metadata().getOrDefault("bundle_profile", "");
        ModelFamilyBundleProfile bundleProfile = ModelFamilyBundleProfile.fromKey(profileKey);

        String directReason = desc.metadata().getOrDefault("direct_safetensor", "");
        ModelFamilyDirectSupport directSupport = ModelFamilyDirectSupport.PENDING;
        if (directReason.startsWith("experimental")) {
            directSupport = ModelFamilyDirectSupport.EXPERIMENTAL;
        } else if (directReason.startsWith("pending")) {
            directSupport = ModelFamilyDirectSupport.PENDING;
        } else if (desc.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)) {
            if (adapters.isEmpty()) {
                directSupport = ModelFamilyDirectSupport.DECLARED_NO_ADAPTER;
            } else {
                directSupport = ModelFamilyDirectSupport.READY;
                directReason = "";
            }
        }

        java.util.Map<String, String> caveats = new java.util.HashMap<>();
        for (java.util.Map.Entry<String, String> entry : desc.metadata().entrySet()) {
            if (entry.getKey().endsWith("_direct_safetensor")) {
                String key = entry.getKey().replace("_direct_safetensor", "");
                caveats.put(key, entry.getValue());
            }
        }

        return new ModelFamilySupportReport(
                descriptor().id(),
                desc.displayName(),
                desc.modelTypes(),
                archClassNames,
                adapterIds,
                tokenizerIds,
                tokenizerKinds,
                bundleProfile,
                desc.capabilities(),
                directSupport,
                directReason,
                caveats,
                java.util.Map.of());
    }

    default ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        String arch = config.primaryArchitecture();
        for (ModelArchitecture adapter : architectureAdapters()) {
            if (adapter.supportedArchClassNames().contains(arch)) {
                return adapter.runtimeTraits(config);
            }
        }
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }

    default ModelFamilyRuntimeManifest runtimeManifest() {
        String chatTemplates = descriptor().metadata().getOrDefault("chat_template_ids", "");
        List<String> chatTemplateIds = chatTemplates.isEmpty() ? List.of() : List.of(chatTemplates.split(","));

        return new ModelFamilyRuntimeManifest(
                descriptor().id(),
                descriptor().modelTypes(),
                architectureAdapters().stream().map(ModelArchitecture::id).toList(),
                tokenizerDescriptors().stream().map(ModelTokenizerDescriptor::id).toList(),
                chatTemplateIds,
                descriptor().metadata().getOrDefault("bundle_profile", "").equals("core")
                        ? ModelFamilyBundleProfile.CORE
                        : ModelFamilyBundleProfile.OPTIONAL,
                descriptor().capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                        ? ModelFamilyDirectSupport.READY
                        : ModelFamilyDirectSupport.PENDING,
                !tokenizerDescriptors().isEmpty(),
                descriptor().capabilities().contains(ModelFamilyCapability.CHAT_TEMPLATE),
                descriptor().capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                        && !architectureAdapters().isEmpty(),
                unifiedRuntimeRequirements());
    }
}
