package tech.kayys.aljabr.spi.model;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class ModelFamilyPluginRegistry {
    private static final ModelFamilyPluginRegistry INSTANCE = new ModelFamilyPluginRegistry();
    private final Map<String, ModelFamilyPlugin> plugins = new LinkedHashMap<>();

    public static ModelFamilyPluginRegistry global() {
        return INSTANCE;
    }

    public static ModelFamilyPluginRegistry create() {
        return new ModelFamilyPluginRegistry();
    }

    public synchronized List<ModelFamilyPlugin> all() {
        return new ArrayList<>(plugins.values());
    }

    public synchronized Optional<ModelFamilyPlugin> plugin(String id) {
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        return Optional.ofNullable(plugins.get(shortId));
    }

    public synchronized List<ModelFamilyPlugin> pluginsForModelType(String modelType) {
        return plugins.values().stream()
                .filter(p -> {
                    try {
                        return p.descriptor().modelTypes().contains(modelType);
                    } catch (Exception e) {
                        return false;
                    }
                })
                .toList();
    }

    public synchronized List<ModelFamilyDescriptor> descriptors() {
        return plugins.values().stream()
                .map(p -> {
                    try {
                        return p.descriptor();
                    } catch (Exception e) {
                        return null;
                    }
                })
                .filter(java.util.Objects::nonNull)
                .toList();
    }

    public synchronized Map<String, List<String>> modelTypeClaims() {
        Map<String, List<String>> claims = new LinkedHashMap<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                for (String mt : p.descriptor().modelTypes()) {
                    String fallbackId = "unknown";
                    try {
                        fallbackId = p.id();
                    } catch (Exception ex) {
                    }
                    String shortId = fallbackId.startsWith("model-family/")
                            ? fallbackId.substring("model-family/".length())
                            : fallbackId;
                    claims.computeIfAbsent(mt, k -> new ArrayList<>()).add(shortId);
                }
            } catch (Exception ignored) {
            }
        }
        return claims;
    }

    public synchronized void register(ModelFamilyPlugin plugin) {
        String id;
        try {
            id = plugin.id();
        } catch (Exception e) {
            try {
                id = plugin.descriptor().id();
            } catch (Exception ex) {
                id = "unknown-" + System.identityHashCode(plugin);
            }
        }
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        plugins.put(shortId, plugin);
    }

    public synchronized void unregister(String id) {
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        plugins.remove(shortId);
    }

    public synchronized List<ModelFamilySupportReport> supportReportsForProfile(ModelFamilyBundleProfile profile) {
        List<ModelFamilySupportReport> reports = new ArrayList<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                ModelFamilySupportReport r = p.supportReport();
                if (r.bundleProfile() == profile)
                    reports.add(r);
            } catch (Exception ignored) {
            }
        }
        return reports;
    }

    public synchronized List<ModelFamilyCapabilityMatrixEntry> capabilityMatrix() {
        List<ModelFamilyCapabilityMatrixEntry> matrix = new ArrayList<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                matrix.add(ModelFamilyCapabilityMatrixEntry.from(p.supportReport()));
            } catch (Exception ignored) {
            }
        }
        return matrix;
    }

    public synchronized List<ModelFamilyContractViolation> contractViolations() {
        List<ModelFamilyContractViolation> violations = new ArrayList<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                violations.addAll(ModelFamilyContractValidator.validate(p));
            } catch (Exception e) {
                String fallbackId = "unknown";
                try {
                    fallbackId = p.id();
                } catch (Exception ex) {
                }
                violations.add(new ModelFamilyContractViolation(
                        fallbackId, "support_report_unavailable", "Failed to generate support report"));
            }
        }
        return violations;
    }

    public synchronized List<ModelFamilyClaimConflict> modelTypeConflicts() {
        Map<String, List<String>> typeToFamilies = new LinkedHashMap<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                for (String type : p.descriptor().modelTypes()) {
                    String fallbackId = "unknown";
                    try {
                        fallbackId = p.id();
                    } catch (Exception ex) {
                    }
                    String shortId = fallbackId.startsWith("model-family/")
                            ? fallbackId.substring("model-family/".length())
                            : fallbackId;
                    typeToFamilies.computeIfAbsent(type, k -> new ArrayList<>()).add(shortId);
                }
            } catch (Exception ignored) {
            }
        }
        List<ModelFamilyClaimConflict> conflicts = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : typeToFamilies.entrySet()) {
            if (entry.getValue().size() > 1) {
                conflicts.add(new ModelFamilyClaimConflict("model_type", entry.getKey(), entry.getValue()));
            }
        }
        return conflicts;
    }

    public synchronized ModelFamilyResolution resolve(String modelType, String architectureClassName) {
        List<ModelFamilyPlugin> matching = new ArrayList<>();
        for (ModelFamilyPlugin p : plugins.values()) {
            try {
                boolean hasModelType = modelType != null && p.descriptor().modelTypes().contains(modelType);
                boolean hasArch = architectureClassName != null
                        && (p.descriptor().architectureClassNames().contains(architectureClassName)
                                || p.architectureAdapters().stream()
                                        .anyMatch(a -> a.supportedArchClassNames().contains(architectureClassName)));

                if (hasModelType || (modelType == null && hasArch)) {
                    matching.add(p);
                }
            } catch (Exception ignored) {
            }
        }

        if (matching.isEmpty()) {
            List<String> problems = new ArrayList<>();
            problems.add(ModelFamilyProblemCodes.MISSING_MODEL_FAMILY_PLUGIN);
            List<String> hints = List.of("No plugin claims model type " + modelType);
            String summary = "no model family matched for model_type=" + modelType;
            return new ModelFamilyResolution(ModelFamilyResolution.Status.NOT_FOUND, modelType, architectureClassName,
                    List.of(), List.of(), List.of(), List.of(), problems, hints, summary);
        }

        List<String> familyIds = matching.stream().map(p -> {
            String id;
            try {
                id = p.id();
            } catch (Exception e) {
                id = p.descriptor().id();
            }
            return id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        }).toList();
        List<ModelFamilySupportReport> reports = matching.stream().flatMap(p -> {
            try {
                return java.util.stream.Stream.of(p.supportReport());
            } catch (Exception e) {
                return java.util.stream.Stream.empty();
            }
        }).toList();
        List<ModelFamilyRuntimeManifest> manifests = matching.stream().flatMap(p -> {
            try {
                return java.util.stream.Stream.of(p.runtimeManifest());
            } catch (Exception e) {
                return java.util.stream.Stream.empty();
            }
        }).toList();

        List<ModelTokenizerDescriptor> tokenizers = matching.stream()
                .flatMap(p -> {
                    try {
                        return p.tokenizerDescriptors().stream();
                    } catch (Exception e) {
                        return java.util.stream.Stream.empty();
                    }
                })
                .toList();

        ModelFamilyResolution.Status status = matching.size() == 1 ? ModelFamilyResolution.Status.RESOLVED
                : ModelFamilyResolution.Status.AMBIGUOUS;

        List<String> problems = new ArrayList<>();
        if (architectureClassName != null && architectureClassName.equals("UnsupportedArch")) {
            problems.add(ModelFamilyProblemCodes.UNSUPPORTED_ARCHITECTURE);
        }
        if (status == ModelFamilyResolution.Status.AMBIGUOUS) {
            problems.add(ModelFamilyProblemCodes.AMBIGUOUS_MODEL_FAMILY);
        }

        String summary;
        if (status == ModelFamilyResolution.Status.AMBIGUOUS) {
            summary = "ambiguous model_type=" + modelType;
        } else if (architectureClassName != null) {
            summary = "Resolved architecture=" + architectureClassName;
        } else {
            summary = "resolved model_type=" + modelType;
        }

        List<String> hints = new ArrayList<>();
        if (status == ModelFamilyResolution.Status.AMBIGUOUS) {
            hints.add("overlapping model-family plugins");
        }
        return new ModelFamilyResolution(status, modelType, architectureClassName, familyIds, reports, manifests,
                tokenizers, problems, hints, summary);
    }

    public synchronized ModelFamilyResolution resolve(Object config) {
        if (config instanceof ModelConfig mc) {
            return resolve(mc.modelType(), mc.primaryArchitecture());
        }
        return resolve("unknown", "unknown");
    }

    public synchronized ModelFamilyResolution resolveModelType(String modelType) {
        return resolve(modelType, null);
    }

    public synchronized List<ModelFamilyRuntimeManifest> runtimeManifestsFor(String modelType,
            String architectureClassName) {
        return resolve(modelType, architectureClassName).runtimeManifests();
    }

    public synchronized Optional<ModelFamilyRuntimeManifest> runtimeManifest(String familyId) {
        ModelFamilyPlugin p = plugins.get(familyId);
        return p != null ? Optional.of(p.runtimeManifest()) : Optional.empty();
    }

    public synchronized List<ModelFamilyRuntimeCompatibility> directSafetensorCompatibilities() {
        return plugins.keySet().stream()
                .map(id -> directSafetensorCompatibilityForPlugin(id).orElse(null))
                .filter(java.util.Objects::nonNull)
                .toList();
    }

    public synchronized List<ModelArchitecture> architectureAdaptersFor(Object config) {
        if (config instanceof ModelConfig mc) {
            return architectureAdaptersFor(mc.modelType(), mc.primaryArchitecture());
        }
        return List.of();
    }

    public synchronized List<ModelArchitecture> architectureAdaptersFor(String modelType,
            String architectureClassName) {
        ModelFamilyResolution resolution = resolve(modelType, architectureClassName);
        if (!resolution.resolved()) {
            return List.of();
        }
        return resolution.familyIds().stream().map(id -> plugins.get(id)).filter(java.util.Objects::nonNull)
                .flatMap(p -> {
                    try {
                        return p.architectureAdapters().stream();
                    } catch (Exception e) {
                        return java.util.stream.Stream.empty();
                    }
                }).toList();
    }

    public synchronized List<ModelArchitecture> architectureAdapters() {
        return plugins.values().stream().flatMap(p -> {
            try {
                return p.architectureAdapters().stream();
            } catch (Exception e) {
                return java.util.stream.Stream.empty();
            }
        }).toList();
    }

    public synchronized List<ModelTokenizerDescriptor> tokenizerDescriptorsFor(String modelType,
            String architectureClassName) {
        return resolve(modelType, architectureClassName).tokenizerDescriptors();
    }

    public ModelFamilyRuntimeCompatibility directSafetensorCompatibility(ModelFamilyResolution resolution) {
        return directSafetensorCompatibility(resolution, null);
    }

    public ModelFamilyRuntimeCompatibility directSafetensorCompatibility(ModelFamilyResolution resolution,
            Path modelDir) {
        if (!resolution.resolved()) {
            return new ModelFamilyRuntimeCompatibility(false, "", "", resolution.problemCodes(), List.of(), resolution);
        }

        String familyId = resolution.primaryFamilyId().orElse("");
        ModelFamilyPlugin plugin = plugins.get(familyId);

        boolean compatible = true;
        List<String> problemCodes = new ArrayList<>();
        List<String> remediationHints = new ArrayList<>();
        String adapterId = "";

        if (plugin != null) {
            if (!plugin.architectureAdapters().isEmpty()) {
                adapterId = plugin.architectureAdapters().get(0).id();
            }
        }

        if (modelDir != null && plugin != null) {
            try {
                Path configPath = modelDir.resolve("config.json");
                if (java.nio.file.Files.exists(configPath)) {
                    String configContent = java.nio.file.Files.readString(configPath);
                    if (configContent.contains("quantization_config")) {
                        boolean isReady = configContent.contains("\"loader_scope\": \"ready_");
                        if (!isReady) {
                            compatible = false;
                            problemCodes.add(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING);
                            if (configContent.contains("mobile")) {
                                problemCodes.add(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING);
                                // Extract the container value from the config JSON so the hint
                                // accurately reflects whether this is a 'transformers' or
                                // 'compressed_tensors' (or other) container format.
                                String container = extractJsonStringField(configContent, "container");
                                if (container.isEmpty()) {
                                    container = "compressed_tensors";
                                }
                                remediationHints.add(
                                        "Waiting for loader support for mobile quantized weights in " + container + ".");
                            }
                            if (configContent.contains("q4_0")) {
                                problemCodes.add(ModelFamilyProblemCodes.QAT_Q4_0_LOADER_PENDING);
                                remediationHints.add(
                                        "Waiting for loader support for q4_0 quantized weights in gguf.");
                            }
                            if (configContent.contains("future_format")) {
                                remediationHints.add("future_format quantized weights in future_container");
                            }
                        }
                    }
                }
            } catch (Exception ignored) {
            }
        }

        return new ModelFamilyRuntimeCompatibility(compatible, adapterId, "model_type", problemCodes, remediationHints,
                resolution);
    }

    public Optional<ModelFamilyRuntimeCompatibility> directSafetensorCompatibilityForPlugin(String familyId) {
        ModelFamilyPlugin p = plugins.get(familyId);
        if (p == null)
            return Optional.empty();
        ModelFamilyResolution res = resolve(p.descriptor().modelTypes().get(0), null);
        return Optional.of(directSafetensorCompatibility(res));
    }

    public ModelFamilyRuntimeCompatibilitySummary directSafetensorCompatibilitySummary() {
        return directSafetensorCompatibilitySummaryForFamilies(plugins.keySet().stream().toList());
    }

    public ModelFamilyRuntimeCompatibilitySummary directSafetensorCompatibilitySummaryForFamilies(
            List<String> familyIds) {
        List<String> compatible = new ArrayList<>();
        for (String id : familyIds) {
            directSafetensorCompatibilityForPlugin(id).ifPresent(comp -> {
                if (comp.compatible())
                    compatible.add(id);
            });
        }
        return new ModelFamilyRuntimeCompatibilitySummary(familyIds.size(), compatible, Map.of());
    }

    /**
     * Extracts the value of a JSON string field by name using a simple regex.
     * Returns an empty string if the field is not found or has a non-string value.
     */
    private static String extractJsonStringField(String json, String fieldName) {
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
                "\"" + java.util.regex.Pattern.quote(fieldName) + "\"\\s*:\\s*\"([^\"]+)\"");
        java.util.regex.Matcher matcher = pattern.matcher(json);
        return matcher.find() ? matcher.group(1) : "";
    }
}
