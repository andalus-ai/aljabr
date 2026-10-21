package tech.kayys.aljabr.spi.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Validates the contract between a {@link ModelFamilyPlugin} declaration and
 * its
 * runtime surface (adapters, tokenizers, support report).
 *
 * <p>
 * Call {@link #validate(ModelFamilyPlugin)} for a single plugin or
 * {@link #validateAll(List)} to detect cross-plugin conflicts.
 * @author bhangun
 */
public final class ModelFamilyContractValidator {

    /** Pattern for valid lowercase-underscore identifiers. */
    private static final java.util.regex.Pattern SAFE_ID = java.util.regex.Pattern.compile("[a-z][a-z0-9_]*");

    /** Valid origin path prefix. */
    private static final String ORIGIN_PREFIX = "3rdparty/";

    private ModelFamilyContractValidator() {
    }

    // ── Public API ────────────────────────────────────────────────────

    /**
     * Validates a single plugin against its declared contract.
     *
     * @param plugin the plugin to validate; must not be {@code null}
     * @return a (possibly empty) list of detected violations
     */
    public static List<ModelFamilyContractViolation> validate(ModelFamilyPlugin plugin) {
        List<ModelFamilyContractViolation> v = new ArrayList<>();
        ModelFamilyDescriptor desc = plugin.descriptor();

        checkFamilyId(v, desc);
        checkPluginId(v, plugin, desc);
        checkModelTypes(v, desc);
        checkBundleProfile(v, desc);
        checkOrigin(v, desc);
        checkMultimodal(v, desc);
        checkTokenizerDescriptors(v, plugin, desc);
        checkDirectSafetensor(v, plugin, desc);
        checkScopedDirectSafetensorReasons(v, desc);
        checkArchitectureAdapters(v, plugin, desc);
        checkUnifiedRuntimeRequirements(v, plugin, desc);
        checkSupportReport(v, plugin);

        return List.copyOf(v);
    }

    /**
     * Validates a list of plugins and additionally detects cross-plugin conflicts
     * (e.g. duplicate model-type claims).
     *
     * @param plugins the plugins to validate; must not be {@code null}
     * @return a (possibly empty) list of all detected violations
     */
    public static List<ModelFamilyContractViolation> validateAll(List<ModelFamilyPlugin> plugins) {
        List<ModelFamilyContractViolation> v = new ArrayList<>();
        for (ModelFamilyPlugin p : plugins) {
            v.addAll(validate(p));
        }
        checkDuplicateModelTypeClaims(v, plugins);
        return List.copyOf(v);
    }

    // ── Individual checks ─────────────────────────────────────────────

    private static void checkFamilyId(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        if (!SAFE_ID.matcher(desc.id()).matches()) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "invalid_family_id",
                    "Family id '" + desc.id() + "' must match [a-z][a-z0-9_]*"));
        }
    }

    private static void checkPluginId(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin, ModelFamilyDescriptor desc) {
        String id;
        try {
            id = plugin.id();
        } catch (Exception e) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "plugin_id_mismatch",
                    "plugin.id() threw an exception: " + e.getMessage()));
            return;
        }
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        if (!shortId.equals(desc.id())) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "plugin_id_mismatch",
                    "plugin.id() '" + shortId + "' does not match descriptor id '" + desc.id() + "'"));
        }
    }

    private static final java.util.regex.Pattern SAFE_MODEL_TYPE = java.util.regex.Pattern.compile("[a-z][a-z0-9_.-]*");

    private static void checkModelTypes(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        for (String type : desc.modelTypes()) {
            if (!SAFE_MODEL_TYPE.matcher(type).matches()) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "invalid_model_type",
                        "Model type '" + type + "' must match [a-z][a-z0-9_.-]*"));
            }
        }
    }

    private static void checkBundleProfile(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        String profileKey = desc.metadata().get("bundle_profile");
        if (profileKey != null && ModelFamilyBundleProfile.fromKey(profileKey) == null) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "unknown_bundle_profile",
                    "Bundle profile '" + profileKey + "' is not a recognised value "
                            + "(expected: core, optional, experimental)"));
        }
    }

    private static void checkOrigin(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        String origin = desc.metadata().get("origin");
        if (origin == null || origin.isBlank()) {
            origin = desc.metadata().get("modeling_origin");
        }
        if (origin == null || origin.isBlank()) {
            return;
        }
        if (origin.contains(" ") || origin.contains("\t")) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "origin_contains_whitespace",
                    "Origin path '" + origin + "' must not contain whitespace"));
        }
        if (!origin.startsWith(ORIGIN_PREFIX)) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "unexpected_origin_path",
                    "Origin '" + origin + "' must start with '" + ORIGIN_PREFIX + "'"));
        }
    }

    private static void checkMultimodal(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        if (desc.capabilities().contains(ModelFamilyCapability.MULTIMODAL)) {
            // Require at least one modality declaration in metadata
            boolean hasModality = desc.metadata().containsKey("modalities")
                    || desc.metadata().containsKey("modality");
            if (!hasModality) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "multimodal_without_modality",
                        "MULTIMODAL capability declared but no 'modalities' key in metadata"));
            }
        }
    }

    private static void checkTokenizerDescriptors(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin, ModelFamilyDescriptor desc) {
        boolean hasTokenizerCap = desc.capabilities().contains(ModelFamilyCapability.TOKENIZER);
        String metaStatus = desc.metadata().get("tokenizer_metadata_status");
        String metaPendingReason = desc.metadata().get("tokenizer_metadata_pending_reason");
        boolean isPending = "pending".equalsIgnoreCase(metaStatus);
        boolean isReady = "ready".equalsIgnoreCase(metaStatus);

        // Validate tokenizer descriptor IDs
        for (ModelTokenizerDescriptor td : plugin.tokenizerDescriptors()) {
            if (!SAFE_ID.matcher(td.id().replace("-", "_")).matches()) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "invalid_tokenizer_id",
                        "Tokenizer id '" + td.id() + "' contains invalid characters"));
            }
            // Check for path traversal in required files
            for (List<String> group : td.requiredFileGroups()) {
                for (String file : group) {
                    if (file.contains("..")) {
                        v.add(ModelFamilyContractViolation.of(desc.id(), "tokenizer_unsafe_file",
                                "Tokenizer file '" + file + "' contains an unsafe path segment ('..')"));
                    }
                }
            }
        }

        if (hasTokenizerCap && plugin.tokenizerDescriptors().isEmpty()) {
            if (!isPending) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "tokenizer_capability_without_descriptor",
                        "TOKENIZER capability declared but no tokenizer descriptors provided"));
            }
        }

        // Metadata consistency
        if (isPending && (metaPendingReason == null || metaPendingReason.isBlank())) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "tokenizer_metadata_pending_reason_missing",
                    "tokenizer_metadata_status is 'pending' but tokenizer_metadata_pending_reason is missing"));
        }
        if (isReady && plugin.tokenizerDescriptors().isEmpty()) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "tokenizer_metadata_ready_without_descriptor",
                    "tokenizer_metadata_status is 'ready' but no tokenizer descriptors provided"));
        }
        if (metaPendingReason != null && !isPending) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "tokenizer_metadata_pending_reason_without_pending_status",
                    "tokenizer_metadata_pending_reason is set but tokenizer_metadata_status is not 'pending'"));
        }
    }

    private static void checkDirectSafetensor(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin, ModelFamilyDescriptor desc) {
        if (desc.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                && plugin.architectureAdapters().isEmpty()) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "direct_safetensor_without_adapter",
                    "DIRECT_SAFETENSOR_INFERENCE declared but no architecture adapters provided"));
        }
    }

    private static void checkScopedDirectSafetensorReasons(List<ModelFamilyContractViolation> v,
            ModelFamilyDescriptor desc) {
        for (Map.Entry<String, String> entry : desc.metadata().entrySet()) {
            if (entry.getKey().endsWith("_direct_safetensor")) {
                String value = entry.getValue();
                // Value must start with "pending_" to be a valid scoped reason
                if (!value.startsWith("pending_")) {
                    v.add(ModelFamilyContractViolation.of(desc.id(), "invalid_scoped_direct_safetensor_reason",
                            "Scoped direct safetensor key '" + entry.getKey()
                                    + "' has invalid value '" + value
                                    + "' (must start with 'pending_')"));
                }
            }
        }
    }

    private static void checkArchitectureAdapters(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin, ModelFamilyDescriptor desc) {
        Set<String> declaredModelTypes = new HashSet<>(desc.modelTypes());
        Set<String> declaredArchNames = new HashSet<>(desc.architectureClassNames());

        for (ModelArchitecture adapter : plugin.architectureAdapters()) {
            boolean matchesModelType = adapter.supportedModelTypes().stream()
                    .anyMatch(declaredModelTypes::contains);
            boolean matchesArchClass = adapter.supportedArchClassNames().stream()
                    .anyMatch(declaredArchNames::contains);

            if (!matchesModelType && !matchesArchClass) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "architecture_adapter_unclaimed",
                        "Architecture adapter '" + adapter.id() + "' does not match any declared "
                                + "model type or architecture class name in the descriptor"));
            }
        }
    }

    private static void checkUnifiedRuntimeRequirements(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin, ModelFamilyDescriptor desc) {
        boolean hasMultimodal = desc.capabilities().contains(ModelFamilyCapability.MULTIMODAL)
                || desc.capabilities().contains(ModelFamilyCapability.VISION);

        for (ModelFamilyUnifiedRuntimeRequirement req : plugin.unifiedRuntimeRequirements()) {
            if (req.metadata().isEmpty()) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "unified_runtime_requirement_missing_metadata",
                        "Unified runtime requirement for '" + req.modelType() + "' has no metadata"));
            }
            if (!desc.modelTypes().contains(req.modelType())) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "unified_runtime_requirement_unclaimed_model_type",
                        "Unified runtime requirement references model type '" + req.modelType()
                                + "' which is not declared in the descriptor"));
            }
            if (hasMultimodal && req.requiredInputModalities().isEmpty()) {
                v.add(ModelFamilyContractViolation.of(desc.id(), "unified_runtime_requirement_missing_modalities",
                        "Unified runtime requirement for '" + req.modelType()
                                + "' must declare at least one modality when MULTIMODAL/VISION is in capabilities"));
            }
        }
    }

    private static void checkSupportReport(List<ModelFamilyContractViolation> v,
            ModelFamilyPlugin plugin) {
        ModelFamilyDescriptor desc = plugin.descriptor();
        ModelFamilySupportReport report;
        try {
            report = plugin.supportReport();
        } catch (Exception e) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_unavailable",
                    "supportReport() threw: " + e.getMessage()));
            return;
        }
        if (report == null) {
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_unavailable",
                    "supportReport() returned null"));
            return;
        }

        // Derive expected values from plugin surface
        List<String> expectedArchClassNames = plugin.architectureAdapters().stream()
                .flatMap(a -> a.supportedArchClassNames().stream()).distinct().toList();
        List<String> expectedAdapterIds = plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id).toList();
        List<String> expectedTokenizerIds = plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id).toList();
        List<ModelTokenizerKind> expectedTokenizerKinds = plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::kind).distinct().toList();
        String profileKey = desc.metadata().getOrDefault("bundle_profile", "");
        ModelFamilyBundleProfile expectedProfile = ModelFamilyBundleProfile.fromKey(profileKey);
        boolean hasDirect = desc.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                && !plugin.architectureAdapters().isEmpty();
        ModelFamilyDirectSupport expectedDirect = hasDirect
                ? ModelFamilyDirectSupport.READY
                : ModelFamilyDirectSupport.PENDING;

        if (!desc.id().equals(report.id()))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_id_mismatch",
                    "Report id '" + report.id() + "' != descriptor id '" + desc.id() + "'"));
        if (!desc.displayName().equals(report.displayName()))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_display_name_mismatch",
                    "Report displayName '" + report.displayName() + "' != '" + desc.displayName() + "'"));
        if (!sorted(desc.modelTypes()).equals(sorted(report.modelTypes())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_model_types_mismatch",
                    "Report modelTypes " + report.modelTypes() + " != descriptor " + desc.modelTypes()));
        if (!sorted(expectedArchClassNames).equals(sorted(report.architectureClassNames())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_architectures_mismatch",
                    "Report architectures " + report.architectureClassNames() + " != expected "
                            + expectedArchClassNames));
        if (!sorted(expectedAdapterIds).equals(sorted(report.architectureAdapterIds())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_architecture_adapters_mismatch",
                    "Report adapterIds " + report.architectureAdapterIds() + " != expected " + expectedAdapterIds));
        if (!sorted(expectedTokenizerIds).equals(sorted(report.tokenizerProfileIds())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_tokenizers_mismatch",
                    "Report tokenizerIds " + report.tokenizerProfileIds() + " != expected " + expectedTokenizerIds));
        if (!sortedKinds(expectedTokenizerKinds).equals(sortedKinds(report.tokenizerKinds())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_tokenizer_kinds_mismatch",
                    "Report tokenizerKinds " + report.tokenizerKinds() + " != expected " + expectedTokenizerKinds));
        if (!sortedCaps(desc.capabilities()).equals(sortedCaps(report.capabilities())))
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_capabilities_mismatch",
                    "Report capabilities " + report.capabilities() + " != descriptor " + desc.capabilities()));
        if (expectedProfile != null && report.bundleProfile() != expectedProfile)
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_bundle_profile_mismatch",
                    "Report bundleProfile " + report.bundleProfile() + " != expected " + expectedProfile));
        if (report.directSafetensorStatus() != expectedDirect)
            v.add(ModelFamilyContractViolation.of(desc.id(), "support_report_direct_safetensor_mismatch",
                    "Report directSupport " + report.directSafetensorStatus() + " != expected " + expectedDirect));
    }

    private static void checkDuplicateModelTypeClaims(List<ModelFamilyContractViolation> v,
            List<ModelFamilyPlugin> plugins) {
        Map<String, List<String>> typeToFamilies = new java.util.HashMap<>();
        for (ModelFamilyPlugin p : plugins) {
            for (String type : p.descriptor().modelTypes()) {
                typeToFamilies.computeIfAbsent(type.toLowerCase(java.util.Locale.ROOT),
                        k -> new ArrayList<>()).add(p.descriptor().id());
            }
        }
        for (Map.Entry<String, List<String>> e : typeToFamilies.entrySet()) {
            if (e.getValue().size() > 1) {
                v.add(ModelFamilyContractViolation.of("", "duplicate_model_type_claim",
                        "Model type '" + e.getKey() + "' is claimed by multiple families: "
                                + e.getValue()));
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private static List<String> sorted(List<String> list) {
        return list.stream().sorted().toList();
    }

    private static List<String> sortedKinds(List<ModelTokenizerKind> list) {
        return list.stream().map(Enum::name).sorted().toList();
    }

    private static List<String> sortedCaps(List<ModelFamilyCapability> list) {
        return list.stream().map(Enum::name).sorted().toList();
    }
}
