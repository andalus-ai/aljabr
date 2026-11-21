package tech.kayys.aljabr.spi.model.ir;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Registry-backed, metadata-only planner for model format conversions.
 *
 * <p>Target format implementations are supplied through
 * {@link TargetFormatCapability}; this planner owns selection and consistent
 * handling of unknown targets, while each target owns its compatibility rules.
 * No tensor payload is opened or copied during planning.
 */
public final class ModelConversionPlanner {

    private final Map<String, TargetFormatCapability> capabilities;

    public ModelConversionPlanner(Collection<? extends TargetFormatCapability> providers) {
        Objects.requireNonNull(providers, "providers must not be null");
        Map<String, TargetFormatCapability> registered = new LinkedHashMap<>();
        for (TargetFormatCapability provider : providers) {
            Objects.requireNonNull(provider, "provider must not be null");
            String key = normalize(provider.targetFormat());
            TargetFormatCapability previous = registered.putIfAbsent(key, provider);
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate target format capability: " + key);
            }
        }
        this.capabilities = Map.copyOf(registered);
    }

    /**
     * Builds a conversion preflight plan for the requested target format.
     * Unknown targets produce an explicit UNSUPPORTED report rather than
     * silently falling back to a different writer.
     */
    public ConversionPlan plan(ModelDescriptor source, String targetFormat) {
        Objects.requireNonNull(source, "source must not be null");
        String target = normalize(targetFormat);
        TargetFormatCapability capability = capabilities.get(target);

        CompatibilityReport report;
        if (capability == null) {
            report = new CompatibilityReport(
                    CompatibilityReport.ConversionFidelity.UNSUPPORTED,
                    java.util.List.of(new CompatibilityReport.Diagnostic(
                            CompatibilityReport.Severity.ERROR,
                            "TARGET_FORMAT_UNAVAILABLE",
                            "No target capability provider is registered for format '" + target + "'")));
        } else {
            report = Objects.requireNonNull(
                    capability.assess(source),
                    "target capability provider returned a null compatibility report");
        }

        return new ConversionPlan(source.identity(), source.sourceFormat(), target, report);
    }

    public boolean supportsTarget(String targetFormat) {
        return capabilities.containsKey(normalize(targetFormat));
    }

    private static String normalize(String format) {
        if (format == null || format.isBlank()) {
            throw new IllegalArgumentException("targetFormat must not be blank");
        }
        return format.trim().toLowerCase(Locale.ROOT);
    }
}
