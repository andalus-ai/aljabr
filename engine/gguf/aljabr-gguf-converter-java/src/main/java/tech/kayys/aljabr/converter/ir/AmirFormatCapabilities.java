package tech.kayys.aljabr.converter.ir;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Declares the AMIR semantics a target container can preserve.
 *
 * <p>This is a format profile, not an executor feature switch. Profiles
 * describe persisted semantics so planning and execution can share the same
 * fidelity policy without duplicating format-specific checks.
 */
public record AmirFormatCapabilities(
        String format,
        Set<Feature> preservedFeatures,
        boolean requiresKnownArchitecture) {

    public enum Feature {
        TYPED_TENSOR_METADATA,
        TYPED_MODEL_METADATA,
        ARCHITECTURE_DESCRIPTOR,
        GRAPH_COLLECTION,
        ASSET_COLLECTION,
        COMPONENT_COLLECTION,
        OPAQUE_MODEL_EXTENSIONS,
        OPAQUE_TENSOR_EXTENSIONS
    }

    public AmirFormatCapabilities {
        format = Objects.requireNonNull(format, "format must not be null")
                .trim().toLowerCase(Locale.ROOT);
        if (format.isEmpty()) {
            throw new IllegalArgumentException("format must not be blank");
        }
        preservedFeatures = Set.copyOf(Objects.requireNonNull(
                preservedFeatures, "preservedFeatures must not be null"));
    }

    public boolean preserves(Feature feature) {
        return preservedFeatures.contains(Objects.requireNonNull(feature, "feature must not be null"));
    }

    /** Capabilities of the AMIR Safetensors envelope writer. */
    public static AmirFormatCapabilities safetensors() {
        return new AmirFormatCapabilities("safetensors", EnumSet.of(
                Feature.TYPED_TENSOR_METADATA,
                Feature.TYPED_MODEL_METADATA,
                Feature.ARCHITECTURE_DESCRIPTOR,
                Feature.GRAPH_COLLECTION,
                Feature.ASSET_COLLECTION,
                Feature.COMPONENT_COLLECTION), true);
    }

    /** Capabilities of the AMIR GGUF metadata-envelope writer. */
    public static AmirFormatCapabilities gguf() {
        return new AmirFormatCapabilities("gguf", EnumSet.allOf(Feature.class), true);
    }

    /**
     * Resolves a built-in target profile.
     *
     * @throws IllegalArgumentException when no capability profile is registered
     */
    public static AmirFormatCapabilities forTarget(String format) {
        Objects.requireNonNull(format, "format must not be null");
        return switch (format.trim().toLowerCase(Locale.ROOT)) {
            case "safetensors" -> safetensors();
            case "gguf" -> gguf();
            default -> throw new IllegalArgumentException(
                    "No AMIR fidelity capability profile is registered for target format '" + format + "'");
        };
    }
}
