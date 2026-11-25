package tech.kayys.aljabr.gguf.api;

import java.util.Map;
import java.util.Set;

/**
 * A backend's answer to whether it can execute a specific GGUF model.
 *
 * <p>Providers should derive this profile from inexpensive GGUF metadata and
 * tensor inventory inspection. Probing must not load full weights or execute
 * the model.</p>
 *
 * <p>The five-argument constructor is retained for source compatibility.
 * Legacy ready capabilities conservatively advertise text generation only;
 * providers must use the six-argument constructor to advertise other tasks.</p>
 */
public record GgufBackendCapability(
        boolean generationReady,
        String architecture,
        Set<String> quantizationTypes,
        String reason,
        Map<String, Object> diagnostics,
        GgufCapabilitySet capabilities) {

    public GgufBackendCapability {
        quantizationTypes = quantizationTypes == null ? Set.of() : Set.copyOf(quantizationTypes);
        diagnostics = diagnostics == null ? Map.of() : Map.copyOf(diagnostics);
        capabilities = capabilities == null
                ? (generationReady ? GgufCapabilitySet.textGeneration() : GgufCapabilitySet.none())
                : capabilities;
    }

    /**
     * Backward-compatible constructor for providers written against the
     * original five-component capability record.
     */
    public GgufBackendCapability(
            boolean generationReady,
            String architecture,
            Set<String> quantizationTypes,
            String reason,
            Map<String, Object> diagnostics) {
        this(generationReady, architecture, quantizationTypes, reason, diagnostics, null);
    }

    public static GgufBackendCapability ready(
            String architecture, Set<String> quantizationTypes, Map<String, Object> diagnostics) {
        return ready(architecture, quantizationTypes, diagnostics, GgufCapabilitySet.textGeneration());
    }

    public static GgufBackendCapability ready(
            String architecture,
            Set<String> quantizationTypes,
            Map<String, Object> diagnostics,
            GgufCapabilitySet capabilities) {
        return new GgufBackendCapability(
                true, architecture, quantizationTypes, null, diagnostics, capabilities);
    }

    public static GgufBackendCapability notReady(
            String architecture, Set<String> quantizationTypes, String reason, Map<String, Object> diagnostics) {
        return new GgufBackendCapability(
                false, architecture, quantizationTypes, reason, diagnostics, GgufCapabilitySet.none());
    }

    public static GgufBackendCapability notReady(
            String architecture,
            Set<String> quantizationTypes,
            String reason,
            Map<String, Object> diagnostics,
            GgufCapabilitySet detectedCapabilities) {
        return new GgufBackendCapability(
                false, architecture, quantizationTypes, reason, diagnostics, detectedCapabilities);
    }
}
