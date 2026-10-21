package tech.kayys.aljabr.gguf.api;

import java.util.Map;
import java.util.Set;

/**
 * A backend's answer to "can you actually generate for this specific model?"
 *
 * <p>{@link GgufBackendProvider#probe} must answer this from GGUF header
 * metadata only (architecture string, tensor names/types, quantization
 * formats present) — never by loading full tensor data or running the
 * model — so that {@link GgufInferenceEngine}'s AUTO selection stays cheap
 * even for large models and can probe every available provider on every
 * load.</p>
 *
 * <p>{@code diagnostics} is an open bag for whatever readiness signals a
 * backend already tracks internally (e.g. the Java engine's
 * {@code rowDotReady} / {@code decoderTensorsReady} / {@code knownTensorTypeRatio}
 * flags) so they keep flowing through to {@code ModelHandle} metadata for
 * operational visibility, without the API module needing to know what any
 * particular backend tracks.</p>
 */
public record GgufBackendCapability(
        boolean generationReady,
        String architecture,
        Set<String> quantizationTypes,
        String reason,
        Map<String, Object> diagnostics) {

    public GgufBackendCapability {
        quantizationTypes = quantizationTypes == null ? Set.of() : Set.copyOf(quantizationTypes);
        diagnostics = diagnostics == null ? Map.of() : Map.copyOf(diagnostics);
    }

    public static GgufBackendCapability ready(
            String architecture, Set<String> quantizationTypes, Map<String, Object> diagnostics) {
        return new GgufBackendCapability(true, architecture, quantizationTypes, null, diagnostics);
    }

    public static GgufBackendCapability notReady(
            String architecture, Set<String> quantizationTypes, String reason, Map<String, Object> diagnostics) {
        return new GgufBackendCapability(false, architecture, quantizationTypes, reason, diagnostics);
    }
}
