package tech.kayys.aljabr.gguf.api;

import tech.kayys.aljabr.spi.inference.InferenceRequest;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Resolves which backend the caller asked for, purely as a normalized
 * string token.
 */
public record GgufBackendSelection(
        String requestedValue,
        String normalizedValue,
        String source,
        boolean explicit) {

    private static final String DEFAULT_SOURCE = "default:auto";

    public static GgufBackendSelection resolve(Map<String, Object> metadata) {
        Candidate candidate = firstCandidate(metadata);
        if (candidate.value().isEmpty()) {
            return auto(DEFAULT_SOURCE, "");
        }

        String raw = candidate.value().get();
        String normalized = normalize(raw);
        if (normalized.isEmpty() || normalized.equals("auto") || normalized.equals("default")) {
            return auto(candidate.source(), raw);
        }
        return new GgufBackendSelection(raw, normalized, candidate.source(), true);
    }

    public static GgufBackendSelection resolve(InferenceRequest request) {
        if (request == null) {
            return auto(DEFAULT_SOURCE, "");
        }
        return resolve(request.getMetadata());
    }

    private static GgufBackendSelection auto(String source, String raw) {
        return new GgufBackendSelection(raw, "auto", source, false);
    }

    private static Candidate firstCandidate(Map<String, Object> requestMetadata) {
        if (requestMetadata == null || requestMetadata.isEmpty()) {
            return new Candidate(Optional.empty(), "none");
        }

        Optional<String> requestBackend = stringValue(requestMetadata.get("gguf.backend"))
                .or(() -> stringValue(requestMetadata.get("backend")));
        if (requestBackend.isPresent()) {
            return new Candidate(requestBackend, "request.metadata.gguf.backend");
        }

        return new Candidate(Optional.empty(), "none");
    }

    private static Optional<String> stringValue(Object value) {
        if (value == null) {
            return Optional.empty();
        }
        String s = value.toString().trim();
        return s.isEmpty() ? Optional.empty() : Optional.of(s);
    }

    private static String normalize(String value) {
        if (value == null) {
            return "";
        }
        return value.trim().toLowerCase(Locale.ROOT);
    }

    private record Candidate(Optional<String> value, String source) {}
}
