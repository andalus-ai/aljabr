package tech.kayys.aljabr.gguf.api;

import java.util.List;
import java.util.Objects;

/**
 * Immutable result of model-family artifact inspection.
 *
 * <p>This is an artifact-compatibility result only. A MATCH does not imply
 * that a backend can execute the model or any particular modality.</p>
 */
public record GgufModelFamilyInspection(
        String familyId,
        Status status,
        List<String> diagnostics) {

    public enum Status {
        MATCH,
        UNSUPPORTED_ARCHITECTURE,
        INCOMPLETE_ARTIFACT,
        UNSUPPORTED_ARTIFACT
    }

    public GgufModelFamilyInspection {
        familyId = requireText(familyId, "familyId");
        Objects.requireNonNull(status, "status");
        diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics"));
        diagnostics.forEach(value -> requireText(value, "diagnostic"));
        if (status == Status.MATCH && !diagnostics.isEmpty()) {
            throw new IllegalArgumentException("A MATCH result cannot contain failure diagnostics");
        }
        if (status != Status.MATCH && diagnostics.isEmpty()) {
            throw new IllegalArgumentException("A non-MATCH result requires at least one diagnostic");
        }
    }

    public boolean matched() {
        return status == Status.MATCH;
    }

    public static GgufModelFamilyInspection match(String familyId) {
        return new GgufModelFamilyInspection(familyId, Status.MATCH, List.of());
    }

    public static GgufModelFamilyInspection rejected(
            String familyId, Status status, List<String> diagnostics) {
        if (status == Status.MATCH) {
            throw new IllegalArgumentException("Use match() for a successful inspection");
        }
        return new GgufModelFamilyInspection(familyId, status, diagnostics);
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return normalized;
    }
}
