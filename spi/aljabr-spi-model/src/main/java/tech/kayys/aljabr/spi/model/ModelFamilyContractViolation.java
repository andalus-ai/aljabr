package tech.kayys.aljabr.spi.model;

/**
 * A machine-readable contract violation detected by
 * {@link ModelFamilyContractValidator}
 * or {@link ModelFamilyFixtureValidator}.
 *
 * @param code    a stable, lowercase-underscore identifier for tooling (e.g.
 *                {@code "invalid_family_id"})
 * @param summary a human-readable description of the violation
 * @author bhangun
 */
public record ModelFamilyContractViolation(String familyId, String code, String summary) {

    public static ModelFamilyContractViolation of(String familyId, String code, String summary) {
        return new ModelFamilyContractViolation(familyId, code, summary);
    }

    // Kept for backward compatibility with existing validator if needed
    public static ModelFamilyContractViolation of(String code, String summary) {
        return new ModelFamilyContractViolation("", code, summary);
    }
}
