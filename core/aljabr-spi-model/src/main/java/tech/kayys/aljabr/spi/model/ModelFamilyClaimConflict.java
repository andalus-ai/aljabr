package tech.kayys.aljabr.spi.model;

import java.util.List;
/**
 * Immutable record representing modelfamilyclaimconflict data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelFamilyClaimConflict(
        String claimType,
        String claim,
        List<String> familyIds) {

    public ModelFamilyClaimConflict {
        claimType = claimType == null ? "" : claimType;
        claim = claim == null ? "" : claim;
        familyIds = familyIds == null ? List.of() : List.copyOf(familyIds);
    }
}
