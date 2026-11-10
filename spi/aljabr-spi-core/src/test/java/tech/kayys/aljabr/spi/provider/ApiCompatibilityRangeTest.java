package tech.kayys.aljabr.spi.provider;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Set;
import org.junit.jupiter.api.Test;

class ApiCompatibilityRangeTest {
    @Test void includesLowerBoundAndExcludesUpperBound() {
        var range = ApiCompatibilityRange.parse(">=0.1.0 <1.0.0");
        assertTrue(range.includes("0.1.0"));
        assertTrue(range.includes("0.9.99"));
        assertFalse(range.includes("1.0.0"));
        assertFalse(range.includes("0.0.99"));
    }

    @Test void rejectsUnsupportedOrMalformedExpressions() {
        assertThrows(IllegalArgumentException.class, () -> ApiCompatibilityRange.parse("^0.1"));
        assertThrows(IllegalArgumentException.class, () -> ApiCompatibilityRange.parse(">=1.0.0 <1.0.0"));
        assertThrows(IllegalArgumentException.class, () -> ApiCompatibilityRange.parse(">=01.0.0 <2.0.0"));
        assertThrows(IllegalArgumentException.class, () -> ApiCompatibilityRange.parse(">=1.0 <2.0"));
    }

    @Test void rejectsDuplicateCapabilityIdsEvenWhenMetadataDiffers() {
        var first = new CapabilityDescriptor("linear-algebra.dense", "1.0.0", CapabilityStability.STABLE);
        var second = new CapabilityDescriptor("linear-algebra.dense", "2.0.0", CapabilityStability.PREVIEW);
        assertThrows(IllegalArgumentException.class, () -> new ProviderDescriptor(
                "andalus/cpu-reference", "0.1.0", ">=0.1.0 <1.0.0", Set.of(first, second)));
    }

    @Test void rejectsNonCanonicalCapabilityIds() {
        assertThrows(IllegalArgumentException.class,
                () -> new CapabilityDescriptor("Linear.Algebra", "1.0.0", CapabilityStability.STABLE));
    }
}
