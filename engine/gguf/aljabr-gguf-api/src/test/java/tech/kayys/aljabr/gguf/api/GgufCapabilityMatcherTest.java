package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class GgufCapabilityMatcherTest {

    @Test
    void textGenerationProfileMatchesTextProvider() {
        GgufCapabilitySet required = GgufCapabilitySet.textGeneration();
        GgufCapabilityMatch match = GgufCapabilityMatcher.match(
                required, GgufCapabilitySet.textGeneration());

        assertTrue(match.compatible());
        assertTrue(match.diagnostic().contains("satisfies"));
    }

    @Test
    void reportsMissingInputOutputAndTaskCapabilitiesIndependently() {
        GgufCapabilitySet required = new GgufCapabilitySet(
                Set.of(GgufModality.TEXT, GgufModality.AUDIO),
                Set.of(GgufModality.TEXT, GgufModality.AUDIO),
                Set.of(GgufTask.TEXT_GENERATION, GgufTask.AUDIO_PERCEPTION));
        GgufCapabilitySet provider = GgufCapabilitySet.textGeneration();

        GgufCapabilityMatch match = GgufCapabilityMatcher.match(required, provider);

        assertFalse(match.compatible());
        assertEquals(Set.of(GgufModality.AUDIO), match.missingInputModalities());
        assertEquals(Set.of(GgufModality.AUDIO), match.missingOutputModalities());
        assertEquals(Set.of(GgufTask.AUDIO_PERCEPTION), match.missingTasks());
        assertTrue(match.diagnostic().contains("AUDIO_PERCEPTION"));
    }

    @Test
    void emptyRequirementIsCompatibleWithAnyProvider() {
        assertTrue(GgufCapabilityMatcher.match(
                GgufCapabilitySet.none(), GgufCapabilitySet.none()).compatible());
    }

    @Test
    void capabilitySetsAreDefensivelyCopied() {
        var inputs = new java.util.HashSet<>(Set.of(GgufModality.TEXT));
        GgufCapabilitySet capabilities = new GgufCapabilitySet(
                inputs, Set.of(GgufModality.TEXT), Set.of(GgufTask.TEXT_GENERATION));
        inputs.add(GgufModality.AUDIO);

        assertEquals(Set.of(GgufModality.TEXT), capabilities.inputModalities());
        assertThrows(UnsupportedOperationException.class,
                () -> capabilities.inputModalities().add(GgufModality.AUDIO));
    }

    @Test
    void rejectsNullProfiles() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufCapabilityMatcher.match(null, GgufCapabilitySet.none()));
        assertThrows(IllegalArgumentException.class,
                () -> GgufCapabilityMatcher.match(GgufCapabilitySet.none(), null));
    }
}
