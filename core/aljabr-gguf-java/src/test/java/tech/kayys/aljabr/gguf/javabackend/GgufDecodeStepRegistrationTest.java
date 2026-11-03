package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufDecodeStepRegistrationTest {

    @Test
    void failedLayerRegistrationDisablesFullMetalDecode() {
        assertFalse(GgufMetalDecodeRegistration.decodeLayerRegistrationComplete(false, null, false));
    }

    @Test
    void missingRequiredRopeBufferDisablesFullMetalDecode() {
        float[] proportionalRopeFreqs = {1.0f, 0.5f};

        assertFalse(GgufMetalDecodeRegistration.decodeLayerRegistrationComplete(
                true, proportionalRopeFreqs, false));
    }

    @Test
    void successfullyRegisteredRopeBufferAllowsFullMetalDecode() {
        float[] proportionalRopeFreqs = {1.0f, 0.5f};

        assertTrue(GgufMetalDecodeRegistration.decodeLayerRegistrationComplete(
                true, proportionalRopeFreqs, true));
    }

    @Test
    void absentRopeBufferDoesNotRequireRopeRegistration() {
        assertTrue(GgufMetalDecodeRegistration.decodeLayerRegistrationComplete(true, null, false));
    }
}
