package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class GgufDecodeEligibilityTest {

    @Test
    void reportsExplicitlyDisabledDecodePath() {
        assertEquals(
                "disabled by -Daljabr.metal.decode.disable=true",
                GgufDecodeStep.fullGpuDecodeEligibilityFailure(true, false, false));
    }

    @Test
    void reportsMissingNativeDecodeEntryPoint() {
        assertEquals(
                "native Metal decode-step entry point is unavailable",
                GgufDecodeStep.fullGpuDecodeEligibilityFailure(false, false, false));
    }

    @Test
    void preservesSafetyGuardForPerLayerTokenEmbeddings() {
        assertEquals(
                "model uses per-layer token embeddings, which the native full-GPU decode step does not implement",
                GgufDecodeStep.fullGpuDecodeEligibilityFailure(false, true, true));
    }

    @Test
    void returnsNoFailureWhenAllInitialEligibilityChecksPass() {
        assertNull(GgufDecodeStep.fullGpuDecodeEligibilityFailure(false, true, false));
    }

    @Test
    void allowsPerOpFallbackWhenTheFirstNativeDecodeStepFails() {
        assertEquals(
                "native Metal decode-step failed before any native token was committed; per-op fallback is safe",
                GgufDecodeStep.fullGpuDecodeRuntimeFailureReason(false, false));
    }

    @Test
    void requiresDecoderRestartWhenNativeDecodeFailsAfterEarlierNativeTokens() {
        assertEquals(
                "native Metal decode-step failed after prior native tokens; the host KV cache is incomplete",
                GgufDecodeStep.fullGpuDecodeRuntimeFailureReason(false, true));
    }

    @Test
    void hasNoRuntimeFailureReasonWhenNativeDecodeSucceeds() {
        assertNull(GgufDecodeStep.fullGpuDecodeRuntimeFailureReason(true, false));
        assertNull(GgufDecodeStep.fullGpuDecodeRuntimeFailureReason(true, true));
    }
}
