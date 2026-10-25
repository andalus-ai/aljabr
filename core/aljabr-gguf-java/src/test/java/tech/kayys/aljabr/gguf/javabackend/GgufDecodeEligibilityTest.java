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
    void disablesNativeDecodeWhenLogitsAreNotRequested() {
        assertEquals(
                "logits were not requested; per-op decode would desynchronize the native KV cache",
                GgufDecodeStep.fullGpuDecodeRuntimeFallbackReason(false, true));
    }

    @Test
    void disablesNativeDecodeAfterNativeStepFailure() {
        assertEquals(
                "native Metal decode-step invocation failed",
                GgufDecodeStep.fullGpuDecodeRuntimeFallbackReason(true, false));
    }

    @Test
    void retainsNativeDecodeWhenStepSucceedsAndLogitsAreRequested() {
        assertNull(GgufDecodeStep.fullGpuDecodeRuntimeFallbackReason(true, true));
    }
}
