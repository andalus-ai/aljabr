package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNull;

class GgufDecodeEligibilityTest {

    @Test
    void keyProjectionOwnsKvCacheEvenWhenValueProjectionIsOmitted() {
        // Gemma 4 may reuse K as V when attn_v.weight is absent.
        assertTrue(GgufDecodeEligibility.hasOwnKvCache(true));
        assertFalse(GgufDecodeEligibility.hasOwnKvCache(false));
    }

    @Test
    void allowsKAsVProjectionLayoutForNativeDecode() {
        assertFalse(GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                new boolean[] {true, true},
                new boolean[] {true, false},
                new boolean[] {true, false},
                new int[] {256, 512},
                new int[] {8, 1}));
        assertNull(GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, false, false, false));
    }

    @Test
    void rejectsValueOnlyProjectionLayoutForNativeDecode() {
        assertTrue(GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                new boolean[] {true, false},
                new boolean[] {true, true},
                new boolean[] {true, false},
                new int[] {256, 512},
                new int[] {8, 1}));
        assertEquals(
                "model uses V-only projections or incompatible shared-KV geometry, which the native full-GPU decode step does not implement safely",
                GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, false, false, true));
    }

    @Test
    void allowsSharedKvOnlyWhenOwnerGeometryMatches() {
        assertFalse(GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                new boolean[] {true, false},
                new boolean[] {true, false},
                new boolean[] {true, false},
                new int[] {256, 256},
                new int[] {8, 8}));
        assertTrue(GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                new boolean[] {true, false},
                new boolean[] {true, false},
                new boolean[] {true, false},
                new int[] {256, 512},
                new int[] {8, 1}));
    }

    @Test
    void rejectsMismatchedSharedSwaKvOwnerGeometry() {
        assertTrue(GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                new boolean[] {true, true, false},
                new boolean[] {true, true, false},
                new boolean[] {true, false, true},
                new int[] {256, 512, 512},
                new int[] {8, 1, 1}));
    }

    @Test
    void reportsExplicitlyDisabledDecodePath() {
        assertEquals(
                "disabled by -Daljabr.metal.decode.disable=true",
                GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(true, false, false));
    }

    @Test
    void reportsMissingNativeDecodeEntryPoint() {
        assertEquals(
                "native Metal decode-step entry point is unavailable",
                GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, false, false));
    }

    @Test
    void preservesSafetyGuardForPerLayerTokenEmbeddings() {
        assertEquals(
                "model uses per-layer token embeddings, which the native full-GPU decode step does not implement",
                GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, true));
    }

    @Test
    void refusesFullGpuDecodeWhenModelUsesPartialRotaryDimensions() {
        assertEquals(
                "model uses per-layer partial RoPE, which the native full-GPU decode step does not implement",
                GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, false, true));
    }

    @Test
    void returnsNoFailureWhenAllInitialEligibilityChecksPass() {
        assertNull(GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, false));
        assertNull(GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(false, true, false, false));
    }

    @Test
    void allowsPerOpFallbackWhenTheFirstNativeDecodeStepFails() {
        assertEquals(
                "native Metal decode-step failed before any native token was committed; per-op fallback is safe",
                GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false, false));
    }

    @Test
    void requiresDecoderRestartWhenNativeDecodeFailsAfterEarlierNativeTokens() {
        assertEquals(
                "native Metal decode-step failed after prior native tokens; the host KV cache is incomplete",
                GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false, true));
    }

    @Test
    void hasNoRuntimeFailureReasonWhenNativeDecodeSucceeds() {
        assertNull(GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(true, false));
        assertNull(GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(true, true));
    }

    @Test
    void failedPromptKvSyncDisablesFullGpuDecodeBeforeGeneration() {
        assertEquals(
                "Metal KV-cache sync failed at layer 7 before generation; host KV cache is intact and per-op fallback is safe",
                GgufDecodeEligibility.fullGpuDecodeKvCacheSyncFailureReason(7));
    }

    @Test
    void absentKvSyncFailureLayerHasNoFailureReason() {
        assertNull(GgufDecodeEligibility.fullGpuDecodeKvCacheSyncFailureReason(-1));
    }
}
