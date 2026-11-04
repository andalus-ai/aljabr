package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GgufMetalDecodeStateTest {
    @Test void nativeDecodeIsAvailableInitially() {
        var state = new GgufMetalDecodeState();
        assertTrue(state.canUse(true));
        assertFalse(state.canUse(false));
        assertFalse(state.hasCommitted());
        assertFalse(state.requiresRestart());
    }
    @Test void failureBeforeFirstNativeCommitAllowsHostFallback() {
        var state = new GgufMetalDecodeState();
        assertEquals(GgufMetalDecodeState.RuntimeFailure.SAFE_HOST_FALLBACK, state.recordRuntimeFailure());
        assertFalse(state.canUse(true));
        assertFalse(state.hasCommitted());
        assertFalse(state.requiresRestart());
    }
    @Test void failureAfterNativeCommitRequiresFullContextRestart() {
        var state = new GgufMetalDecodeState();
        state.recordSuccessfulDecode();
        assertEquals(GgufMetalDecodeState.RuntimeFailure.CONTEXT_RESTART_REQUIRED, state.recordRuntimeFailure());
        assertTrue(state.hasCommitted());
        assertTrue(state.requiresRestart());
        assertFalse(state.canUse(true));
    }
    @Test void prefillSyncFailureDisablesNativeDecodeWithoutRestart() {
        var state = new GgufMetalDecodeState();
        state.disableBeforeCommit();
        assertFalse(state.canUse(true));
        assertFalse(state.hasCommitted());
        assertFalse(state.requiresRestart());
    }
}
