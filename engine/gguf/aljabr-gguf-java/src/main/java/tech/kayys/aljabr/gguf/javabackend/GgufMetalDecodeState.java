package tech.kayys.aljabr.gguf.javabackend;

/** Tracks safe fallback from full-GPU decode to the host KV cache. */
final class GgufMetalDecodeState {
    enum RuntimeFailure { SAFE_HOST_FALLBACK, CONTEXT_RESTART_REQUIRED }
    private boolean disabled;
    private boolean committed;
    private boolean restartRequired;
    boolean canUse(boolean available) { return available && !disabled && !restartRequired; }
    boolean disabled() { return disabled; }
    boolean hasCommitted() { return committed; }
    boolean requiresRestart() { return restartRequired; }
    void recordSuccessfulDecode() { committed = true; }
    void disableBeforeCommit() { disabled = true; }
    RuntimeFailure recordRuntimeFailure() {
        disabled = true;
        if (committed) {
            restartRequired = true;
            return RuntimeFailure.CONTEXT_RESTART_REQUIRED;
        }
        return RuntimeFailure.SAFE_HOST_FALLBACK;
    }
}
