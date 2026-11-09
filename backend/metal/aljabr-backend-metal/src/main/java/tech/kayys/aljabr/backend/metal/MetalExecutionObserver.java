package tech.kayys.aljabr.backend.metal;

/** Receives per-operation execution decisions from a Metal compute backend. */
@FunctionalInterface
public interface MetalExecutionObserver {
    MetalExecutionObserver NO_OP = event -> { };

    void onExecution(MetalExecutionEvent event);
}
