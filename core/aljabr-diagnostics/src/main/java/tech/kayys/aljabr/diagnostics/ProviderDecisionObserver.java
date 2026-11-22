package tech.kayys.aljabr.diagnostics;

/** Synchronous observer for provider selection and fallback explanations. */
@FunctionalInterface
public interface ProviderDecisionObserver {

    /** Shared disabled observer; emits no records and allocates no events. */
    ProviderDecisionObserver NOOP = event -> {
        // Intentionally disabled.
    };

    void onDecision(ProviderDecisionEvent event);

    static ProviderDecisionObserver noop() {
        return NOOP;
    }
}
