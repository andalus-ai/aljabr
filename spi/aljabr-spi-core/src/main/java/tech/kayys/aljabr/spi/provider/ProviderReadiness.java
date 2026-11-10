package tech.kayys.aljabr.spi.provider;

/**
 * Individual lifecycle/evidence facts for a provider or device.
 * These values are not a single linear state machine; multiple facts may apply.
 */
public enum ProviderReadiness {
    DECLARED,
    DISCOVERED,
    INSTALLED,
    DEVICE_DETECTED,
    AVAILABLE,
    INITIALIZED,
    SESSION_OPEN,
    VALIDATED
}