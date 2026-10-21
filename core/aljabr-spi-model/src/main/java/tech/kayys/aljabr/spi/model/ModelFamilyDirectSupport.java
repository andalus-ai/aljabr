package tech.kayys.aljabr.spi.model;

/**
 * Readiness level for direct SafeTensor inference (no weight conversion
 * required).
 * @author bhangun
 */
public enum ModelFamilyDirectSupport {
    /** Direct SafeTensor inference is not yet ready. */
    PENDING,
    EXPERIMENTAL,
    DECLARED_NO_ADAPTER,
    READY
}
