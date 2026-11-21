package tech.kayys.aljabr.spi.model.ir;

/**
 * Preflight capability contract implemented by a target format adapter.
 *
 * <p>Assessment must be metadata-only: implementations must not read tensor
 * payload bytes, allocate output files, or mutate the source document.
 */
public interface TargetFormatCapability {

    /** Stable, case-insensitive target format identifier (for example, gguf). */
    String targetFormat();

    /**
     * Assesses whether the target can represent the source model and reports
     * every known loss, unsupported feature, or required transformation.
     */
    CompatibilityReport assess(ModelDescriptor source);
}
