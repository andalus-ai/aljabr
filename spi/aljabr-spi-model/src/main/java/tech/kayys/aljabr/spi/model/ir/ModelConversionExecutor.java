package tech.kayys.aljabr.spi.model.ir;

import java.io.IOException;

/**
 * Service-provider contract for executing a preflighted model conversion.
 *
 * <p>Implementations own source/target-specific payload resolution, writing,
 * and atomic commit behavior. They must reject unsupported plans and must not
 * silently discard data when a plan reports PARTIAL fidelity.
 */
public interface ModelConversionExecutor {

    /** Returns true when this executor can handle the requested format pair. */
    boolean supports(String sourceFormat, String targetFormat);

    /**
     * Executes a conversion described by the supplied AMIR model and plan.
     *
     * @throws IOException when validation, payload transfer, or output commit fails
     */
    ConversionExecutionResult execute(
            ModelDescriptor model,
            ConversionPlan plan,
            ConversionRequest request) throws IOException;
}
