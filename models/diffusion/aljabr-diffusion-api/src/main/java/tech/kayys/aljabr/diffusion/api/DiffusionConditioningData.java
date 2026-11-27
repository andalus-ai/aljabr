package tech.kayys.aljabr.diffusion.api;

/**
 * Owned result of preparing one request conditioning input.
 *
 * <p>Implementations may expose strongly typed data through integration-
 * specific subinterfaces (for example decoded image tensors or tokenized
 * prompt data). The generic API only tracks provenance and lifetime; it does
 * not prescribe an in-memory representation.</p>
 *
 * <p>The caller that receives this value owns it and must close it. Closing
 * releases resources acquired during preparation.</p>
 */
public interface DiffusionConditioningData extends AutoCloseable {

    String inputId();

    DiffusionConditioningInput.Modality modality();

    @Override
    void close() throws Exception;
}
