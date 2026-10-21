package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

/**
 * Provider interface for model repositories.
 * @author bhangun
 */
public interface ModelRepositoryProvider {

    /**
     * @return the scheme supported by this provider (e.g., "hf", "local", "s3")
     */
    String scheme();

    /**
     * Create a new instance of the model repository.
     * 
     * @param context the repository context
     * @return the model repository instance
     */
    ModelRepository create(RepositoryContext context);
}
