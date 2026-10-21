package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class PositionalEncoding extends NNModule {
    public PositionalEncoding(int dModel, int maxLen) {}
    
    @Override
    public Tensor forward(Tensor input) {
        return input; // placeholder
    }
}
