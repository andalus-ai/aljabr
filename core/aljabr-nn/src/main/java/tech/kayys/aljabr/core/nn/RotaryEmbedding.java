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
public class RotaryEmbedding extends NNModule {
    public RotaryEmbedding(int headDim, int maxSeqLen) {}
    
    @Override
    public Tensor forward(Tensor input) {
        return input; // placeholder
    }
    
    public Tensor apply(Tensor q, Tensor k) {
        return q; // placeholder
    }
}
