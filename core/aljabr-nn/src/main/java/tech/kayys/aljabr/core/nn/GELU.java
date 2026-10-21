package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * GELU activation layer.
 * @author bhangun
 */
public class GELU extends NNModule {

    @Override
    public Tensor forward(Tensor input) {
        return input.gelu();
    }

    @Override
    public String toString() {
        return "GELU()";
    }
}
