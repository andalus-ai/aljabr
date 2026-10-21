package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Sigmoid activation layer.
 * @author bhangun
 */
public class Sigmoid extends NNModule {

    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }

    @Override
    public String toString() {
        return "Sigmoid()";
    }
}
