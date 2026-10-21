package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Softmax activation layer.
 * @author bhangun
 */
public class Softmax extends NNModule {

    private final int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    public Softmax() {
        this(-1);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(dim);
    }

    public int dim() {
        return dim;
    }

    @Override
    public String toString() {
        return "Softmax(dim=" + dim + ")";
    }
}
