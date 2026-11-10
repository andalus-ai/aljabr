package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Flatten layer — reshapes input to [batch_size, -1].
 * @author bhangun
 */
public class Flatten extends NNModule {

    private final int startDim;
    private final int endDim;

    public Flatten(int startDim, int endDim) {
        this.startDim = startDim;
        this.endDim = endDim;
    }

    public Flatten() {
        this(1, -1);
    }

    @Override
    public Tensor forward(Tensor input) {
        long[] shape = input.shape().dims();
        int ndim = shape.length;

        int start = startDim >= 0 ? startDim : ndim + startDim;
        int end = endDim >= 0 ? endDim : ndim + endDim;

        long flatSize = 1;
        for (int d = start; d <= end; d++) {
            flatSize *= shape[d];
        }

        long[] newShape = new long[start + 1 + (ndim - end - 1)];
        for (int i = 0; i < start; i++) {
            newShape[i] = shape[i];
        }
        newShape[start] = flatSize;
        for (int i = end + 1; i < ndim; i++) {
            newShape[start + 1 + (i - end - 1)] = shape[i];
        }

        return input.reshape(newShape);
    }

    @Override
    public String toString() {
        return String.format("Flatten(start_dim=%d, end_dim=%d)", startDim, endDim);
    }
}
