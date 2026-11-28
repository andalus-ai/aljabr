package tech.kayys.aljabr.models;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * LIFO stack of borrowed encoder feature maps used by UNet decoder stages.
 *
 * <p>This container never retains or releases tensor storage. The caller owns
 * every pushed tensor and must keep it alive until it is popped and consumed.
 * A stack is scoped to one forward pass and must not be shared across threads.</p>
 */
public final class UNetSkipConnections {

    private final Deque<Tensor> values = new ArrayDeque<>();

    public void push(Tensor tensor) {
        values.push(Objects.requireNonNull(tensor, "skip tensor must not be null"));
    }

    public Tensor pop() {
        if (values.isEmpty()) {
            throw new IllegalStateException("No encoder skip connection is available");
        }
        return values.pop();
    }

    public int size() {
        return values.size();
    }

    public boolean isEmpty() {
        return values.isEmpty();
    }

    public void requireEmpty() {
        if (!values.isEmpty()) {
            throw new IllegalStateException(
                    "UNet forward pass left " + values.size() + " unconsumed skip connection(s)");
        }
    }
}
