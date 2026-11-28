package tech.kayys.aljabr.models.diffusion;

import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Owned, architecture-ready inputs for one UNet2DConditional invocation.
 *
 * <p>The mapper transfers ownership of both tensors to this value. The
 * timestep embedding is required; encoder hidden states may be absent for
 * unconditional architectures. Closing this value closes each distinct
 * tensor exactly once.</p>
 */
public final class UNet2DModelInputs implements AutoCloseable {
    private final Tensor timestepEmbedding;
    private final Tensor encoderHiddenStates;
    private boolean closed;

    public UNet2DModelInputs(Tensor timestepEmbedding, Tensor encoderHiddenStates) {
        this.timestepEmbedding = Objects.requireNonNull(
                timestepEmbedding, "timestepEmbedding");
        if (timestepEmbedding == encoderHiddenStates) {
            throw new IllegalArgumentException(
                    "timestep embedding and encoder hidden states must be distinct tensors");
        }
        this.encoderHiddenStates = encoderHiddenStates;
    }

    public Tensor timestepEmbedding() {
        ensureOpen();
        return timestepEmbedding;
    }

    public Tensor encoderHiddenStates() {
        ensureOpen();
        return encoderHiddenStates;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            timestepEmbedding.close();
        } finally {
            if (encoderHiddenStates != null) {
                encoderHiddenStates.close();
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("UNet model inputs are closed");
        }
    }
}
