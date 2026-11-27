package tech.kayys.aljabr.diffusion.execution.loop;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Pair of newly owned predictions produced by one batched classifier-free
 * guidance evaluation. The guidance engine closes both tensors after use.
 */
public record GuidancePredictions(Tensor unconditional, Tensor conditional) implements AutoCloseable {

    public GuidancePredictions {
        Objects.requireNonNull(unconditional, "unconditional");
        Objects.requireNonNull(conditional, "conditional");
        if (unconditional == conditional) {
            throw new IllegalArgumentException("Guidance predictions must be distinct tensors");
        }
    }

    @Override
    public void close() {
        try {
            unconditional.close();
        } finally {
            conditional.close();
        }
    }
}
