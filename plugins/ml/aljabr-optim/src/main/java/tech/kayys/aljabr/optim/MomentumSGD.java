package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Stochastic gradient descent with a per-parameter momentum buffer.
 *
 * <p>The velocity follows {@code v = momentum * previousVelocity + gradient};
 * parameters are updated as {@code parameter -= learningRate * v}. A replaced
 * parameter tensor starts with a fresh velocity buffer.</p>
 */
public final class MomentumSGD implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float momentum;
    private final boolean nesterov;
    private final Map<String, Tensor> velocities = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();

    public MomentumSGD(float learningRate, float momentum) {
        this(learningRate, momentum, false);
    }

    public MomentumSGD(float learningRate, float momentum, boolean nesterov) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(momentum) || momentum < 0.0f || momentum >= 1.0f) {
            throw new IllegalArgumentException("Momentum must be in [0, 1)");
        }
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.nesterov = nesterov;
    }

    public float learningRate() {
        return learningRate;
    }

    public float momentum() {
        return momentum;
    }

    public boolean nesterov() {
        return nesterov;
    }

    @Override
    public void step(Map<String, Parameter> parameters, Map<String, Tensor> gradients) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        Objects.requireNonNull(gradients, "gradients must not be null");

        boolean hasGradient = false;
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = Objects.requireNonNull(entry.getKey(), "parameter name must not be null");
            Parameter parameter = Objects.requireNonNull(entry.getValue(), "parameter " + name);
            Tensor gradient = gradients.get(name);
            if (gradient != null && parameter.isTrainable()) {
                parameter.validateGradient(gradient);
                hasGradient = true;
            }
        }
        if (!hasGradient) {
            return;
        }

        try (AutogradContext.Scope ignored = AutogradContext.pauseRecording()) {
            Map<String, Tensor> nextVelocities = new LinkedHashMap<>();
            Map<String, Tensor> updates = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                    String name = entry.getKey();
                    Parameter parameter = entry.getValue();
                    Tensor gradient = gradients.get(name);
                    if (gradient == null || !parameter.isTrainable()) {
                        continue;
                    }

                    Tensor current = parameter.tensor();
                    boolean replaced = parameterSnapshots.containsKey(name)
                            && parameterSnapshots.get(name) != current;
                    Tensor previous = replaced ? null : velocities.get(name);
                    Tensor base = previous == null ? gradient.zerosLike() : previous.mul(momentum);
                    Tensor velocity = null;
                    Tensor lookahead = null;
                    Tensor direction = null;
                    Tensor scaledDirection = null;
                    try {
                        velocity = base.add(gradient);
                        nextVelocities.put(name, velocity);
                        if (nesterov) {
                            lookahead = velocity.mul(momentum);
                            direction = gradient.add(lookahead);
                        } else {
                            direction = velocity;
                        }
                        scaledDirection = direction.mul(learningRate);
                        updates.put(name, current.sub(scaledDirection));
                    } finally {
                        base.release();
                        if (lookahead != null) lookahead.release();
                        if (nesterov && direction != null) direction.release();
                        if (scaledDirection != null) scaledDirection.release();
                    }
                }

                // Publish parameter values before replacing optimizer state. If a
                // parameter publication fails, newly computed velocities remain
                // uncommitted and are released below; the next step detects any
                // parameter values that were already published and resets their
                // momentum rather than pairing them with stale state.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextVelocities.forEach((name, value) -> {
                    Tensor old = velocities.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> parameterSnapshots.put(name, value));
            } catch (RuntimeException | Error failure) {
                nextVelocities.forEach((name, value) -> {
                    if (velocities.get(name) != value) value.release();
                });
                updates.forEach((name, value) -> {
                    if (parameters.get(name).tensor() != value) value.release();
                });
                throw failure;
            }
        }
    }

    @Override
    public void reset() {
        velocities.values().forEach(Tensor::release);
        velocities.clear();
        parameterSnapshots.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
