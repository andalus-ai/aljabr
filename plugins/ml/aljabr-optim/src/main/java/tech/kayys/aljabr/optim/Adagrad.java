package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Adaptive gradient optimizer with a cumulative sum of squared gradients.
 *
 * <p>For each parameter, the accumulator is updated as
 * {@code sumSquares += gradient * gradient}; the parameter update is
 * {@code parameter -= learningRate * gradient / (sqrt(sumSquares) + epsilon)}.
 * Replacing a parameter tensor starts its accumulator from zero.</p>
 *
 * <p>This implementation intentionally omits learning-rate decay and
 * weight decay. The accumulator is unbounded and is reset explicitly through
 * {@link #reset()}.</p>
 */
public final class Adagrad implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float epsilon;
    private final Map<String, Tensor> accumulatedSquares = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();

    public Adagrad(float learningRate, float epsilon) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    public float learningRate() {
        return learningRate;
    }

    public float epsilon() {
        return epsilon;
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
            Map<String, Tensor> nextAccumulatedSquares = new LinkedHashMap<>();
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
                    Tensor previous = replaced ? null : accumulatedSquares.get(name);
                    Tensor squared = null;
                    Tensor accumulated = null;
                    Tensor root = null;
                    Tensor denominator = null;
                    Tensor normalized = null;
                    Tensor scaled = null;
                    try {
                        squared = gradient.mul(gradient);
                        accumulated = previous == null ? squared.copy() : previous.add(squared);
                        nextAccumulatedSquares.put(name, accumulated);

                        root = accumulated.sqrt();
                        denominator = root.add(epsilon);
                        normalized = gradient.div(denominator);
                        scaled = normalized.mul(learningRate);
                        updates.put(name, current.sub(scaled));
                    } finally {
                        if (squared != null) squared.release();
                        if (root != null) root.release();
                        if (denominator != null) denominator.release();
                        if (normalized != null) normalized.release();
                        if (scaled != null) scaled.release();
                    }
                }

                // Publish parameter values before committing optimizer state.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextAccumulatedSquares.forEach((name, value) -> {
                    Tensor old = accumulatedSquares.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> parameterSnapshots.put(name, value));
            } catch (RuntimeException | Error failure) {
                nextAccumulatedSquares.forEach((name, value) -> {
                    if (accumulatedSquares.get(name) != value) value.release();
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
        accumulatedSquares.values().forEach(Tensor::release);
        accumulatedSquares.clear();
        parameterSnapshots.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
