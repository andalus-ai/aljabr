package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * RMSProp with an exponentially decaying average of squared gradients.
 *
 * <p>For each parameter, the square average is updated as
 * {@code s = alpha * previousS + (1 - alpha) * gradient^2}; the parameter
 * update is {@code parameter -= learningRate * gradient / (sqrt(s) + epsilon)}.
 * Replacing a parameter tensor starts its square average from zero.</p>
 *
 * <p>This implementation intentionally omits centered RMSProp, momentum,
 * and weight decay. Those options can be added independently without changing
 * the basic optimizer contract.</p>
 */
public final class RMSProp implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float alpha;
    private final float epsilon;
    private final Map<String, Tensor> squareAverages = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();

    public RMSProp(float learningRate, float alpha, float epsilon) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(alpha) || alpha < 0.0f || alpha >= 1.0f) {
            throw new IllegalArgumentException("alpha must be in [0, 1)");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.learningRate = learningRate;
        this.alpha = alpha;
        this.epsilon = epsilon;
    }

    public float learningRate() {
        return learningRate;
    }

    public float alpha() {
        return alpha;
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
            Map<String, Tensor> nextSquareAverages = new LinkedHashMap<>();
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
                    Tensor previous = replaced ? null : squareAverages.get(name);
                    Tensor base = previous == null
                            ? gradient.zerosLike()
                            : previous.mul(alpha);
                    Tensor squared = null;
                    Tensor contribution = null;
                    Tensor squareAverage = null;
                    Tensor root = null;
                    Tensor denominator = null;
                    Tensor normalized = null;
                    Tensor scaled = null;
                    try {
                        squared = gradient.mul(gradient);
                        contribution = squared.mul(1.0f - alpha);
                        squareAverage = base.add(contribution);
                        nextSquareAverages.put(name, squareAverage);

                        root = squareAverage.sqrt();
                        denominator = root.add(epsilon);
                        normalized = gradient.div(denominator);
                        scaled = normalized.mul(learningRate);
                        updates.put(name, current.sub(scaled));
                    } finally {
                        base.release();
                        if (squared != null) squared.release();
                        if (contribution != null) contribution.release();
                        if (root != null) root.release();
                        if (denominator != null) denominator.release();
                        if (normalized != null) normalized.release();
                        if (scaled != null) scaled.release();
                    }
                }

                // Publish parameter values before committing the new optimizer
                // state. Snapshot identity lets the next step detect any values
                // published before a later parameter publication failed.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextSquareAverages.forEach((name, value) -> {
                    Tensor old = squareAverages.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> parameterSnapshots.put(name, value));
            } catch (RuntimeException | Error failure) {
                nextSquareAverages.forEach((name, value) -> {
                    if (squareAverages.get(name) != value) value.release();
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
        squareAverages.values().forEach(Tensor::release);
        squareAverages.clear();
        parameterSnapshots.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
