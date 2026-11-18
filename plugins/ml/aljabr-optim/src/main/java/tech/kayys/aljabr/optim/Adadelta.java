package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Adadelta with exponentially decaying averages of squared gradients and updates.
 *
 * <p>For each parameter, this implementation computes
 * {@code E[g^2] = rho * E[g^2] + (1-rho) * g^2},
 * {@code delta = -sqrt(E[delta^2] + epsilon) / sqrt(E[g^2] + epsilon) * g},
 * and {@code E[delta^2] = rho * E[delta^2] + (1-rho) * delta^2}.
 * The parameter is updated by adding {@code delta}; there is no global
 * learning-rate multiplier in this standard form.</p>
 */
public final class Adadelta implements Optimizer, AutoCloseable {
    private final float rho;
    private final float epsilon;
    private final Map<String, Tensor> gradientSquareAverages = new LinkedHashMap<>();
    private final Map<String, Tensor> updateSquareAverages = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();

    public Adadelta(float rho, float epsilon) {
        if (!Float.isFinite(rho) || rho < 0.0f || rho >= 1.0f) {
            throw new IllegalArgumentException("rho must be in [0, 1)");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.rho = rho;
        this.epsilon = epsilon;
    }

    public float rho() { return rho; }
    public float epsilon() { return epsilon; }

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
        if (!hasGradient) return;

        try (AutogradContext.Scope ignored = AutogradContext.pauseRecording()) {
            Map<String, Tensor> nextGradientSquares = new LinkedHashMap<>();
            Map<String, Tensor> nextUpdateSquares = new LinkedHashMap<>();
            Map<String, Tensor> updates = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                    String name = entry.getKey();
                    Parameter parameter = entry.getValue();
                    Tensor gradient = gradients.get(name);
                    if (gradient == null || !parameter.isTrainable()) continue;

                    Tensor current = parameter.tensor();
                    boolean replaced = parameterSnapshots.containsKey(name)
                            && parameterSnapshots.get(name) != current;
                    Tensor previousGradientSquares = replaced ? null : gradientSquareAverages.get(name);
                    Tensor previousUpdateSquares = replaced ? null : updateSquareAverages.get(name);

                    Tensor gradientSquares = updateAverage(gradient, previousGradientSquares);
                    nextGradientSquares.put(name, gradientSquares);

                    Tensor gradientPlusEpsilon = null;
                    Tensor gradientRoot = null;
                    Tensor updatePlusEpsilon = null;
                    Tensor updateRoot = null;
                    Tensor ratio = null;
                    Tensor scaledGradient = null;
                    Tensor delta = null;
                    Tensor deltaSquares = null;
                    Tensor updateContribution = null;
                    Tensor decayedUpdateSquares = null;
                    Tensor nextUpdateAverage = null;
                    try {
                        gradientPlusEpsilon = gradientSquares.add(epsilon);
                        gradientRoot = gradientPlusEpsilon.sqrt();
                        Tensor previousUpdate = previousUpdateSquares == null
                                ? gradient.zerosLike() : previousUpdateSquares;
                        try {
                            updatePlusEpsilon = previousUpdate.add(epsilon);
                        } finally {
                            if (previousUpdateSquares == null) previousUpdate.release();
                        }
                        updateRoot = updatePlusEpsilon.sqrt();
                        ratio = updateRoot.div(gradientRoot);
                        scaledGradient = gradient.mul(ratio);
                        delta = scaledGradient.mul(-1.0f);

                        deltaSquares = delta.mul(delta);
                        updateContribution = deltaSquares.mul(1.0f - rho);
                        decayedUpdateSquares = previousUpdateSquares == null
                                ? deltaSquares.zerosLike() : previousUpdateSquares.mul(rho);
                        nextUpdateAverage = decayedUpdateSquares.add(updateContribution);
                        nextUpdateSquares.put(name, nextUpdateAverage);
                        updates.put(name, current.add(delta));
                    } finally {
                        if (gradientPlusEpsilon != null) gradientPlusEpsilon.release();
                        if (gradientRoot != null) gradientRoot.release();
                        if (updatePlusEpsilon != null) updatePlusEpsilon.release();
                        if (updateRoot != null) updateRoot.release();
                        if (ratio != null) ratio.release();
                        if (scaledGradient != null) scaledGradient.release();
                        if (delta != null) delta.release();
                        if (deltaSquares != null) deltaSquares.release();
                        if (updateContribution != null) updateContribution.release();
                        if (decayedUpdateSquares != null) decayedUpdateSquares.release();
                    }
                }

                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));
                nextGradientSquares.forEach((name, value) -> {
                    Tensor old = gradientSquareAverages.put(name, value);
                    if (old != null) old.release();
                });
                nextUpdateSquares.forEach((name, value) -> {
                    Tensor old = updateSquareAverages.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> parameterSnapshots.put(name, value));
            } catch (RuntimeException | Error failure) {
                nextGradientSquares.forEach((name, value) -> {
                    if (gradientSquareAverages.get(name) != value) value.release();
                });
                nextUpdateSquares.forEach((name, value) -> {
                    if (updateSquareAverages.get(name) != value) value.release();
                });
                updates.forEach((name, value) -> {
                    if (parameters.get(name).tensor() != value) value.release();
                });
                throw failure;
            }
        }
    }

    private Tensor updateAverage(Tensor gradient, Tensor previous) {
        Tensor base = previous == null ? gradient.zerosLike() : previous.mul(rho);
        Tensor squared = null;
        Tensor contribution = null;
        try {
            squared = gradient.mul(gradient);
            contribution = squared.mul(1.0f - rho);
            return base.add(contribution);
        } finally {
            base.release();
            if (squared != null) squared.release();
            if (contribution != null) contribution.release();
        }
    }

    @Override
    public void reset() {
        gradientSquareAverages.values().forEach(Tensor::release);
        updateSquareAverages.values().forEach(Tensor::release);
        gradientSquareAverages.clear();
        updateSquareAverages.clear();
        parameterSnapshots.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
