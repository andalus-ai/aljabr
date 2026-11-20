package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Adamax: Adam's infinity-norm variant with bias-corrected first moments.
 *
 * <p>The first moment is an exponential average of gradients. The infinity
 * norm is updated element-wise as {@code u = max(beta2 * u, abs(gradient))}.
 * The parameter update is {@code p -= learningRate * (m / (1-beta1^t)) /
 * (u + epsilon)}.</p>
 *
 * <p>Replacing a parameter tensor restarts its moments and step counter.
 * Missing gradients and frozen parameters are skipped.</p>
 */
public final class Adamax implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final OptimizerState state = new OptimizerState();

    /** Uses beta1=0.9, beta2=0.999, and epsilon=1e-8. */
    public Adamax(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1.0e-8f);
    }

    public Adamax(float learningRate, float beta1, float beta2, float epsilon) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f) {
            throw new IllegalArgumentException("beta1 must be in [0, 1)");
        }
        if (!Float.isFinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f) {
            throw new IllegalArgumentException("beta2 must be in [0, 1)");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public float learningRate() { return learningRate; }
    public float beta1() { return beta1; }
    public float beta2() { return beta2; }
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
            Map<String, Tensor> nextFirst = new LinkedHashMap<>();
            Map<String, Tensor> nextInfinity = new LinkedHashMap<>();
            Map<String, Tensor> updates = new LinkedHashMap<>();
            Map<String, Long> nextSteps = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                    String name = entry.getKey();
                    Parameter parameter = entry.getValue();
                    Tensor gradient = gradients.get(name);
                    if (gradient == null || !parameter.isTrainable()) continue;

                    Tensor current = parameter.tensor();
                    boolean replaced = state.parameterSnapshots.containsKey(name)
                            && state.parameterSnapshots.get(name) != current;
                    long nextStep = replaced ? 1L : state.parameterSteps.getOrDefault(name, 0L) + 1L;
                    float correction1 = 1.0f - (float) Math.pow(beta1, nextStep);

                    Tensor first = updateFirstMoment(
                            gradient, replaced ? null : state.firstMoment.get(name));
                    nextFirst.put(name, first);
                    Tensor infinity = updateInfinityNorm(
                            gradient, replaced ? null : state.secondMoment.get(name));
                    nextInfinity.put(name, infinity);

                    Tensor correctedFirst = null;
                    Tensor denominator = null;
                    Tensor adaptive = null;
                    Tensor scaled = null;
                    try {
                        correctedFirst = first.div(correction1);
                        denominator = infinity.add(epsilon);
                        adaptive = correctedFirst.div(denominator);
                        scaled = adaptive.mul(learningRate);
                        updates.put(name, current.sub(scaled));
                    } finally {
                        if (correctedFirst != null) correctedFirst.release();
                        if (denominator != null) denominator.release();
                        if (adaptive != null) adaptive.release();
                        if (scaled != null) scaled.release();
                    }
                    nextSteps.put(name, nextStep);
                }

                // Publish parameter values before committing optimizer state.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextFirst.forEach((name, value) -> {
                    Tensor old = state.firstMoment.put(name, value);
                    if (old != null) old.release();
                });
                nextInfinity.forEach((name, value) -> {
                    Tensor old = state.secondMoment.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> {
                    state.parameterSnapshots.put(name, value);
                    state.parameterSteps.put(name, nextSteps.get(name));
                });
            } catch (RuntimeException | Error failure) {
                nextFirst.forEach((name, value) -> {
                    if (state.firstMoment.get(name) != value) value.release();
                });
                nextInfinity.forEach((name, value) -> {
                    if (state.secondMoment.get(name) != value) value.release();
                });
                updates.forEach((name, value) -> {
                    if (parameters.get(name).tensor() != value) value.release();
                });
                throw failure;
            }
        }
    }

    private Tensor updateFirstMoment(Tensor gradient, Tensor previous) {
        Tensor base = previous == null ? gradient.zerosLike() : previous.mul(beta1);
        Tensor contribution = null;
        try {
            contribution = gradient.mul(1.0f - beta1);
            return base.add(contribution);
        } finally {
            base.release();
            if (contribution != null) contribution.release();
        }
    }

    private Tensor updateInfinityNorm(Tensor gradient, Tensor previous) {
        Tensor absoluteGradient = null;
        Tensor decayedPrevious = null;
        Tensor result = null;
        try {
            absoluteGradient = gradient.abs();
            if (previous == null) {
                result = absoluteGradient;
                absoluteGradient = null;
            } else {
                decayedPrevious = previous.mul(beta2);
                result = elementwiseMaximum(decayedPrevious, absoluteGradient);
            }
            return result;
        } catch (RuntimeException | Error failure) {
            if (result != null) result.release();
            throw failure;
        } finally {
            if (absoluteGradient != null) absoluteGradient.release();
            if (decayedPrevious != null) decayedPrevious.release();
        }
    }

    /** Computes element-wise max(a,b) using existing tensor arithmetic. */
    private static Tensor elementwiseMaximum(Tensor a, Tensor b) {
        Tensor halfA = null;
        Tensor halfB = null;
        Tensor difference = null;
        Tensor absoluteDifference = null;
        Tensor halfDifference = null;
        Tensor partial = null;
        Tensor result = null;
        try {
            // The inputs are non-negative norms. Halving before addition avoids
            // overflow when both values are close to Float.MAX_VALUE.
            halfA = a.mul(0.5f);
            halfB = b.mul(0.5f);
            difference = a.sub(b);
            absoluteDifference = difference.abs();
            halfDifference = absoluteDifference.mul(0.5f);
            partial = halfA.add(halfB);
            result = partial.add(halfDifference);
            return result;
        } catch (RuntimeException | Error failure) {
            if (result != null) result.release();
            throw failure;
        } finally {
            if (halfA != null) halfA.release();
            if (halfB != null) halfB.release();
            if (difference != null) difference.release();
            if (absoluteDifference != null) absoluteDifference.release();
            if (halfDifference != null) halfDifference.release();
            if (partial != null) partial.release();
        }
    }

    @Override
    public void reset() {
        state.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
