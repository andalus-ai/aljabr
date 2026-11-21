package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Yogi adaptive optimizer. Its second-moment accumulator uses a signed
 * correction so that the accumulator does not grow without bound as readily
 * as the conventional Adam second moment.
 *
 * <p>{@code m = beta1*mPrevious + (1-beta1)*gradient} and
 * {@code v = vPrevious - (1-beta2)*gradient^2*sign(vPrevious-gradient^2)}.
 * Bias-corrected moments are used in the parameter update.</p>
 *
 * <p>Replacing a parameter tensor restarts its moments and step counter.
 * Missing gradients and frozen parameters are skipped.</p>
 */
public final class Yogi implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final OptimizerState state = new OptimizerState();

    /** Uses beta1=0.9, beta2=0.999, and epsilon=1e-3. */
    public Yogi(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1.0e-3f);
    }

    public Yogi(float learningRate, float beta1, float beta2, float epsilon) {
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
            Map<String, Tensor> nextSecond = new LinkedHashMap<>();
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
                    long step = replaced ? 1L : state.parameterSteps.getOrDefault(name, 0L) + 1L;
                    Tensor previousFirst = replaced ? null : state.firstMoment.get(name);
                    Tensor previousSecond = replaced ? null : state.secondMoment.get(name);

                    Tensor first = updateFirstMoment(gradient, previousFirst);
                    nextFirst.put(name, first);
                    Tensor second = updateYogiSecondMoment(gradient, previousSecond);
                    nextSecond.put(name, second);

                    Tensor firstHat = null;
                    Tensor secondHat = null;
                    Tensor root = null;
                    Tensor denominator = null;
                    Tensor adaptive = null;
                    Tensor scaled = null;
                    try {
                        float correction1 = 1.0f - (float) Math.pow(beta1, step);
                        float correction2 = 1.0f - (float) Math.pow(beta2, step);
                        firstHat = first.div(correction1);
                        secondHat = second.div(correction2);
                        root = secondHat.sqrt();
                        denominator = root.add(epsilon);
                        adaptive = firstHat.div(denominator);
                        scaled = adaptive.mul(learningRate);
                        updates.put(name, current.sub(scaled));
                    } finally {
                        if (firstHat != null) firstHat.release();
                        if (secondHat != null) secondHat.release();
                        if (root != null) root.release();
                        if (denominator != null) denominator.release();
                        if (adaptive != null) adaptive.release();
                        if (scaled != null) scaled.release();
                    }
                    nextSteps.put(name, step);
                }

                // Publish parameter values before committing optimizer state.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextFirst.forEach((name, value) -> {
                    Tensor old = state.firstMoment.put(name, value);
                    if (old != null) old.release();
                });
                nextSecond.forEach((name, value) -> {
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
                nextSecond.forEach((name, value) -> {
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

    private Tensor updateYogiSecondMoment(Tensor gradient, Tensor previous) {
        Tensor squaredGradient = null;
        Tensor base = null;
        Tensor difference = null;
        Tensor direction = null;
        Tensor signedSquared = null;
        Tensor correction = null;
        try {
            squaredGradient = gradient.mul(gradient);
            base = previous == null ? squaredGradient.zerosLike() : previous;
            difference = base.sub(squaredGradient);
            direction = difference.sign();
            signedSquared = squaredGradient.mul(direction);
            correction = signedSquared.mul(1.0f - beta2);
            return base.sub(correction);
        } finally {
            if (squaredGradient != null) squaredGradient.release();
            if (previous == null && base != null) base.release();
            if (difference != null) difference.release();
            if (direction != null) direction.release();
            if (signedSquared != null) signedSquared.release();
            if (correction != null) correction.release();
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
