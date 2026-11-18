package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Nesterov-accelerated adaptive moment estimation (Nadam), using the Dozat
 * formulation of the lookahead first-moment estimate.
 *
 * <p>The first and second moments are bias-corrected independently. The
 * lookahead direction is
 * {@code beta1 * mHat + (1 - beta1) * gradient / (1 - beta1^step)}.</p>
 */
public final class Nadam implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final Map<String, Tensor> firstMoments = new LinkedHashMap<>();
    private final Map<String, Tensor> secondMoments = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();
    private final Map<String, Long> parameterSteps = new LinkedHashMap<>();

    public Nadam(float learningRate, float beta1, float beta2, float epsilon) {
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
                    boolean replaced = parameterSnapshots.containsKey(name)
                            && parameterSnapshots.get(name) != current;
                    long step = replaced ? 1L : parameterSteps.getOrDefault(name, 0L) + 1L;
                    float correction1 = 1.0f - (float) Math.pow(beta1, step);
                    float correction2 = 1.0f - (float) Math.pow(beta2, step);
                    Tensor previousFirst = replaced ? null : firstMoments.get(name);
                    Tensor previousSecond = replaced ? null : secondMoments.get(name);

                    Tensor first = updateFirstMoment(gradient, previousFirst);
                    nextFirst.put(name, first);
                    Tensor second = updateSecondMoment(gradient, previousSecond);
                    nextSecond.put(name, second);
                    updates.put(name, computeUpdate(current, gradient, first, second, correction1, correction2));
                    nextSteps.put(name, step);
                }

                // Keep optimizer state unchanged until every parameter publication succeeds.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextFirst.forEach((name, value) -> {
                    Tensor old = firstMoments.put(name, value);
                    if (old != null) old.release();
                });
                nextSecond.forEach((name, value) -> {
                    Tensor old = secondMoments.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> {
                    parameterSnapshots.put(name, value);
                    parameterSteps.put(name, nextSteps.get(name));
                });
            } catch (RuntimeException | Error failure) {
                nextFirst.forEach((name, value) -> {
                    if (firstMoments.get(name) != value) value.release();
                });
                nextSecond.forEach((name, value) -> {
                    if (secondMoments.get(name) != value) value.release();
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

    private Tensor updateSecondMoment(Tensor gradient, Tensor previous) {
        Tensor base = previous == null ? gradient.zerosLike() : previous.mul(beta2);
        Tensor squared = null;
        Tensor contribution = null;
        try {
            squared = gradient.mul(gradient);
            contribution = squared.mul(1.0f - beta2);
            return base.add(contribution);
        } finally {
            base.release();
            if (squared != null) squared.release();
            if (contribution != null) contribution.release();
        }
    }

    private Tensor computeUpdate(
            Tensor current, Tensor gradient, Tensor first, Tensor second,
            float correction1, float correction2) {
        Tensor firstHat = null;
        Tensor lookaheadMomentum = null;
        Tensor lookaheadGradient = null;
        Tensor direction = null;
        Tensor secondHat = null;
        Tensor root = null;
        Tensor denominator = null;
        Tensor adaptive = null;
        Tensor scaled = null;
        try {
            firstHat = first.div(correction1);
            lookaheadMomentum = firstHat.mul(beta1);
            lookaheadGradient = gradient.mul((1.0f - beta1) / correction1);
            direction = lookaheadMomentum.add(lookaheadGradient);
            secondHat = second.div(correction2);
            root = secondHat.sqrt();
            denominator = root.add(epsilon);
            adaptive = direction.div(denominator);
            scaled = adaptive.mul(learningRate);
            return current.sub(scaled);
        } finally {
            if (firstHat != null) firstHat.release();
            if (lookaheadMomentum != null) lookaheadMomentum.release();
            if (lookaheadGradient != null) lookaheadGradient.release();
            if (direction != null) direction.release();
            if (secondHat != null) secondHat.release();
            if (root != null) root.release();
            if (denominator != null) denominator.release();
            if (adaptive != null) adaptive.release();
            if (scaled != null) scaled.release();
        }
    }

    @Override
    public void reset() {
        firstMoments.values().forEach(Tensor::release);
        secondMoments.values().forEach(Tensor::release);
        firstMoments.clear();
        secondMoments.clear();
        parameterSnapshots.clear();
        parameterSteps.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
