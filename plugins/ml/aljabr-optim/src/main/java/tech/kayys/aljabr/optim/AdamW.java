package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/** Adam with decoupled weight decay. */
public final class AdamW implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final OptimizerState state = new OptimizerState();

    public AdamW(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
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
        if (!Float.isFinite(weightDecay) || weightDecay < 0.0f) {
            throw new IllegalArgumentException("weight decay must be finite and non-negative");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
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
            Map<String, Tensor> nextFirst = new LinkedHashMap<>();
            Map<String, Tensor> nextSecond = new LinkedHashMap<>();
            Map<String, Tensor> updates = new LinkedHashMap<>();
            Map<String, Long> nextSteps = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                    String name = entry.getKey();
                    Parameter parameter = entry.getValue();
                    Tensor gradient = gradients.get(name);
                    if (gradient == null || !parameter.isTrainable()) {
                        continue;
                    }

                    Tensor current = parameter.tensor();
                    boolean parameterReplaced = state.parameterSnapshots.containsKey(name)
                            && state.parameterSnapshots.get(name) != current;
                    long nextStep = parameterReplaced ? 1L : state.parameterSteps.getOrDefault(name, 0L) + 1L;
                    float correction1 = 1.0f - (float) Math.pow(beta1, nextStep);
                    float correction2 = 1.0f - (float) Math.pow(beta2, nextStep);
                    Tensor previousFirst = parameterReplaced ? null : state.firstMoment.get(name);
                    Tensor previousSecond = parameterReplaced ? null : state.secondMoment.get(name);
                    Tensor first = updateFirstMoment(gradient, previousFirst);
                    Tensor second = updateSecondMoment(gradient, previousSecond);
                    Tensor update = computeUpdate(current, first, second, correction1, correction2);
                    nextFirst.put(name, first);
                    nextSecond.put(name, second);
                    updates.put(name, update);
                    nextSteps.put(name, nextStep);
                }

                // Commit state only after every parameter update has been computed.
                nextFirst.forEach((name, value) -> {
                    Tensor old = state.firstMoment.put(name, value);
                    if (old != null) old.release();
                });
                nextSecond.forEach((name, value) -> {
                    Tensor old = state.secondMoment.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> {
                    parameters.get(name).updateTensor(value);
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
            Tensor current, Tensor first, Tensor second, float correction1, float correction2) {
        Tensor firstHat = null;
        Tensor secondHat = null;
        Tensor root = null;
        Tensor denominator = null;
        Tensor adaptive = null;
        Tensor decay = null;
        Tensor decayed = null;
        Tensor scaledAdaptive = null;
        try {
            firstHat = first.div(correction1);
            secondHat = second.div(correction2);
            root = secondHat.sqrt();
            denominator = root.add(epsilon);
            adaptive = firstHat.div(denominator);
            decay = current.mul(learningRate * weightDecay);
            decayed = current.sub(decay);
            scaledAdaptive = adaptive.mul(learningRate);
            return decayed.sub(scaledAdaptive);
        } finally {
            if (firstHat != null) firstHat.release();
            if (secondHat != null) secondHat.release();
            if (root != null) root.release();
            if (denominator != null) denominator.release();
            if (adaptive != null) adaptive.release();
            if (decay != null) decay.release();
            if (decayed != null) decayed.release();
            if (scaledAdaptive != null) scaledAdaptive.release();
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
