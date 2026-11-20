package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Lion (Evolved Sign Momentum) with decoupled weight decay.
 *
 * <p>The momentum is updated as {@code m = beta2 * m + (1 - beta2) * gradient}.
 * The update direction is {@code sign(beta1 * m + (1 - beta1) * gradient)}.
 * Parameters are decayed and then moved by the signed direction:
 * {@code p = p * (1 - learningRate * weightDecay) - learningRate * direction}.</p>
 *
 * <p>Replacing a parameter tensor resets its momentum. Missing gradients and
 * frozen parameters are skipped.</p>
 */
public final class Lion implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final Map<String, Tensor> momentum = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();

    /** Uses the commonly published Lion beta defaults and no weight decay. */
    public Lion(float learningRate) {
        this(learningRate, 0.9f, 0.99f, 0.0f);
    }

    public Lion(float learningRate, float beta1, float beta2) {
        this(learningRate, beta1, beta2, 0.0f);
    }

    public Lion(float learningRate, float beta1, float beta2, float weightDecay) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f) {
            throw new IllegalArgumentException("beta1 must be in [0, 1)");
        }
        if (!Float.isFinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f) {
            throw new IllegalArgumentException("beta2 must be in [0, 1)");
        }
        if (!Float.isFinite(weightDecay) || weightDecay < 0.0f) {
            throw new IllegalArgumentException("weight decay must be finite and non-negative");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weightDecay = weightDecay;
    }

    public float learningRate() {
        return learningRate;
    }

    public float beta1() {
        return beta1;
    }

    public float beta2() {
        return beta2;
    }

    public float weightDecay() {
        return weightDecay;
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
            Map<String, Tensor> nextMomentum = new LinkedHashMap<>();
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
                    Tensor previous = replaced ? null : momentum.get(name);
                    Tensor base = previous == null ? gradient.zerosLike() : previous.mul(beta2);
                    Tensor contribution = null;
                    Tensor next = null;
                    Tensor weightedMomentum = null;
                    Tensor weightedGradient = null;
                    Tensor directionInput = null;
                    Tensor direction = null;
                    Tensor decay = null;
                    Tensor decayed = null;
                    Tensor scaledDirection = null;
                    try {
                        contribution = gradient.mul(1.0f - beta2);
                        next = base.add(contribution);
                        nextMomentum.put(name, next);

                        weightedMomentum = next.mul(beta1);
                        weightedGradient = gradient.mul(1.0f - beta1);
                        directionInput = weightedMomentum.add(weightedGradient);
                        direction = directionInput.sign();

                        decay = current.mul(learningRate * weightDecay);
                        decayed = current.sub(decay);
                        scaledDirection = direction.mul(learningRate);
                        updates.put(name, decayed.sub(scaledDirection));
                    } finally {
                        base.release();
                        if (contribution != null) contribution.release();
                        if (weightedMomentum != null) weightedMomentum.release();
                        if (weightedGradient != null) weightedGradient.release();
                        if (directionInput != null) directionInput.release();
                        if (direction != null) direction.release();
                        if (decay != null) decay.release();
                        if (decayed != null) decayed.release();
                        if (scaledDirection != null) scaledDirection.release();
                    }
                }

                // Publish parameter values before committing momentum. If a
                // publication fails partway through, snapshot identity detects
                // already-published values and resets their stale momentum next step.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextMomentum.forEach((name, value) -> {
                    Tensor old = momentum.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> parameterSnapshots.put(name, value));
            } catch (RuntimeException | Error failure) {
                nextMomentum.forEach((name, value) -> {
                    if (momentum.get(name) != value) value.release();
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
        momentum.values().forEach(Tensor::release);
        momentum.clear();
        parameterSnapshots.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
