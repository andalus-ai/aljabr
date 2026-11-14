package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;

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

        boolean parameterWasReplaced = parameters.entrySet().stream().anyMatch(entry -> {
            String name = entry.getKey();
            Parameter parameter = entry.getValue();
            return gradients.get(name) != null && parameter.isTrainable()
                    && state.parameterSnapshots.containsKey(name)
                    && state.parameterSnapshots.get(name) != parameter.tensor();
        });
        if (parameterWasReplaced) {
            // The bias-correction step and all moment histories must share one epoch.
            state.clear();
        }

        long nextStep = state.step + 1;
        Map<String, Tensor> nextFirst = new LinkedHashMap<>();
        Map<String, Tensor> nextSecond = new LinkedHashMap<>();
        Map<String, Tensor> nextSnapshots = new LinkedHashMap<>();
        Map<String, Tensor> updates = new LinkedHashMap<>();
        float correction1 = 1.0f - (float) Math.pow(beta1, nextStep);
        float correction2 = 1.0f - (float) Math.pow(beta2, nextStep);

        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = entry.getKey();
            Parameter parameter = entry.getValue();
            Tensor gradient = gradients.get(name);
            if (gradient == null || !parameter.isTrainable()) {
                continue;
            }

            Tensor current = parameter.tensor();
            Tensor oldFirst = state.firstMoment.get(name);
            Tensor oldSecond = state.secondMoment.get(name);
            Tensor first = (oldFirst == null ? gradient.zerosLike() : oldFirst.mul(beta1))
                    .add(gradient.mul(1.0f - beta1));
            Tensor second = (oldSecond == null ? gradient.zerosLike() : oldSecond.mul(beta2))
                    .add(gradient.mul(gradient).mul(1.0f - beta2));
            Tensor firstHat = first.div(correction1);
            Tensor secondHat = second.div(correction2);
            Tensor adaptive = firstHat.div(secondHat.sqrt().add(epsilon));
            Tensor decayed = current.sub(current.mul(learningRate * weightDecay));

            nextFirst.put(name, first);
            nextSecond.put(name, second);
            nextSnapshots.put(name, current);
            updates.put(name, decayed.sub(adaptive.mul(learningRate)));
        }

        // Commit the newly computed state only after every parameter update was computed.
        nextFirst.forEach((name, value) -> {
            Tensor old = state.firstMoment.put(name, value);
            if (old != null) old.release();
        });
        nextSecond.forEach((name, value) -> {
            Tensor old = state.secondMoment.put(name, value);
            if (old != null) old.release();
        });
        nextSnapshots.forEach((name, value) -> {
            if (state.parameterSnapshots.get(name) != value) {
                state.parameterSnapshots.put(name, value);
            }
        });
        updates.forEach((name, value) -> {
            parameters.get(name).updateTensor(value);
            state.parameterSnapshots.put(name, value);
        });
        state.step = nextStep;
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
