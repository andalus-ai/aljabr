package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Stateless stochastic gradient descent. */
public final class SGD implements Optimizer {
    private final float learningRate;

    public SGD(float learningRate) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        this.learningRate = learningRate;
    }

    public float learningRate() {
        return learningRate;
    }

    @Override
    public void step(Map<String, Parameter> parameters, Map<String, Tensor> gradients) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        Objects.requireNonNull(gradients, "gradients must not be null");

        Map<String, Tensor> updates = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = Objects.requireNonNull(entry.getKey(), "parameter name must not be null");
            Parameter parameter = Objects.requireNonNull(entry.getValue(), "parameter " + name);
            Tensor gradient = gradients.get(name);
            if (gradient == null || !parameter.isTrainable()) {
                continue;
            }
            parameter.validateGradient(gradient);
            Tensor current = parameter.tensor();
            updates.put(name, current.sub(gradient.mul(learningRate)));
        }
        // Validation and computation finish before any registered value changes.
        updates.forEach((name, value) -> parameters.get(name).updateTensor(value));
    }

    @Override
    public void reset() {
        // SGD has no per-parameter state.
    }
}
