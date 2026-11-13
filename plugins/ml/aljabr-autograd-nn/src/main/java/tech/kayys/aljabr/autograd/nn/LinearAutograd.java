package tech.kayys.aljabr.autograd.nn;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Differentiable adapter for the existing {@link Linear} module.
 *
 * <p>This adapter is deliberately optional: the NN module and tensor foundation
 * remain independent of the autograd implementation.</p>
 */
public final class LinearAutograd {

    private LinearAutograd() {
    }

    /**
     * Executes a rank-two linear layer while recording gradients for its input
     * and registered parameters.
     *
     * @param context differentiation context that owns the forward graph
     * @param layer existing NN linear module
     * @param input tracked rank-two input
     * @return output and tracked parameter handles for this forward pass
     */
    public static ForwardPass forward(AutogradContext context, Linear layer, Variable input) {
        Objects.requireNonNull(context, "context must not be null");
        Objects.requireNonNull(layer, "layer must not be null");
        Objects.requireNonNull(input, "input must not be null");

        Map<String, Parameter> parameters = layer.namedParameterObjects();
        Parameter weightParameter = parameters.get("weight");
        if (weightParameter == null) {
            throw new IllegalArgumentException("Linear module has no registered weight parameter");
        }

        Tensor inputValue = input.value();
        Tensor weightValue = weightParameter.tensor();
        if (inputValue.shape().rank() != 2 || weightValue.shape().rank() != 2) {
            throw new UnsupportedOperationException("Linear autograd currently requires rank-two input and weight");
        }
        if (inputValue.shape().dim(1) != weightValue.shape().dim(1)) {
            throw new IllegalArgumentException("Linear input feature count does not match weight shape");
        }

        Variable weight = context.variable(weightValue, weightParameter.requiresGrad());
        Variable output = input.matmul(weight.transpose(0, 1));

        Map<String, Variable> parameterVariables = new LinkedHashMap<>();
        parameterVariables.put("weight", weight);

        Parameter biasParameter = parameters.get("bias");
        if (biasParameter != null) {
            Variable bias = context.variable(biasParameter.tensor(), biasParameter.requiresGrad());
            output = output.add(bias);
            parameterVariables.put("bias", bias);
        }

        return new ForwardPass(output, input, parameterVariables, parameters);
    }

    /** Result of one differentiable linear forward pass. */
    public static final class ForwardPass {

        private final Variable output;
        private final Variable input;
        private final Map<String, Variable> parameterVariables;
        private final Map<String, Parameter> parameters;

        private ForwardPass(
                Variable output,
                Variable input,
                Map<String, Variable> parameterVariables,
                Map<String, Parameter> parameters) {
            this.output = output;
            this.input = input;
            this.parameterVariables = Collections.unmodifiableMap(new LinkedHashMap<>(parameterVariables));
            this.parameters = Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
        }

        public Variable output() {
            return output;
        }

        public Variable input() {
            return input;
        }

        /** Tracked parameters used by this forward pass, keyed by local parameter name. */
        public Map<String, Variable> parameterVariables() {
            return parameterVariables;
        }

        /**
         * Returns gradients associated with the module's stable parameter handles.
         * Gradient values are borrowed from the forward pass' autograd context.
         */
        public Map<Parameter, Tensor> parameterGradients() {
            Map<Parameter, Tensor> gradients = new LinkedHashMap<>();
            parameterVariables.forEach((name, variable) -> {
                Tensor gradient = variable.grad();
                if (gradient != null) {
                    gradients.put(parameters.get(name), gradient);
                }
            });
            return Collections.unmodifiableMap(gradients);
        }
    }
}
