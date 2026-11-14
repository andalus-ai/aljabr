package tech.kayys.aljabr.autograd.nn;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class DifferentiableLinearTest {

    @Test
    void computesInputWeightAndBiasGradientsForBatchedLinear() {
        CpuBackend backend = new CpuBackend();
        Tensor weight = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
        Tensor bias = TensorFactory.of(new float[] { 0.5f, -0.5f }, backend, 2);
        Tensor inputValue = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f }, backend, 2, 2);
        try (Linear layer = new Linear(weight, bias);
             AutogradContext context = new AutogradContext()) {
            Variable input = context.variable(inputValue);
            DifferentiableLinear differentiable = new DifferentiableLinear(layer, context);

            differentiable.forward(input).sum().backward();

            assertThat(input.grad().toFloatArray()).containsExactly(4f, 6f, 4f, 6f);
            assertThat(differentiable.weight().grad().toFloatArray())
                    .containsExactly(4f, 6f, 4f, 6f);
            assertThat(differentiable.bias().grad().toFloatArray()).containsExactly(2f, 2f);
        } finally {
            inputValue.release();
        }
    }

    @Test
    void supportsLinearWithoutBias() {
        CpuBackend backend = new CpuBackend();
        Tensor weight = TensorFactory.of(new float[] { 2f, 3f }, backend, 1, 2);
        Tensor inputValue = TensorFactory.of(new float[] { 4f, 5f }, backend, 1, 2);
        try (Linear layer = new Linear(weight, null);
             AutogradContext context = new AutogradContext()) {
            Variable input = context.variable(inputValue);
            DifferentiableLinear differentiable = new DifferentiableLinear(layer, context);

            differentiable.forward(input).sum().backward();

            assertThat(input.grad().toFloatArray()).containsExactly(2f, 3f);
            assertThat(differentiable.weight().grad().toFloatArray()).containsExactly(4f, 5f);
            assertThat(differentiable.bias()).isNull();
        } finally {
            inputValue.release();
        }
    }

    @Test
    void exposesValidatedParameterGradientsWithoutApplyingUpdates() {
        CpuBackend backend = new CpuBackend();
        Tensor weight = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        Tensor inputValue = TensorFactory.of(new float[] { 3f, 4f }, backend, 1, 2);
        try (Linear layer = new Linear(weight, null);
             AutogradContext context = new AutogradContext()) {
            Variable input = context.variable(inputValue);
            DifferentiableLinear differentiable = new DifferentiableLinear(layer, context);

            assertThat(differentiable.weightParameter())
                    .isSameAs(layer.namedParameterObjects().get("weight"));
            assertThat(differentiable.weightGradient()).isNull();

            differentiable.forward(input).sum().backward();

            assertThat(differentiable.weightGradient().toFloatArray()).containsExactly(3f, 4f);
            assertThat(differentiable.biasGradient()).isNull();
        } finally {
            inputValue.release();
        }
    }

    @Test
    void rejectsUseAfterRegisteredParameterIsReplaced() {
        CpuBackend backend = new CpuBackend();
        Tensor weight = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        Tensor inputValue = TensorFactory.of(new float[] { 3f, 4f }, backend, 1, 2);
        try (Linear layer = new Linear(weight, null);
             AutogradContext context = new AutogradContext()) {
            Variable input = context.variable(inputValue);
            DifferentiableLinear differentiable = new DifferentiableLinear(layer, context);
            var parameter = differentiable.weightParameter();
            Tensor replacement = TensorFactory.of(new float[] { 0f, 0f }, backend, 1, 2);
            parameter.updateTensor(replacement);

            org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalStateException.class, () -> differentiable.forward(input));
        } finally {
            inputValue.release();
            weight.release(); // The module no longer owns the replaced tensor.
        }
    }
}
