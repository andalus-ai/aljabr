package tech.kayys.aljabr.autograd.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class LinearAutogradTest {

    @Test
    void computesInputWeightAndBiasGradients() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, backend, 2, 3);
        Tensor weight = TensorFactory.of(new float[] { 1f, 0f, 0f, 0f, 1f, 0f }, backend, 2, 3);
        Tensor bias = TensorFactory.of(new float[] { 0.5f, -0.5f }, backend, 2);
        Linear layer = new Linear(weight, bias);

        try (AutogradContext context = new AutogradContext()) {
            Variable trackedInput = context.variable(input);
            LinearAutograd.ForwardPass pass = LinearAutograd.forward(context, layer, trackedInput);

            assertArrayEquals(new float[] { 1.5f, 1.5f, 4.5f, 4.5f },
                    pass.output().value().toFloatArray(), 1.0e-6f);

            pass.output().sum().backward();

            assertArrayEquals(new float[] { 1f, 1f, 0f, 1f, 1f, 0f },
                    pass.input().grad().toFloatArray(), 1.0e-6f);

            Map<Parameter, Tensor> gradients = pass.parameterGradients();
            assertEquals(2, gradients.size());
            assertArrayEquals(new float[] { 5f, 7f, 9f, 5f, 7f, 9f },
                    gradients.get(layer.namedParameterObjects().get("weight")).toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] { 2f, 2f },
                    gradients.get(layer.namedParameterObjects().get("bias")).toFloatArray(), 1.0e-6f);
        } finally {
            layer.close();
            input.release();
            backend.close();
        }
    }

    @Test
    void doesNotExposeGradientsForFrozenParameters() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        Tensor weight = TensorFactory.of(new float[] { 3f, 4f }, backend, 1, 2);
        Linear layer = new Linear(weight, null);
        layer.namedParameterObjects().get("weight").setRequiresGrad(false);

        try (AutogradContext context = new AutogradContext()) {
            LinearAutograd.ForwardPass pass =
                    LinearAutograd.forward(context, layer, context.variable(input));
            pass.output().sum().backward();
            assertEquals(0, pass.parameterGradients().size());
        } finally {
            layer.close();
            input.release();
            backend.close();
        }
    }
}
