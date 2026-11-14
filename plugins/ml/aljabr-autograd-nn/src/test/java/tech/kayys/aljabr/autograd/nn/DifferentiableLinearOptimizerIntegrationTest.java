package tech.kayys.aljabr.autograd.nn;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.autograd.AutogradContext;
import tech.kayys.aljabr.autograd.Variable;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.optim.SGD;

class DifferentiableLinearOptimizerIntegrationTest {

    @Test
    void sgdStepReducesLinearMeanSquaredError() {
        CpuBackend backend = new CpuBackend();
        try (Linear layer = new Linear(backend, 1, 1, false)) {
            Parameter weight = layer.namedParameterObjects().get("weight");
            weight.updateTensor(TensorFactory.zeros(backend, 1, 1));

            Tensor inputValue = TensorFactory.ones(backend, 1, 1);
            Tensor targetValue = TensorFactory.ones(backend, 1, 1).mul(2.0f);
            float initialLoss;
            try (AutogradContext context = new AutogradContext()) {
                Variable input = context.variable(inputValue, false);
                Variable target = context.variable(targetValue, false);
                DifferentiableLinear linear = new DifferentiableLinear(layer, context);
                Variable prediction = linear.forward(input);
                Variable error = prediction.sub(target);
                Variable loss = error.mul(error).mean();
                initialLoss = loss.value().item();
                loss.backward();

                Tensor weightGradient = linear.weightGradient();
                assertThat(weightGradient).isNotNull();
                new SGD(0.1f).step(
                        Map.of("weight", linear.weightParameter()),
                        Map.of("weight", weightGradient));
            }

            try (AutogradContext context = new AutogradContext()) {
                Variable input = context.variable(inputValue, false);
                Variable target = context.variable(targetValue, false);
                Variable prediction = new DifferentiableLinear(layer, context).forward(input);
                Variable error = prediction.sub(target);
                float updatedLoss = error.mul(error).mean().value().item();
                assertThat(updatedLoss).isLessThan(initialLoss);
            } finally {
                inputValue.release();
                targetValue.release();
            }
        }
    }
}
