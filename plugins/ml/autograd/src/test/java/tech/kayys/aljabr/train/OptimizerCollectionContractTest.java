package tech.kayys.aljabr.train;

import static org.junit.jupiter.api.Assertions.assertThrows;
import java.util.List;
import org.junit.jupiter.api.Test;

class OptimizerCollectionContractTest {

    @Test
    void sgdRejectsCollectionUpdatesThatCannotReplaceImmutableTensors() {
        assertThrows(UnsupportedOperationException.class, () -> new SGD(0.01f).step(List.of()));
    }

    @Test
    void adamwRejectsCollectionUpdatesThatCannotReplaceImmutableTensors() {
        assertThrows(UnsupportedOperationException.class,
                () -> new AdamW(0.001f, 0.9f, 0.999f, 1.0e-8f, 0.01f).step(List.of()));
    }

    @Test
    void sgdUpdatesLiveModuleParameterThroughHandle() {
        tech.kayys.aljabr.backend.cpu.CpuBackend backend = new tech.kayys.aljabr.backend.cpu.CpuBackend();
        try (tech.kayys.aljabr.core.nn.Linear layer =
                new tech.kayys.aljabr.core.nn.Linear(backend, 2, 2, true)) {
            var parameter = layer.namedParameterObjects().get("weight");
            var original = parameter.tensor();
            var gradient = original.zerosLike().add(1.0f);
            new SGD(0.1f).stepParameters(java.util.Map.of("weight", parameter),
                    java.util.Map.of("weight", gradient));
            org.junit.jupiter.api.Assertions.assertNotSame(original, parameter.tensor());
            org.junit.jupiter.api.Assertions.assertSame(parameter.tensor(), layer.parameters().get("weight"));
            org.junit.jupiter.api.Assertions.assertTrue(parameter.requiresGrad());
            org.junit.jupiter.api.Assertions.assertNotEquals(original.toFloatArray()[0], parameter.tensor().toFloatArray()[0]);
        }
    }

    @Test
    void adamwUpdatesLiveModuleParameterThroughHandle() {
        tech.kayys.aljabr.backend.cpu.CpuBackend backend = new tech.kayys.aljabr.backend.cpu.CpuBackend();
        try (tech.kayys.aljabr.core.nn.Linear layer =
                new tech.kayys.aljabr.core.nn.Linear(backend, 2, 2, true)) {
            var parameter = layer.namedParameterObjects().get("weight");
            var original = parameter.tensor();
            var gradient = original.zerosLike().add(1.0f);
            new AdamW(0.01f, 0.9f, 0.999f, 1.0e-8f, 0.01f).stepParameters(
                    java.util.Map.of("weight", parameter), java.util.Map.of("weight", gradient));
            org.junit.jupiter.api.Assertions.assertNotSame(original, parameter.tensor());
            org.junit.jupiter.api.Assertions.assertSame(parameter.tensor(), layer.parameters().get("weight"));
            org.junit.jupiter.api.Assertions.assertTrue(parameter.requiresGrad());
        }
    }

    @Test
    void parameterHandleOptimizersLeaveFrozenParametersUntouched() {
        tech.kayys.aljabr.backend.cpu.CpuBackend backend = new tech.kayys.aljabr.backend.cpu.CpuBackend();
        try (tech.kayys.aljabr.core.nn.Linear layer =
                new tech.kayys.aljabr.core.nn.Linear(backend, 2, 2, true)) {
            var parameter = layer.namedParameterObjects().get("weight");
            parameter.setRequiresGrad(false);
            var original = parameter.tensor();
            var gradient = original.zerosLike().add(1.0f);
            new SGD(0.1f).stepParameters(java.util.Map.of("weight", parameter),
                    java.util.Map.of("weight", gradient));
            org.junit.jupiter.api.Assertions.assertSame(original, parameter.tensor());
        }
    }

    @Test
    void optimizersRejectInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new SGD(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.001f, 1.0f, 0.999f, 1.0e-8f, 0.01f));
    }
}
