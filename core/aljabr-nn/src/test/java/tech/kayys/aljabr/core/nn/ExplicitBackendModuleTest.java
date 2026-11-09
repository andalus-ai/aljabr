package tech.kayys.aljabr.core.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class ExplicitBackendModuleTest {

    @Test
    void linearParametersAndOutputsUseTheConfiguredBackend() {
        CpuBackend backend = new CpuBackend();
        Linear linear = new Linear(backend, 2, 3);
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, backend, 1, 2);
        Tensor output = null;
        try {
            assertEquals(backend, linear.computeBackend());
            assertEquals(backend, linear.parameters().get("weight").backend());
            output = linear.forward(input);
            assertEquals(backend, output.backend());
        } finally {
            if (output != null) output.release();
            input.release();
            linear.close();
        }
    }

    @Test
    void linearRejectsInputFromAnotherBackend() {
        CpuBackend configuredBackend = new CpuBackend();
        CpuBackend inputBackend = new CpuBackend();
        Linear linear = new Linear(configuredBackend, 2, 3);
        Tensor input = TensorFactory.of(new float[] { 1f, 2f }, inputBackend, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> linear.forward(input));
        } finally {
            input.release();
            linear.close();
        }
    }

    @Test
    void parameterizedLayersCreateAllStateOnTheExplicitBackend() {
        CpuBackend backend = new CpuBackend();
        List<NNModule> modules = List.of(
                new Conv1d(backend, 2, 4, 3, 1, 1),
                new Conv2d(backend, 2, 4, 3),
                new Embedding(backend, 16, 8),
                new BatchNorm2d(backend, 4),
                new LayerNorm(backend, 8),
                new RMSNorm(backend, 8, 1e-5f),
                new MultiheadAttention(backend, 8, 2));

        try {
            for (NNModule module : modules) {
                assertEquals(backend, module.computeBackend(), module.getClass().getSimpleName());
                module.stateDict().forEach((name, tensor) ->
                        assertEquals(backend, tensor.backend(), module.getClass().getSimpleName() + "." + name));
            }
        } finally {
            modules.forEach(NNModule::close);
        }
    }

    @Test
    void explicitSequentialAndTransformerContainersPropagateBackendOwnership() {
        CpuBackend backend = new CpuBackend();
        Sequential sequential = new Sequential(backend, new Linear(backend, 8, 8));
        TransformerEncoderLayer encoder = new TransformerEncoderLayer(backend, 8, 2, 16, 0.1f);
        TransformerDecoderLayer decoder = new TransformerDecoderLayer(backend, 8, 2, 16, 0.1f);
        TransformerEncoder encoderModel = TransformerEncoder.builder(backend).nLayers(2).dModel(8).build();
        try {
            assertEquals(backend, sequential.computeBackend());
            assertEquals(backend, encoder.computeBackend());
            assertEquals(backend, decoder.computeBackend());
            assertEquals(backend, encoderModel.computeBackend());
            for (NNModule module : List.of(sequential, encoder, decoder)) {
                module.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
            }
        } finally {
            sequential.close();
            encoder.close();
            decoder.close();
            encoderModel.close();
        }
    }

    @Test
    void explicitContainerRejectsSubmoduleWithDifferentBackend() {
        CpuBackend owner = new CpuBackend();
        CpuBackend other = new CpuBackend();
        assertThrows(IllegalArgumentException.class,
                () -> new Sequential(owner, new Linear(other, 4, 4)));
    }

    @Test
    void loraParametersShareTheBaseLinearBackend() {
        CpuBackend backend = new CpuBackend();
        Linear base = new Linear(backend, 8, 4);
        LoRALayer lora = new LoRALayer(backend, base, 2, 4.0);
        try {
            assertEquals(backend, lora.computeBackend());
            lora.parameters().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
            assertEquals(backend, lora.children().get("base").computeBackend());
        } finally {
            lora.close();
        }
    }
}
