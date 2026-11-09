package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.nn.NNModule;

class ExplicitBackendModelTest {

    @Test
    void bertFactoriesPropagateTheExplicitBackend() {
        CpuBackend backend = new CpuBackend();
        NNModule encoder = BERT.bertBase(backend, BERT.VOCAB_SIZE_BASE);
        NNModule classifier = BERT.forSequenceClassification(3, BERT.VOCAB_SIZE_BASE, backend);
        try {
            assertEquals(backend, encoder.computeBackend());
            assertEquals(backend, classifier.computeBackend());
            assertEquals(backend, classifier.children().get("encoder").computeBackend());
            assertEquals(backend, classifier.children().get("classifier").computeBackend());
            classifier.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
        } finally {
            encoder.close();
            classifier.close();
        }
    }

    @Test
    void gpt2PropagatesTheExplicitBackendToEveryChildAndParameter() {
        CpuBackend backend = new CpuBackend();
        GPT2.GPT2Model model = new GPT2.GPT2Model(backend, 8, 2, 2, 16, 32, 64, 0.1f);
        try {
            assertEquals(backend, model.computeBackend());
            model.children().values().forEach(child -> assertEquals(backend, child.computeBackend()));
            model.stateDict().values().forEach(tensor -> assertEquals(backend, tensor.backend()));
        } finally {
            model.close();
        }
    }

    @Test
    void bertAndGpt2RejectInputsFromAnotherBackend() {
        CpuBackend owner = new CpuBackend();
        CpuBackend other = new CpuBackend();
        NNModule bert = BERT.bertBase(owner, BERT.VOCAB_SIZE_BASE);
        GPT2.GPT2Model gpt2 = new GPT2.GPT2Model(owner, 8, 2, 1, 16, 32, 64, 0.1f);
        Tensor input = TensorFactory.of(new float[8], other, 1, 1, 8);
        try {
            assertThrows(IllegalArgumentException.class, () -> bert.forward(input));
            assertThrows(IllegalArgumentException.class, () -> gpt2.forward(input));
        } finally {
            input.release();
            bert.close();
            gpt2.close();
        }
    }
}
