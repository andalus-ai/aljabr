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
    void optimizersRejectInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> new SGD(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.001f, 1.0f, 0.999f, 1.0e-8f, 0.01f));
    }
}
