package tech.kayys.aljabr.diffusion.execution.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.scheduler.api.PredictionType;

class SigmaStepContextTest {

    @Test
    void exposesSignedSigmaDeltaAndTerminalState() {
        SigmaStepContext ordinary =
                new SigmaStepContext(2, 4.0d, 1.5d, PredictionType.EPSILON);
        assertEquals(-2.5d, ordinary.deltaSigma(), 0.0d);
        assertFalse(ordinary.terminalStep());

        SigmaStepContext terminal =
                new SigmaStepContext(3, 1.5d, 0.0d, PredictionType.SAMPLE);
        assertEquals(-1.5d, terminal.deltaSigma(), 0.0d);
        assertTrue(terminal.terminalStep());
    }

    @Test
    void acceptsFlowAndVelocityAsExplicitSemanticsWithoutAssumingConversions() {
        assertEquals(PredictionType.FLOW,
                new SigmaStepContext(0, 2.0d, 1.0d, PredictionType.FLOW).predictionType());
        assertEquals(PredictionType.VELOCITY,
                new SigmaStepContext(0, 2.0d, 1.0d, PredictionType.VELOCITY).predictionType());
    }

    @Test
    void rejectsInvalidCoordinatesAndMissingPredictionType() {
        assertThrows(IllegalArgumentException.class,
                () -> new SigmaStepContext(-1, 2.0d, 1.0d, PredictionType.EPSILON));
        assertThrows(IllegalArgumentException.class,
                () -> new SigmaStepContext(0, 0.0d, 0.0d, PredictionType.EPSILON));
        assertThrows(IllegalArgumentException.class,
                () -> new SigmaStepContext(0, 1.0d, 2.0d, PredictionType.EPSILON));
        assertThrows(NullPointerException.class,
                () -> new SigmaStepContext(0, 1.0d, 0.0d, null));
    }
}
