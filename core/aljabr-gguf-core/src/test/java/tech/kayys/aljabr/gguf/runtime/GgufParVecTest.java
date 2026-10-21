package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufParVecTest {
    @Test
    void byteDotVectorAutoModeEnablesOn128BitOrWiderFloatVectors() {
        assertEquals(
                GgufTensorOps.preferredFloatVectorLanes() >= 4,
                GgufTensorOps.byteDotVectorPreferred());
        assertEquals(GgufTensorOps.byteDotVectorPreferred(), GgufTensorOps.resolveByteDotVectorEnabled(null));
        assertEquals(GgufTensorOps.byteDotVectorPreferred(), GgufTensorOps.resolveByteDotVectorEnabled(""));
        assertEquals(GgufTensorOps.byteDotVectorPreferred(), GgufTensorOps.resolveByteDotVectorEnabled("auto"));
        assertTrue(GgufTensorOps.resolveByteDotVectorEnabled("true"));
        assertFalse(GgufTensorOps.resolveByteDotVectorEnabled("false"));
    }
}
