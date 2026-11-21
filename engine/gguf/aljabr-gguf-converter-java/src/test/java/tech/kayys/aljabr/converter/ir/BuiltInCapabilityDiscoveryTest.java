package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.ModelConversionPlanners;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BuiltInCapabilityDiscoveryTest {

    @Test
    void discoversBuiltInTargetFormatCapabilities() {
        var planner = ModelConversionPlanners.discover();

        assertTrue(planner.supportsTarget("gguf"));
        assertTrue(planner.supportsTarget("safetensors"));
    }
}
