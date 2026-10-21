package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static tech.kayys.aljabr.gguf.runtime.GgufOffFx.assertPreparedQ8Offset;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4NLBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4XSBlock;

class GgufIQOffTest {
    @Test
    void buildsPreparedIQMatricesFromNonZeroTensorOffsets() {
        assertPreparedQ8Offset(
                20,
                32,
                18,
                (block, quant) -> writeIQ4NLBlock(block, (short) 0x3c00, quant),
                (byte) 0x99,
                416.0f,
                36L);
        assertPreparedQ8Offset(
                23,
                256,
                136,
                (block, quant) -> writeIQ4XSBlock(block, (short) 0x3c00, quant),
                (byte) 0x99,
                3328.0f,
                288L);
    }
}
