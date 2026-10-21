package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static tech.kayys.aljabr.gguf.runtime.GgufOffFx.assertPreparedQ8Offset;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeNVFP4Block;

class GgufF4OffTest {
    @Test
    void buildsPreparedFp4MatricesFromNonZeroTensorOffsets() {
        assertPreparedQ8Offset(
                39,
                32,
                17,
                (block, quant) -> writeMXFP4Block(block, (byte) 128, quant),
                (byte) 0x3C,
                -16.0f,
                36L);
        assertPreparedQ8Offset(
                40,
                64,
                36,
                (block, quant) -> writeNVFP4Block(block, (byte) 0x40, quant),
                (byte) 0x3C,
                -32.0f,
                80L);
    }
}
