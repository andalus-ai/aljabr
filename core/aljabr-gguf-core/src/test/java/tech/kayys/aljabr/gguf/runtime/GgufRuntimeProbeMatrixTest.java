package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.assertPreparedMatrixProbe;
import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.oneTensorModel;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8_1Block;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRuntimeProbeMatrixTest {
    @Test
    void probesPreparedQ32MatrixPathForQ4_0Tensor() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{32, 2}, 2, 2L * 18),
                    36,
                    2,
                    2);

            assertPreparedMatrixProbe(probe, "Q4_0", 2, 32, -1.7058823f);
            assertTrue(probe.compactSummary().contains("cachedGenericMatVec="));
        }
    }

    @Test
    void probesPreparedQ8MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(34);
            writeQ8Block(segment, (short) 0x3c00, (byte) -2);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{32, 1}, 8, 34),
                    34,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "Q8_0", 1, 32, 1.7647059f);
        }
    }

    @Test
    void probesPreparedQ8_1MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(36);
            writeQ8_1Block(segment, (byte) -2);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{32, 1}, 9, 36),
                    36,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "Q8_1", 1, 32, 1.7647059f);
        }
    }
}
