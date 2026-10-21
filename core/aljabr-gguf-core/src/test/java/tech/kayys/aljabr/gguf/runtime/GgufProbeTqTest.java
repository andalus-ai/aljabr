package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.assertPreparedMatrixProbe;
import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.oneTensorModel;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ1_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ2_0Block;

class GgufProbeTqTest {
    @Test
    void probesPreparedTQ1_0MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(54);
            writeTQ1_0Block(segment, (byte) 49);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{256, 1}, 34, 54),
                    54,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "TQ1_0", 1, 256, 1.5882353f);
        }
    }

    @Test
    void probesPreparedTQ2_0MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(66);
            writeTQ2_0Block(segment, (byte) 0x24);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{256, 1}, 35, 66),
                    66,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "TQ2_0", 1, 256, 0.35294118f);
        }
    }
}
