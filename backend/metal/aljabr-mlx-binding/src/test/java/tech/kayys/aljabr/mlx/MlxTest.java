package tech.kayys.aljabr.mlx;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.mlx.binding.MlxBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class MlxTest {

    @Test
    public void testMatmul() {
        // Path to the built dylib
        Path libPath = Paths.get("build/native/lib/libaljabr_mlx_bridge.dylib").toAbsolutePath();
        MlxBinding.load(libPath);
        MlxBinding.init();

        try (Arena arena = Arena.ofConfined()) {
            float[] aData = { 1, 2, 3, 4 };
            long[] aShape = { 2, 2 };
            float[] bData = { 5, 6, 7, 8 };
            long[] bShape = { 2, 2 };

            MemorySegment a = MlxBinding.arrayFromFloat(arena, aData, aShape);
            MemorySegment b = MlxBinding.arrayFromFloat(arena, bData, bShape);

            MemorySegment c = MlxBinding.matmul(a, b);
            MlxBinding.eval(c);

            float[] cData = new float[4];
            MlxBinding.getData(c, cData);

            // [1,2] [5,6] [1*5+2*7, 1*6+2*8] [19, 22]
            // [3,4] [7,8] = [3*5+4*7, 3*6+4*8] = [43, 50]
            float[] expected = { 19, 22, 43, 50 };
            assertArrayEquals(expected, cData, 1e-5f);

            MlxBinding.free(a);
            MlxBinding.free(b);
            MlxBinding.free(c);
        }
    }
}
