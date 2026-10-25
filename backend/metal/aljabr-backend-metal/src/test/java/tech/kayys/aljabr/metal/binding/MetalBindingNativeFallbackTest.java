package tech.kayys.aljabr.metal.binding;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalBindingNativeFallbackTest {
    @Test
    void decodeLayerRegistrationKeepsSharedKvOwnerInStableAbiPosition() {
        // The native ABI must begin with (layer, kvLayerIndex), followed by
        // seven pointers, then metadata. In particular, kvLayerIndex must not
        // be duplicated after isSwa: that would desynchronize FFM from C.
        Class<?>[] expectedParameterTypes = {
                int.class, int.class, // layer, kvLayerIndex
                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                MemorySegment.class, // seven weight pointers
                int.class, int.class, int.class, int.class, int.class, int.class, int.class,
                int.class, int.class, int.class, int.class, int.class, int.class,
                int.class, int.class, int.class, int.class,
                float.class, int.class,
                float.class, float.class,
                int.class, int.class, int.class, int.class, int.class
        };

        List<Method> methods = Arrays.stream(MetalBinding.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("registerDecodeLayer"))
                .toList();

        assertEquals(1, methods.size(), "registerDecodeLayer should have one stable entry point");
        assertArrayEquals(expectedParameterTypes, methods.get(0).getParameterTypes());
    }

    @Test
    void fallbackInitializationIsNotReportedAsActiveRuntime() {
        MetalBinding.reset();
        MetalBinding.initializeFallback();

        MetalBinding binding = MetalBinding.getInstance();

        assertFalse(binding.isNativeAvailable());
        assertFalse(binding.isRuntimeActive());
        assertFalse(binding.supportsSwigluFfnMatvecRowsBf16());
        assertFalse(binding.supportsGegluFfnMatvecRowsBf16());
        assertEquals(0, binding.bf16FfnMatvecRowsVariant());
    }

    @Test
    void nativeElementwiseStillWorksWhenRuntimeCpuFallbackIsActive() throws Exception {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertTrue(binding.isNativeAvailable());
        assertFalse(binding.isRuntimeActive());
        assertFalse(binding.nativeElementwiseKernelsAvailable());
        forceRuntimeCpuFallback(binding);

        assertFalse(binding.isRuntimeActive());
        assertTrue(binding.nativeElementwiseFallbackAvailable());
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment gate = arena.allocate(ValueLayout.JAVA_FLOAT, 4);
            MemorySegment up = arena.allocate(ValueLayout.JAVA_FLOAT, 4);
            MemorySegment out = arena.allocate(ValueLayout.JAVA_FLOAT, 4);
            for (int i = 0; i < 4; i++) {
                gate.setAtIndex(ValueLayout.JAVA_FLOAT, i, 0.5f);
                up.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.5f);
            }

            assertEquals(0, binding.siluFfn(out, gate, up, 4));
            float expected = (0.5f / (1.0f + (float) Math.exp(-0.5f))) * 1.5f;
            assertEquals(expected, out.getAtIndex(ValueLayout.JAVA_FLOAT, 0), 1e-6f);
        }
    }

    @Test
    void fallbackInitializationCanUpgradeToNativeLibrary() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        MetalBinding.initializeFallback();
        assertFalse(MetalBinding.getInstance().isNativeAvailable());

        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();

        assertTrue(binding.isNativeAvailable());
        assertFalse(binding.isRuntimeActive());
    }

    @Test
    void nativeAttentionZeroLengthContextZeroFillsOutput() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init());
        assumeTrue(binding.isRuntimeActive(), "Native Metal runtime is not active");

        int batch = 1;
        int tokens = 2;
        int heads = 1;
        int kvHeads = 1;
        int headDim = 2;
        int blockSize = 4;
        int maxBlocks = 1;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.JAVA_FLOAT, batch * tokens * heads * headDim);
            MemorySegment q = arena.allocate(ValueLayout.JAVA_FLOAT, batch * tokens * heads * headDim);
            MemorySegment k = arena.allocate(ValueLayout.JAVA_FLOAT, maxBlocks * kvHeads * blockSize * headDim);
            MemorySegment v = arena.allocate(ValueLayout.JAVA_FLOAT, maxBlocks * kvHeads * blockSize * headDim);
            MemorySegment blockTable = arena.allocate(ValueLayout.JAVA_INT, batch * maxBlocks);
            MemorySegment contextLens = arena.allocate(ValueLayout.JAVA_INT, batch);

            for (int i = 0; i < batch * tokens * heads * headDim; i++) {
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, 42.0f);
                q.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
            }
            blockTable.setAtIndex(ValueLayout.JAVA_INT, 0, 0);
            contextLens.setAtIndex(ValueLayout.JAVA_INT, 0, 0);

            assertEquals(0, binding.attentionGqaWindowed(out, q, k, v, blockTable, contextLens,
                    batch, tokens, heads, kvHeads, headDim, blockSize, maxBlocks,
                    1.0f, 1, 0, 0, 0.0f));
            for (int i = 0; i < batch * tokens * heads * headDim; i++) {
                assertEquals(0.0f, out.getAtIndex(ValueLayout.JAVA_FLOAT, i), 0.0f);
            }
        }
    }

    private static Path findBuiltDylib() {
        return List.of(
                Path.of("target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of(System.getProperty("user.home"), ".aljabr", "libs", "libaljabr_metal.dylib"))
                .stream()
                .filter(Files::isRegularFile)
                .findFirst()
                .orElse(null);
    }

    private static void forceRuntimeCpuFallback(MetalBinding binding) throws Exception {
        Field fallback = MetalBinding.class.getDeclaredField("runtimeCpuFallback");
        fallback.setAccessible(true);
        fallback.setBoolean(binding, true);
    }
}
