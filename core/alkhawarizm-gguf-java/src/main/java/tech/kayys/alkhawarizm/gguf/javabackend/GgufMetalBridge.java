package tech.kayys.alkhawarizm.gguf.javabackend;

import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;
import tech.kayys.alkhawarizm.gguf.loader.GGUFTensorInfo;
import tech.kayys.alkhawarizm.gguf.runtime.GgufTensorOps;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Bridge between Java GGUF backend and Alkhawarizm Metal GPU acceleration.
 *
 * <p>Uses reflection / MethodHandle to load {@code MetalBinding} dynamically if present on the classpath,
 * preventing tight coupling or circular dependencies between {@code alkhawarizm-gguf-java} and
 * {@code alkhawarizm-backend-metal}.</p>
 */
public final class GgufMetalBridge {
    private static final Logger LOG = Logger.getLogger(GgufMetalBridge.class.getName());

    private static final boolean METAL_ENABLED;
    private static final Object METAL_BINDING_INSTANCE;
    private static final MethodHandle MATVEC_Q4K_MH;
    private static final MethodHandle MATVEC_Q6K_MH;
    private static final MethodHandle MATVEC_Q8_0_MH;

    // Thread-local or reusable buffers for inputs/outputs to prevent allocations in inner loop
    private static final ThreadLocal<MetalBufferCache> BUFFERS = ThreadLocal.withInitial(MetalBufferCache::new);

    static {
        boolean enabled = false;
        Object instance = null;
        MethodHandle q4kHandle = null;
        MethodHandle q6kHandle = null;
        MethodHandle q8_0Handle = null;

        boolean disableMetal = Boolean.getBoolean("alkhawarizm.metal.disable") ||
                "false".equalsIgnoreCase(System.getProperty("alkhawarizm.metal.enabled"));

        if (!disableMetal) {
            try {
                // Ensure discovery checks for libalkhawarizm_metal.dylib
                String override = System.getProperty("alkhawarizm.metal.dylib");
                if (override == null || override.contains("libaqli_metal.dylib")) {
                    String home = System.getProperty("user.home");
                    if (home != null) {
                        java.nio.file.Path p = java.nio.file.Path.of(home, ".aqli", "libs", "libalkhawarizm_metal.dylib");
                        if (java.nio.file.Files.exists(p)) {
                            System.setProperty("alkhawarizm.metal.dylib", p.toString());
                        }
                    }
                }

                Class<?> mbClass = Class.forName("tech.kayys.alkhawarizm.metal.binding.MetalBinding");
                MethodHandles.Lookup lookup = MethodHandles.publicLookup();

                MethodHandle initMH = lookup.findStatic(mbClass, "initialize", MethodType.methodType(boolean.class));
                boolean initialized = (boolean) initMH.invokeExact();

                if (initialized) {
                    MethodHandle getInstanceMH = lookup.findStatic(mbClass, "getInstance", MethodType.methodType(mbClass));
                    instance = getInstanceMH.invoke();

                    MethodHandle initRuntimeMH = lookup.findVirtual(mbClass, "init", MethodType.methodType(int.class));
                    int rc = (int) initRuntimeMH.invoke(instance);

                    if (rc == 0) {
                        MethodHandle isActiveMH = lookup.findVirtual(mbClass, "isRuntimeActive", MethodType.methodType(boolean.class));
                        boolean active = (boolean) isActiveMH.invoke(instance);

                        if (active) {
                            // matvecTransposedRightQ4K(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N)
                            q4kHandle = lookup.findVirtual(mbClass, "matvecTransposedRightQ4K",
                                    MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class));

                            // matvecTransposedRightQ6K(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N)
                            try {
                                q6kHandle = lookup.findVirtual(mbClass, "matvecTransposedRightQ6K",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class));
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightQ8_0(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N)
                            try {
                                q8_0Handle = lookup.findVirtual(mbClass, "matvecTransposedRightQ8_0",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class));
                            } catch (NoSuchMethodException ignored) {
                            }

                            enabled = true;
                            LOG.info("Alkhawarizm Metal GPU acceleration enabled for Java GGUF backend.");
                        }
                    }
                }
            } catch (ClassNotFoundException e) {
                LOG.fine("MetalBinding not on classpath, continuing with CPU vector execution.");
            } catch (Throwable t) {
                LOG.log(Level.WARNING, "Failed to initialize Alkhawarizm Metal GPU acceleration: " + t.getMessage(), t);
            }
        }

        METAL_ENABLED = enabled;
        METAL_BINDING_INSTANCE = instance;
        MATVEC_Q4K_MH = q4kHandle;
        MATVEC_Q6K_MH = q6kHandle;
        MATVEC_Q8_0_MH = q8_0Handle;
    }

    private static final class MetalBufferCache {
        final Arena arena = Arena.ofShared();
        MemorySegment segA;
        int capA;
        MemorySegment segC;
        int capC;

        MemorySegment ensureA(int countFloats) {
            if (segA == null || capA < countFloats) {
                capA = Math.max(countFloats, 8192);
                segA = arena.allocate(ValueLayout.JAVA_FLOAT, capA);
            }
            return segA;
        }

        MemorySegment ensureC(int countFloats) {
            if (segC == null || capC < countFloats) {
                capC = Math.max(countFloats, 8192);
                segC = arena.allocate(ValueLayout.JAVA_FLOAT, capC);
            }
            return segC;
        }
    }

    public static boolean isAvailable() {
        return METAL_ENABLED;
    }

    /**
     * Executes matrix-vector multiplication via Metal GPU if supported tensor type and GPU is active.
     *
     * @return true if successfully computed by Metal, false if caller should fall back to CPU.
     */
    public static boolean matVec(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        if (!METAL_ENABLED || tensor == null) {
            return false;
        }

        int typeId = (int) tensor.typeId();
        MethodHandle handle;
        if (typeId == 12) { // Q4_K
            handle = MATVEC_Q4K_MH;
        } else if (typeId == 14) { // Q6_K
            handle = MATVEC_Q6K_MH;
        } else if (typeId == 8) { // Q8_0
            handle = MATVEC_Q8_0_MH;
        } else {
            return false;
        }

        if (handle == null) {
            return false;
        }

        int K = (int) GgufTensorOps.matrixColumns(tensor);
        int N = (int) GgufTensorOps.matrixRows(tensor);

        if (vector.length < K || output.length < N) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(K);
            MemorySegment segC = cache.ensureC(N);

            MemorySegment.copy(vector, 0, segA, ValueLayout.JAVA_FLOAT, 0, K);
            MemorySegment segB = GgufTensorOps.tensorData(model, tensor);

            int rc = (int) handle.invoke(METAL_BINDING_INSTANCE, segC, segA, segB, K, N);
            if (rc == 0) {
                MemorySegment.copy(segC, ValueLayout.JAVA_FLOAT, 0, output, 0, N);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVec failed, falling back to CPU", t);
            return false;
        }
    }

    private GgufMetalBridge() {}
}
