package tech.kayys.aljabr.gguf.javabackend;

import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Discovers and initializes the optional Metal binding without making the GGUF
 * bridge depend directly on the Metal backend module.
 *
 * <p>Optional operations are resolved independently so older or partially
 * featured bindings can still expose the operations they support.</p>
 */
final class GgufMetalBindingDiscovery {
    record Result(
            boolean enabled,
            Object instance,
            MethodHandle q4kHandle,
            MethodHandle q4kSoftcapHandle,
            MethodHandle q6kHandle,
            MethodHandle q6kSoftcapHandle,
            MethodHandle q8_0Handle,
            MethodHandle pairHandle,
            MethodHandle tripleHandle,
            MethodHandle batchHandle,
            MethodHandle tripleBatchHandle,
            MethodHandle ffnFusedHandle,
            MethodHandle ffnFusedBatchHandle,
            MethodHandle woFfnFusedHandle,
            MethodHandle woFfnFusedLayerBatchHandle,
            MethodHandle woFfnNextQkvFusedHandle,
            MethodHandle woFfnLogitsFusedHandle,
            MethodHandle registerModelBufferHandle,
            MethodHandle softcapHandle,
            MethodHandle fa4Handle,
            MethodHandle initKvCacheHandle,
            MethodHandle syncKvCacheHandle,
            MethodHandle registerNormBufferHandle,
            MethodHandle registerRopeFreqsHandle,
            MethodHandle registerDecodeLayerHandle,
            MethodHandle registerOutputHandle,
            MethodHandle registerTokenEmbdHandle,
            MethodHandle decodeStepHandle,
            MethodHandle decodeStepTokenHandle
    ) {}

    private GgufMetalBindingDiscovery() {}

    static Result discover(Logger log) {
        boolean enabled = false;
        Object instance = null;
        MethodHandle q4kHandle = null;
        MethodHandle q4kSoftcapHandle = null;
        MethodHandle q6kHandle = null;
        MethodHandle q6kSoftcapHandle = null;
        MethodHandle q8_0Handle = null;
        MethodHandle pairHandle = null;
        MethodHandle tripleHandle = null;
        MethodHandle batchHandle = null;
        MethodHandle tripleBatchHandle = null;
        MethodHandle ffnFusedHandle = null;
        MethodHandle ffnFusedBatchHandle = null;
        MethodHandle woFfnFusedHandle = null;
        MethodHandle woFfnFusedLayerBatchHandle = null;
        MethodHandle woFfnNextQkvFusedHandle = null;
        MethodHandle woFfnLogitsFusedHandle = null;
        MethodHandle registerModelBufferHandle = null;
        MethodHandle softcapHandle = null;
        MethodHandle fa4Handle = null;
        MethodHandle initKvCacheHandle = null;
        MethodHandle syncKvCacheHandle = null;
        MethodHandle registerNormBufferHandle = null;
        MethodHandle registerRopeFreqsHandle = null;
        MethodHandle registerDecodeLayerHandle = null;
        MethodHandle registerOutputHandle = null;
        MethodHandle registerTokenEmbdHandle = null;
        MethodHandle decodeStepHandle = null;
        MethodHandle decodeStepTokenHandle = null;

        boolean disableMetal = Boolean.getBoolean("aljabr.metal.disable") ||
                "false".equalsIgnoreCase(System.getProperty("aljabr.metal.enabled"));

        if (!disableMetal) {
            try {
                // MetalLibraryDiscovery is the single source of truth for native
                // library resolution. It prioritizes explicit overrides, local
                // build outputs, and ~/.aljabr/libs. Do not force discovery into
                // the legacy ~/.aqli location, which may contain stale binaries.

                Class<?> mbClass = Class.forName("tech.kayys.aljabr.metal.binding.MetalBinding");
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
                                    MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class))
                                    .bindTo(instance);

                            // matvecTransposedRightQ4KSoftcap(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N, float cap)
                            try {
                                q4kSoftcapHandle = lookup.findVirtual(mbClass, "matvecTransposedRightQ4KSoftcap",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class, float.class))
                                        .bindTo(instance);
                            } catch (Throwable t) {
                                System.err.println("[q4kSoftcapHandle lookup failed] " + t);
                            }

                            // matvecTransposedRightQ6K(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N)
                            try {
                                q6kHandle = lookup.findVirtual(mbClass, "matvecTransposedRightQ6K",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightQ6KSoftcap(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N, float cap)
                            try {
                                q6kSoftcapHandle = lookup.findVirtual(mbClass, "matvecTransposedRightQ6KSoftcap",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class, float.class))
                                        .bindTo(instance);
                            } catch (Throwable t) {
                                System.err.println("[q6kSoftcapHandle lookup failed] " + t);
                            }

                            // matvecTransposedRightQ8_0(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N)
                            try {
                                q8_0Handle = lookup.findVirtual(mbClass, "matvecTransposedRightQ8_0",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightPair(MemorySegment C0, MemorySegment C1, MemorySegment A, MemorySegment B0, MemorySegment B1, int K, int N0, int N1, int type0, int type1)
                            try {
                                pairHandle = lookup.findVirtual(mbClass, "matvecTransposedRightPair",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightTriple(MemorySegment C0, MemorySegment C1, MemorySegment C2, MemorySegment A, MemorySegment B0, MemorySegment B1, MemorySegment B2, int K, int N0, int N1, int N2, int type0, int type1, int type2)
                            try {
                                tripleHandle = lookup.findVirtual(mbClass, "matvecTransposedRightTriple",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightBatch(MemorySegment C, MemorySegment A, MemorySegment B, int K, int N, int type, int batchSize)
                            try {
                                batchHandle = lookup.findVirtual(mbClass, "matvecTransposedRightBatch",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // matvecTransposedRightTripleBatch(MemorySegment C0, MemorySegment C1, MemorySegment C2, MemorySegment A, MemorySegment B0, MemorySegment B1, MemorySegment B2, int K, int N0, int N1, int N2, int type0, int type1, int type2, int batchSize)
                            try {
                                tripleBatchHandle = lookup.findVirtual(mbClass, "matvecTransposedRightTripleBatch",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // ffnGatedFused(MemorySegment outDown, MemorySegment inA, MemorySegment gateW, MemorySegment upW, MemorySegment downW, int dim, int ffnDim, int typeGate, int typeUp, int typeDown, int activationKind)
                            try {
                                ffnFusedHandle = lookup.findVirtual(mbClass, "ffnGatedFused",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // ffnGatedFusedBatch(MemorySegment outDown, MemorySegment inA, MemorySegment gateW, MemorySegment upW, MemorySegment downW, int dim, int ffnDim, int typeGate, int typeUp, int typeDown, int activationKind, int batchSize)
                            try {
                                ffnFusedBatchHandle = lookup.findVirtual(mbClass, "ffnGatedFusedBatch",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // registerModelBuffer(MemorySegment basePtr, long totalBytes)
                            try {
                                registerModelBufferHandle = lookup.findVirtual(mbClass, "registerModelBuffer",
                                        MethodType.methodType(int.class, MemorySegment.class, long.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // woFfnFused(...)
                            try {
                                woFfnFusedHandle = lookup.findVirtual(mbClass, "woFfnFused",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                MemorySegment.class, int.class, int.class,
                                                MemorySegment.class, int.class,
                                                MemorySegment.class, int.class, int.class,
                                                float.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class,
                                                int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // woFfnFusedLayerBatch(...)
                            try {
                                woFfnFusedLayerBatchHandle = lookup.findVirtual(mbClass, "woFfnFusedLayerBatch",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, float.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // woFfnNextQkvFused(...)
                            try {
                                woFfnNextQkvFusedHandle = lookup.findVirtual(mbClass, "woFfnNextQkvFused",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                MemorySegment.class, int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, int.class,
                                                float.class,
                                                float.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class,
                                                int.class, int.class,
                                                int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class,
                                                int.class, int.class, int.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                            // woFfnLogitsFused(...)
                            try {
                                woFfnLogitsFusedHandle = lookup.findVirtual(mbClass, "woFfnLogitsFused",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                MemorySegment.class, int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, int.class,
                                                float.class,
                                                float.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class,
                                                int.class, int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                MemorySegment.class, float.class))
                                        .bindTo(instance);
                            } catch (NoSuchMethodException ignored) {
                            }

                             // softcap(MemorySegment logits, int N, float cap)
                             try {
                                 softcapHandle = lookup.findVirtual(mbClass, "softcap",
                                         MethodType.methodType(int.class,
                                                 MemorySegment.class, int.class, float.class))
                                         .bindTo(instance);
                             } catch (NoSuchMethodException ignored) {
                             }

                            // FlashAttention-4 GPU SDPA lookup
                            try {
                                Class<?> faClass = Class.forName("tech.kayys.aljabr.metal.binding.MetalFlashAttentionBinding");
                                MethodHandle faInit = lookup.findStatic(faClass, "initialize", MethodType.methodType(boolean.class));
                                faInit.invoke();
                                MethodHandle faGetInstance = lookup.findStatic(faClass, "getInstance", MethodType.methodType(faClass));
                                Object faInstance = faGetInstance.invoke();
                                fa4Handle = lookup.findVirtual(faClass, "fa4Attention",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, float.class, boolean.class, boolean.class))
                                        .bindTo(faInstance);
                                LOG.fine("[GgufMetalBridge] MetalFlashAttentionBinding bound successfully.");
                            } catch (Throwable t) {
                                LOG.fine("MetalFlashAttentionBinding lookup failed: " + t.getMessage());
                            }

                            // Decode step bindings
                            try {
                                initKvCacheHandle = lookup.findVirtual(mbClass, "initKVCache",
                                        MethodType.methodType(int.class, int.class, int.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] initKVCache handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] initKVCache bind FAILED: " + t); }

                            try {
                                syncKvCacheHandle = lookup.findVirtual(mbClass, "syncKVCache",
                                        MethodType.methodType(int.class, int.class, int.class, MemorySegment.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] syncKVCache handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] syncKVCache bind FAILED: " + t); }

                            try {
                                registerNormBufferHandle = lookup.findVirtual(mbClass, "registerNormBuffer",
                                        MethodType.methodType(int.class, int.class, MemorySegment.class, long.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] registerNormBuffer handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerNormBuffer bind FAILED: " + t); }

                            try {
                                registerRopeFreqsHandle = lookup.findVirtual(mbClass, "registerRopeFreqs",
                                        MethodType.methodType(int.class, int.class, MemorySegment.class, int.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] registerRopeFreqs handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerRopeFreqs bind FAILED: " + t); }

                            try {
                                registerDecodeLayerHandle = lookup.findVirtual(mbClass, "registerDecodeLayer",
                                        MethodType.methodType(int.class,
                                                int.class, int.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, int.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class,
                                                int.class, int.class, int.class, int.class,
                                                float.class, int.class,
                                                float.class, float.class,
                                                int.class, int.class,
                                                int.class, int.class,
                                                int.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] registerDecodeLayer handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerDecodeLayer bind FAILED: " + t); }

                            try {
                                registerOutputHandle = lookup.findVirtual(mbClass, "registerOutput",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                float.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] registerOutput handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerOutput bind FAILED: " + t); }

                            try {
                                decodeStepHandle = lookup.findVirtual(mbClass, "decodeStep",
                                        MethodType.methodType(int.class,
                                                int.class, int.class, float.class,
                                                MemorySegment.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] decodeStep handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] decodeStep bind FAILED: " + t); }

                            try {
                                registerTokenEmbdHandle = lookup.findVirtual(mbClass, "registerTokenEmbd",
                                        MethodType.methodType(int.class, MemorySegment.class, int.class, int.class, float.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] registerTokenEmbd handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerTokenEmbd bind FAILED: " + t); }

                            try {
                                decodeStepTokenHandle = lookup.findVirtual(mbClass, "decodeStepToken",
                                        MethodType.methodType(int.class, int.class, int.class, float.class, int.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.fine("[GgufMetalBridge] decodeStepToken handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] decodeStepToken bind FAILED: " + t); }

                            enabled = true;
                            LOG.fine("Aljabr Metal GPU acceleration enabled for Java GGUF backend.");
                        }
                    }
                }
            } catch (ClassNotFoundException e) {
                LOG.fine("MetalBinding not on classpath, continuing with CPU vector execution.");
            } catch (Throwable t) {
                LOG.log(Level.WARNING, "Failed to initialize Aljabr Metal GPU acceleration: " + t.getMessage(), t);
            }
        }


        return new Result(enabled, instance, q4kHandle, q4kSoftcapHandle, q6kHandle,
                q6kSoftcapHandle, q8_0Handle, pairHandle, tripleHandle, batchHandle,
                tripleBatchHandle, ffnFusedHandle, ffnFusedBatchHandle, woFfnFusedHandle,
                woFfnFusedLayerBatchHandle, woFfnNextQkvFusedHandle, woFfnLogitsFusedHandle,
                registerModelBufferHandle, softcapHandle, fa4Handle, initKvCacheHandle,
                syncKvCacheHandle, registerNormBufferHandle, registerRopeFreqsHandle,
                registerDecodeLayerHandle, registerOutputHandle, registerTokenEmbdHandle,
                decodeStepHandle, decodeStepTokenHandle);
    }
}
