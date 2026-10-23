package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Bridge between Java GGUF backend and Aljabr Metal GPU acceleration.
 *
 * <p>Uses reflection / MethodHandle to load {@code MetalBinding} dynamically if present on the classpath,
 * preventing tight coupling or circular dependencies between {@code aljabr-gguf-java} and
 * {@code aljabr-backend-metal}.</p>
 */
public final class GgufMetalBridge {
    private static final Logger LOG = Logger.getLogger(GgufMetalBridge.class.getName());

    private static final boolean METAL_ENABLED;
    private static final Object METAL_BINDING_INSTANCE;
    private static final MethodHandle MATVEC_Q4K_MH;
    private static final MethodHandle MATVEC_Q4K_SOFTCAP_MH;
    private static final MethodHandle MATVEC_Q6K_MH;
    private static final MethodHandle MATVEC_Q6K_SOFTCAP_MH;
    private static final MethodHandle MATVEC_Q8_0_MH;
    private static final MethodHandle MATVEC_PAIR_MH;
    private static final MethodHandle MATVEC_TRIPLE_MH;
    private static final MethodHandle MATVEC_BATCH_MH;
    private static final MethodHandle MATVEC_TRIPLE_BATCH_MH;
    private static final MethodHandle FFN_GATED_FUSED_MH;
    private static final MethodHandle FFN_GATED_FUSED_BATCH_MH;
    private static final MethodHandle WO_FFN_FUSED_MH;
    private static final MethodHandle WO_FFN_NEXT_QKV_FUSED_MH;
    private static final MethodHandle WO_FFN_LOGITS_FUSED_MH;
    private static final MethodHandle REGISTER_MODEL_BUFFER_MH;
    /** GPU logit soft-capping: int softcap(MemorySegment logits, int N, float cap) */
    private static final MethodHandle SOFTCAP_MH;
    /** FlashAttention-4 GPU SDPA: int fa4Attention(out, q, k, v, B, T, S, H, H_kv, D, scale, isCausal, useBf16) */
    private static final MethodHandle FA4_ATTENTION_MH;

    // Full-GPU single command buffer decode step handles
    private static final MethodHandle INIT_KV_CACHE_MH;
    private static final MethodHandle SYNC_KV_CACHE_MH;
    private static final MethodHandle REGISTER_NORM_BUFFER_MH;
    private static final MethodHandle REGISTER_DECODE_LAYER_MH;
    private static final MethodHandle REGISTER_OUTPUT_MH;
    private static final MethodHandle DECODE_STEP_MH;

    // Thread-local or reusable buffers for inputs/outputs to prevent allocations in inner loop
    private static final ThreadLocal<MetalBufferCache> BUFFERS = ThreadLocal.withInitial(MetalBufferCache::new);

    // Pre-loaded static norm buffers: layout [normSlot][layer * dim]
    // normSlot: 0=postAttn, 1=ffnNorm, 2=postFfn, 3=nextAttnNorm (= attnNorm of layer+1)
    // Populated once at model init via registerLayerNorms(). Avoids per-token MemorySegment.copy() for
    // layer-static norm weights (saves ~192 memcpy calls per decode step).
    private static volatile Arena g_normArena = null;
    // g_normSeg[slot] = contiguous off-heap segment for that slot; layer L is at offset L*dim*FLOAT_BYTES
    private static volatile MemorySegment[] g_normSegs = null;
    private static volatile int g_normDim = 0;
    private static volatile int g_normLayers = 0;

    static {
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
        MethodHandle woFfnNextQkvFusedHandle = null;
        MethodHandle woFfnLogitsFusedHandle = null;
        MethodHandle registerModelBufferHandle = null;
        MethodHandle softcapHandle = null;
        MethodHandle fa4Handle = null;
        MethodHandle initKvCacheHandle = null;
        MethodHandle syncKvCacheHandle = null;
        MethodHandle registerNormBufferHandle = null;
        MethodHandle registerDecodeLayerHandle = null;
        MethodHandle registerOutputHandle = null;
        MethodHandle decodeStepHandle = null;

        boolean disableMetal = Boolean.getBoolean("aljabr.metal.disable") ||
                "false".equalsIgnoreCase(System.getProperty("aljabr.metal.enabled"));

        if (!disableMetal) {
            try {
                // Ensure discovery checks for libaljabr_metal.dylib
                String override = System.getProperty("aljabr.metal.dylib");
                if (override == null || override.contains("libaqli_metal.dylib")) {
                    String home = System.getProperty("user.home");
                    if (home != null) {
                        java.nio.file.Path p = java.nio.file.Path.of(home, ".aqli", "libs", "libaljabr_metal.dylib");
                        if (java.nio.file.Files.exists(p)) {
                            System.setProperty("aljabr.metal.dylib", p.toString());
                        }
                    }
                }

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
                                Class<?> faClass = Class.forName("tech.kayys.aljabr.backend.metal.binding.MetalFlashAttentionBinding");
                                MethodHandle faGetInstance = lookup.findStatic(faClass, "getInstance", MethodType.methodType(faClass));
                                Object faInstance = faGetInstance.invoke();
                                fa4Handle = lookup.findVirtual(faClass, "fa4Attention",
                                        MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, float.class, boolean.class, boolean.class))
                                        .bindTo(faInstance);
                                LOG.info("[GgufMetalBridge] MetalFlashAttentionBinding bound successfully.");
                            } catch (Throwable t) {
                                LOG.fine("MetalFlashAttentionBinding lookup failed: " + t.getMessage());
                            }

                            // Decode step bindings
                            try {
                                initKvCacheHandle = lookup.findVirtual(mbClass, "initKVCache",
                                        MethodType.methodType(int.class, int.class, int.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] initKVCache handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] initKVCache bind FAILED: " + t); }

                            try {
                                syncKvCacheHandle = lookup.findVirtual(mbClass, "syncKVCache",
                                        MethodType.methodType(int.class, int.class, int.class, MemorySegment.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] syncKVCache handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] syncKVCache bind FAILED: " + t); }

                            try {
                                registerNormBufferHandle = lookup.findVirtual(mbClass, "registerNormBuffer",
                                        MethodType.methodType(int.class, int.class, MemorySegment.class, long.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] registerNormBuffer handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerNormBuffer bind FAILED: " + t); }

                            try {
                                registerDecodeLayerHandle = lookup.findVirtual(mbClass, "registerDecodeLayer",
                                        MethodType.methodType(int.class,
                                                int.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                MemorySegment.class, MemorySegment.class, MemorySegment.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class, int.class,
                                                int.class, int.class, int.class, int.class, int.class, int.class,
                                                int.class, int.class, int.class, int.class,
                                                float.class, int.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] registerDecodeLayer handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerDecodeLayer bind FAILED: " + t); }

                            try {
                                registerOutputHandle = lookup.findVirtual(mbClass, "registerOutput",
                                        MethodType.methodType(int.class,
                                                MemorySegment.class,
                                                MemorySegment.class, int.class, int.class, int.class,
                                                float.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] registerOutput handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] registerOutput bind FAILED: " + t); }

                            try {
                                decodeStepHandle = lookup.findVirtual(mbClass, "decodeStep",
                                        MethodType.methodType(int.class,
                                                int.class, int.class, float.class,
                                                MemorySegment.class, MemorySegment.class))
                                        .bindTo(instance);
                                LOG.info("[GgufMetalBridge] decodeStep handle bound.");
                            } catch (Throwable t) { LOG.warning("[GgufMetalBridge] decodeStep bind FAILED: " + t); }

                            enabled = true;
                            LOG.info("Aljabr Metal GPU acceleration enabled for Java GGUF backend.");
                        }
                    }
                }
            } catch (ClassNotFoundException e) {
                LOG.fine("MetalBinding not on classpath, continuing with CPU vector execution.");
            } catch (Throwable t) {
                LOG.log(Level.WARNING, "Failed to initialize Aljabr Metal GPU acceleration: " + t.getMessage(), t);
            }
        }

        METAL_ENABLED = enabled;
        METAL_BINDING_INSTANCE = instance;
        MATVEC_Q4K_MH = q4kHandle;
        MATVEC_Q4K_SOFTCAP_MH = q4kSoftcapHandle;
        MATVEC_Q6K_MH = q6kHandle;
        MATVEC_Q6K_SOFTCAP_MH = q6kSoftcapHandle;
        MATVEC_Q8_0_MH = q8_0Handle;
        MATVEC_PAIR_MH = pairHandle;
        MATVEC_TRIPLE_MH = tripleHandle;
        MATVEC_BATCH_MH = batchHandle;
        MATVEC_TRIPLE_BATCH_MH = tripleBatchHandle;
        FFN_GATED_FUSED_MH = ffnFusedHandle;
        FFN_GATED_FUSED_BATCH_MH = ffnFusedBatchHandle;
        WO_FFN_FUSED_MH = woFfnFusedHandle;
        WO_FFN_NEXT_QKV_FUSED_MH = woFfnNextQkvFusedHandle;
        WO_FFN_LOGITS_FUSED_MH = woFfnLogitsFusedHandle;
        REGISTER_MODEL_BUFFER_MH = registerModelBufferHandle;
        SOFTCAP_MH = softcapHandle;
        FA4_ATTENTION_MH = fa4Handle;
        INIT_KV_CACHE_MH = initKvCacheHandle;
        SYNC_KV_CACHE_MH = syncKvCacheHandle;
        REGISTER_NORM_BUFFER_MH = registerNormBufferHandle;
        REGISTER_DECODE_LAYER_MH = registerDecodeLayerHandle;
        REGISTER_OUTPUT_MH = registerOutputHandle;
        DECODE_STEP_MH = decodeStepHandle;
    }

    private static final class MetalBufferCache {
        final Arena arena = Arena.ofShared();
        MemorySegment segA;
        int capA;
        MemorySegment segC0;
        int capC0;
        MemorySegment segC1;
        int capC1;
        MemorySegment segC2;
        int capC2;

        MemorySegment ensureA(int countFloats) {
            if (segA == null || capA < countFloats) {
                capA = Math.max(countFloats, 8192);
                segA = arena.allocate(ValueLayout.JAVA_FLOAT, capA);
            }
            return segA;
        }

        MemorySegment ensureC0(int countFloats) {
            if (segC0 == null || capC0 < countFloats) {
                capC0 = Math.max(countFloats, 8192);
                segC0 = arena.allocate(ValueLayout.JAVA_FLOAT, capC0);
            }
            return segC0;
        }

        MemorySegment ensureC1(int countFloats) {
            if (segC1 == null || capC1 < countFloats) {
                capC1 = Math.max(countFloats, 8192);
                segC1 = arena.allocate(ValueLayout.JAVA_FLOAT, capC1);
            }
            return segC1;
        }

        MemorySegment ensureC2(int countFloats) {
            if (segC2 == null || capC2 < countFloats) {
                capC2 = Math.max(countFloats, 8192);
                segC2 = arena.allocate(ValueLayout.JAVA_FLOAT, capC2);
            }
            return segC2;
        }

        MemorySegment segX;
        int capX;
        MemorySegment segAttn;
        int capAttn;
        MemorySegment segNorm1;
        int capNorm1;
        MemorySegment segNorm2;
        int capNorm2;
        MemorySegment segNorm3;
        int capNorm3;

        MemorySegment ensureX(int countFloats) {
            if (segX == null || capX < countFloats) {
                capX = Math.max(countFloats, 8192);
                segX = arena.allocate(ValueLayout.JAVA_FLOAT, capX);
            }
            return segX;
        }

        MemorySegment ensureAttn(int countFloats) {
            if (segAttn == null || capAttn < countFloats) {
                capAttn = Math.max(countFloats, 8192);
                segAttn = arena.allocate(ValueLayout.JAVA_FLOAT, capAttn);
            }
            return segAttn;
        }

        MemorySegment ensureNorm1(int countFloats) {
            if (segNorm1 == null || capNorm1 < countFloats) {
                capNorm1 = Math.max(countFloats, 8192);
                segNorm1 = arena.allocate(ValueLayout.JAVA_FLOAT, capNorm1);
            }
            return segNorm1;
        }

        MemorySegment ensureNorm2(int countFloats) {
            if (segNorm2 == null || capNorm2 < countFloats) {
                capNorm2 = Math.max(countFloats, 8192);
                segNorm2 = arena.allocate(ValueLayout.JAVA_FLOAT, capNorm2);
            }
            return segNorm2;
        }

        MemorySegment ensureNorm3(int countFloats) {
            if (segNorm3 == null || capNorm3 < countFloats) {
                capNorm3 = Math.max(countFloats, 8192);
                segNorm3 = arena.allocate(ValueLayout.JAVA_FLOAT, capNorm3);
            }
            return segNorm3;
        }

        MemorySegment segOutputNorm;
        int capOutputNorm;
        MemorySegment segLogits;
        int capLogits;

        MemorySegment ensureOutputNorm(int countFloats) {
            if (segOutputNorm == null || capOutputNorm < countFloats) {
                capOutputNorm = Math.max(countFloats, 8192);
                segOutputNorm = arena.allocate(ValueLayout.JAVA_FLOAT, capOutputNorm);
            }
            return segOutputNorm;
        }

        MemorySegment ensureLogits(int countFloats) {
            if (segLogits == null || capLogits < countFloats) {
                capLogits = Math.max(countFloats, 262144);
                segLogits = arena.allocate(ValueLayout.JAVA_FLOAT, capLogits);
            }
            return segLogits;
        }
    }

    public static boolean isAvailable() {
        return METAL_ENABLED;
    }

    private static final java.util.Set<GGUFModel> REGISTERED_MODELS =
            java.util.Collections.newSetFromMap(new java.util.WeakHashMap<>());

    public static synchronized void ensureModelRegistered(GGUFModel model) {
        if (model != null && REGISTER_MODEL_BUFFER_MH != null && REGISTERED_MODELS.add(model)) {
            try {
                MemorySegment seg = model.segment();
                int rc = (int) REGISTER_MODEL_BUFFER_MH.invokeExact(seg, seg.byteSize());
                if (rc == 0) {
                    LOG.info("Registered master Metal model buffer: " + (seg.byteSize() / (1024 * 1024)) + " MB");
                }
            } catch (Throwable t) {
                LOG.log(Level.WARNING, "Failed to register master model buffer", t);
            }
        }
    }

    private static boolean isTypeSupported(int typeId) {
        return typeId == 12 || typeId == 14 || typeId == 8 || typeId == 1 || typeId == 30 || typeId == 16;
    }

    /**
     * Pre-uploads all layer norm arrays into a single persistent off-heap {@link MemorySegment}.
     *
     * <p>Call once at model init time from {@code GgufDecodeStep} constructor. After this, per-token
     * decode paths use {@link #normSlice(int, int)} to get a zero-copy view into the pre-loaded data,
     * eliminating ~192 {@code MemorySegment.copy()} calls per decode step (~0.5 ms savings).</p>
     *
     * @param layers      number of model layers (e.g. 48 for Gemma 4 12B)
     * @param dim         model embedding dimension (e.g. 3840 for Gemma 4 12B)
     * @param postAttnN   postAttnNorm[layer] — may contain nulls; null → slot filled with zeros
     * @param ffnNormN    ffnNorm[layer]       — required, must be non-null for all layers
     * @param postFfnN    postFfnNorm[layer]   — may contain nulls
     * @param attnNormN   attnNorm[layer]      — the next-QKV norm (slot 3 stores layer L+1's attnNorm)
     */
    public static synchronized void registerLayerNorms(
            int layers, int dim,
            float[][] postAttnN, float[][] ffnNormN,
            float[][] postFfnN, float[][] attnNormN) {
        if (!METAL_ENABLED) return;
        // 4 slots: 0=postAttn, 1=ffnNorm, 2=postFfn, 3=attnNorm
        Arena arena = Arena.ofShared();
        MemorySegment[] segs = new MemorySegment[4];
        for (int slot = 0; slot < 4; slot++) {
            segs[slot] = arena.allocate((long) layers * dim * Float.BYTES);
        }
        float[][][] sources = { postAttnN, ffnNormN, postFfnN, attnNormN };
        for (int slot = 0; slot < 4; slot++) {
            float[][] src = sources[slot];
            for (int layer = 0; layer < layers; layer++) {
                float[] arr = (src != null && layer < src.length) ? src[layer] : null;
                if (arr != null && arr.length >= dim) {
                    MemorySegment.copy(arr, 0, segs[slot], ValueLayout.JAVA_FLOAT, (long) layer * dim * Float.BYTES, dim);
                }
                // null / missing → stays as zero-initialized memory (safe; native will skip if hasXxx=0)
            }
        }
        // Publish atomically (volatile write)
        g_normArena = arena;
        g_normSegs = segs;
        g_normDim = dim;
        g_normLayers = layers;

        if (REGISTER_NORM_BUFFER_MH != null) {
            for (int slot = 0; slot < 4; slot++) {
                try {
                    int rc = (int) REGISTER_NORM_BUFFER_MH.invokeExact(slot, segs[slot], (long) layers * dim * Float.BYTES);
                    if (rc != 0) {
                        LOG.warning("[GgufMetalBridge] registerNormBuffer failed for slot " + slot + ", rc=" + rc);
                    }
                } catch (Throwable t) {
                    LOG.log(Level.WARNING, "Failed to register norm buffer slot " + slot + ": " + t.getMessage(), t);
                }
            }
        }

        LOG.info("[GgufMetalBridge] Pre-loaded norm buffers: " + layers + " layers × " + dim + " floats × 4 slots = "
                + (layers * dim * 4 * Float.BYTES / 1024) + " KB");
    }

    /**
     * Returns a slice view into the pre-loaded norm buffer for {@code (slot, layerIdx)}.
     * Returns {@link MemorySegment#NULL} if norms are not registered or the source array was null.
     *
     * @param slot     0=postAttn, 1=ffnNorm, 2=postFfn, 3=attnNorm
     * @param layerIdx layer index
     */
    private static MemorySegment normSlice(int slot, int layerIdx) {
        MemorySegment[] segs = g_normSegs;
        if (segs == null || slot < 0 || slot >= 4 || layerIdx < 0 || layerIdx >= g_normLayers) {
            return MemorySegment.NULL;
        }
        long byteOffset = (long) layerIdx * g_normDim * Float.BYTES;
        return segs[slot].asSlice(byteOffset, (long) g_normDim * Float.BYTES);
    }

    /**
     * Executes matrix-vector multiplication via Metal GPU if supported tensor type and GPU is active.
     */
    public static boolean matVec(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        if (!METAL_ENABLED || tensor == null) {
            return false;
        }
        ensureModelRegistered(model);

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
            MemorySegment segC = cache.ensureC0(N);

            MemorySegment.copy(vector, 0, segA, ValueLayout.JAVA_FLOAT, 0, K);
            MemorySegment segB = GgufTensorOps.tensorData(model, tensor);

            int rc = (int) handle.invokeExact(segC, segA, segB, K, N);
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

    /**
     * Executes matrix-vector multiplication with in-kernel tanh soft-capping in a SINGLE GPU dispatch.
     *
     * @param model GGUF model
     * @param tensor weight tensor (typically outputWeight / Q6_K)
     * @param vector input activation vector (typically normalized x, length K)
     * @param output output logits array (length N)
     * @param cap soft-cap threshold (e.g. 30.0 for Gemma)
     * @return true if executed on GPU successfully, false to fall back
     */
    public static boolean matVecSoftcap(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output, float cap) {
        if (!METAL_ENABLED || tensor == null || cap <= 0.0f) {
            return false;
        }
        int typeId = (int) tensor.typeId();
        MethodHandle handle;
        if (typeId == 12) {
            handle = MATVEC_Q4K_SOFTCAP_MH;
        } else if (typeId == 14) {
            handle = MATVEC_Q6K_SOFTCAP_MH;
        } else {
            return false;
        }
        if (handle == null) {
            return false;
        }
        ensureModelRegistered(model);

        int K = (int) GgufTensorOps.matrixColumns(tensor);
        int N = (int) GgufTensorOps.matrixRows(tensor);

        if (vector.length < K || output.length < N) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(K);
            MemorySegment segC = cache.ensureC0(N);

            MemorySegment.copy(vector, 0, segA, ValueLayout.JAVA_FLOAT, 0, K);
            MemorySegment segB = GgufTensorOps.tensorData(model, tensor);

            int rc = (int) handle.invokeExact(segC, segA, segB, K, N, cap);
            if (rc == 0) {
                MemorySegment.copy(segC, ValueLayout.JAVA_FLOAT, 0, output, 0, N);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVecSoftcap failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes FlashAttention-4 fused causal scaled dot-product attention on the GPU via MPSGraph SDPA.
     *
     * @param output   destination [B, T, H, D]
     * @param query    query tensor [B, T, H, D]
     * @param key      key tensor [B, S, H_kv, D]
     * @param value    value tensor [B, S, H_kv, D]
     * @param B        batch size (1 for single-sequence prefill)
     * @param T        query sequence length
     * @param S        key/value sequence length
     * @param H        query head count (e.g. 16 for Gemma-4 12B)
     * @param H_kv     key/value head count (e.g. 8 for Gemma-4 12B GQA)
     * @param D        head dimension (e.g. 256 for Gemma-4 12B)
     * @param scale    attention scale (typically 1.0f or 1/sqrt(D))
     * @param isCausal true for causal autoregressive attention mask
     * @return true if successfully executed on Metal GPU, false to fall back to CPU loop
     */
    public static boolean fa4Attention(
            float[] output,
            float[] query,
            float[] key,
            float[] value,
            int B, int T, int S, int H, int H_kv, int D,
            float scale, boolean isCausal) {
        if (!METAL_ENABLED || FA4_ATTENTION_MH == null ||
                output == null || query == null || key == null || value == null) {
            return false;
        }
        int totalQ = B * T * H * D;
        int totalKV = B * S * H_kv * D;
        if (query.length < totalQ || output.length < totalQ || key.length < totalKV || value.length < totalKV) {
            return false;
        }
        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segOut = cache.ensureC0(totalQ);
            MemorySegment segQ = cache.ensureA(totalQ);
            MemorySegment segK = cache.ensureC1(totalKV);
            MemorySegment segV = cache.ensureC2(totalKV);

            MemorySegment.copy(query, 0, segQ, ValueLayout.JAVA_FLOAT, 0, totalQ);
            MemorySegment.copy(key, 0, segK, ValueLayout.JAVA_FLOAT, 0, totalKV);
            MemorySegment.copy(value, 0, segV, ValueLayout.JAVA_FLOAT, 0, totalKV);

            int rc = (int) FA4_ATTENTION_MH.invokeExact(
                    segOut, segQ, segK, segV,
                    B, T, S, H, H_kv, D,
                    scale, isCausal, false);

            if (rc == 0) {
                MemorySegment.copy(segOut, ValueLayout.JAVA_FLOAT, 0, output, 0, totalQ);
                return true;
            }
            return false;
        } catch (Throwable t) {
            LOG.log(Level.FINE, "fa4Attention execution failed", t);
            return false;
        }
    }

    /**
     * Executes two matrix-vector multiplications sharing the same input vector in a SINGLE command buffer.
     */
    public static boolean matVecPair(GGUFModel model,
                                     GGUFTensorInfo t0, float[] out0,
                                     GGUFTensorInfo t1, float[] out1,
                                     float[] vector) {
        if (!METAL_ENABLED || MATVEC_PAIR_MH == null || t0 == null || t1 == null) {
            return false;
        }
        ensureModelRegistered(model);

        int type0 = (int) t0.typeId();
        int type1 = (int) t1.typeId();
        if (!isTypeSupported(type0) || !isTypeSupported(type1)) {
            return false;
        }

        int K0 = (int) GgufTensorOps.matrixColumns(t0);
        int K1 = (int) GgufTensorOps.matrixColumns(t1);
        if (K0 != K1 || vector.length < K0) {
            return false;
        }
        int K = K0;

        int N0 = (int) GgufTensorOps.matrixRows(t0);
        int N1 = (int) GgufTensorOps.matrixRows(t1);
        if (out0.length < N0 || out1.length < N1) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(K);
            MemorySegment segC0 = cache.ensureC0(N0);
            MemorySegment segC1 = cache.ensureC1(N1);

            MemorySegment.copy(vector, 0, segA, ValueLayout.JAVA_FLOAT, 0, K);
            MemorySegment segB0 = GgufTensorOps.tensorData(model, t0);
            MemorySegment segB1 = GgufTensorOps.tensorData(model, t1);

            int rc = (int) MATVEC_PAIR_MH.invokeExact(segC0, segC1, segA, segB0, segB1, K, N0, N1, type0, type1);
            if (rc == 0) {
                MemorySegment.copy(segC0, ValueLayout.JAVA_FLOAT, 0, out0, 0, N0);
                MemorySegment.copy(segC1, ValueLayout.JAVA_FLOAT, 0, out1, 0, N1);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVecPair failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes three matrix-vector multiplications sharing the same input vector in a SINGLE command buffer.
     */
    public static boolean matVecTriple(GGUFModel model,
                                       GGUFTensorInfo t0, float[] out0,
                                       GGUFTensorInfo t1, float[] out1,
                                       GGUFTensorInfo t2, float[] out2,
                                       float[] vector) {
        if (!METAL_ENABLED || MATVEC_TRIPLE_MH == null || t0 == null || t1 == null || t2 == null) {
            return false;
        }
        ensureModelRegistered(model);

        int type0 = (int) t0.typeId();
        int type1 = (int) t1.typeId();
        int type2 = (int) t2.typeId();
        if (!isTypeSupported(type0) || !isTypeSupported(type1) || !isTypeSupported(type2)) {
            return false;
        }

        int K0 = (int) GgufTensorOps.matrixColumns(t0);
        int K1 = (int) GgufTensorOps.matrixColumns(t1);
        int K2 = (int) GgufTensorOps.matrixColumns(t2);
        if (K0 != K1 || K0 != K2 || vector.length < K0) {
            return false;
        }
        int K = K0;

        int N0 = (int) GgufTensorOps.matrixRows(t0);
        int N1 = (int) GgufTensorOps.matrixRows(t1);
        int N2 = (int) GgufTensorOps.matrixRows(t2);
        if (out0.length < N0 || out1.length < N1 || out2.length < N2) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(K);
            MemorySegment segC0 = cache.ensureC0(N0);
            MemorySegment segC1 = cache.ensureC1(N1);
            MemorySegment segC2 = cache.ensureC2(N2);

            MemorySegment.copy(vector, 0, segA, ValueLayout.JAVA_FLOAT, 0, K);
            MemorySegment segB0 = GgufTensorOps.tensorData(model, t0);
            MemorySegment segB1 = GgufTensorOps.tensorData(model, t1);
            MemorySegment segB2 = GgufTensorOps.tensorData(model, t2);

            int rc = (int) MATVEC_TRIPLE_MH.invokeExact(segC0, segC1, segC2, segA, segB0, segB1, segB2, K, N0, N1, N2, type0, type1, type2);
            if (rc == 0) {
                MemorySegment.copy(segC0, ValueLayout.JAVA_FLOAT, 0, out0, 0, N0);
                MemorySegment.copy(segC1, ValueLayout.JAVA_FLOAT, 0, out1, 0, N1);
                MemorySegment.copy(segC2, ValueLayout.JAVA_FLOAT, 0, out2, 0, N2);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVecTriple failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes fused FFN (Gate + Up + GeGLU/SwiGLU + Down) entirely on the GPU in a single submission.
     *
     * @param model GGUF model
     * @param ffnGate gate weight tensor
     * @param ffnUp up weight tensor
     * @param ffnDown down weight tensor
     * @param inA normalized input vector x (length dim)
     * @param outDown output destination vector (length dim)
     * @param isGeGLU true for GeGLU (Gemma), false for SwiGLU (Llama)
     * @return true if successfully computed on Metal, false to fall back to CPU
     */
    public static boolean ffnGatedFused(GGUFModel model,
                                        GGUFTensorInfo ffnGate,
                                        GGUFTensorInfo ffnUp,
                                        GGUFTensorInfo ffnDown,
                                        float[] inA,
                                        float[] outDown,
                                        boolean isGeGLU) {
        if (!METAL_ENABLED || FFN_GATED_FUSED_MH == null ||
            ffnGate == null || ffnUp == null || ffnDown == null || inA == null || outDown == null) {
            return false;
        }
        ensureModelRegistered(model);

        int typeGate = (int) ffnGate.typeId();
        int typeUp = (int) ffnUp.typeId();
        int typeDown = (int) ffnDown.typeId();
        if (!isTypeSupported(typeGate) || !isTypeSupported(typeUp) || !isTypeSupported(typeDown)) {
            System.err.println("[FFN_FUSED_FAIL] unsupported types: gate=" + typeGate + " up=" + typeUp + " down=" + typeDown);
            return false;
        }

        int dim = (int) GgufTensorOps.matrixColumns(ffnGate);
        int ffnDim = (int) GgufTensorOps.matrixRows(ffnGate);
        if (inA.length < dim || outDown.length < dim) {
            System.err.println("[FFN_FUSED_FAIL] length mismatch: dim=" + dim + " inA=" + inA.length + " outDown=" + outDown.length);
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(dim);
            MemorySegment segOut = cache.ensureC0(dim);

            MemorySegment.copy(inA, 0, segA, ValueLayout.JAVA_FLOAT, 0, dim);

            MemorySegment segGateW = GgufTensorOps.tensorData(model, ffnGate);
            MemorySegment segUpW = GgufTensorOps.tensorData(model, ffnUp);
            MemorySegment segDownW = GgufTensorOps.tensorData(model, ffnDown);

            int activationKind = isGeGLU ? 2 : 1; // 2=GeGLU, 1=SwiGLU

            int rc = (int) FFN_GATED_FUSED_MH.invokeExact(
                    segOut, segA, segGateW, segUpW, segDownW,
                    dim, ffnDim, typeGate, typeUp, typeDown, activationKind);

            if (rc == 0) {
                MemorySegment.copy(segOut, ValueLayout.JAVA_FLOAT, 0, outDown, 0, dim);
                return true;
            } else {
                System.err.println("[FFN_FUSED_FAIL] native rc=" + rc);
                return false;
            }
        } catch (Throwable t) {
            System.err.println("[FFN_FUSED_FAIL] exception: " + t);
            LOG.log(Level.FINE, "Metal ffnGatedFused failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes matrix-vector multiplication for a batch of tokens in a single command buffer.
     */
    public static boolean matVecBatch(GGUFModel model, GGUFTensorInfo tensor, float[] vectorBatch, float[] outputBatch, int batchSize) {
        if (!METAL_ENABLED || MATVEC_BATCH_MH == null || tensor == null || batchSize <= 0) {
            return false;
        }
        ensureModelRegistered(model);

        int typeId = (int) tensor.typeId();
        if (!isTypeSupported(typeId)) {
            return false;
        }

        int K = (int) GgufTensorOps.matrixColumns(tensor);
        int N = (int) GgufTensorOps.matrixRows(tensor);

        int totalA = batchSize * K;
        int totalC = batchSize * N;
        if (vectorBatch.length < totalA || outputBatch.length < totalC) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(totalA);
            MemorySegment segC = cache.ensureC0(totalC);

            MemorySegment.copy(vectorBatch, 0, segA, ValueLayout.JAVA_FLOAT, 0, totalA);
            MemorySegment segB = GgufTensorOps.tensorData(model, tensor);

            int rc = (int) MATVEC_BATCH_MH.invokeExact(segC, segA, segB, K, N, typeId, batchSize);
            if (rc == 0) {
                MemorySegment.copy(segC, ValueLayout.JAVA_FLOAT, 0, outputBatch, 0, totalC);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVecBatch failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes three matrix-vector multiplications sharing the same batch of input vectors in a single command buffer.
     */
    public static boolean matVecTripleBatch(GGUFModel model,
                                            GGUFTensorInfo t0, float[] out0,
                                            GGUFTensorInfo t1, float[] out1,
                                            GGUFTensorInfo t2, float[] out2,
                                            float[] vectorBatch, int batchSize) {
        if (!METAL_ENABLED || MATVEC_TRIPLE_BATCH_MH == null || t0 == null || t1 == null || t2 == null || batchSize <= 0) {
            return false;
        }
        ensureModelRegistered(model);

        int type0 = (int) t0.typeId();
        int type1 = (int) t1.typeId();
        int type2 = (int) t2.typeId();
        if (!isTypeSupported(type0) || !isTypeSupported(type1) || !isTypeSupported(type2)) {
            return false;
        }

        int K0 = (int) GgufTensorOps.matrixColumns(t0);
        int K1 = (int) GgufTensorOps.matrixColumns(t1);
        int K2 = (int) GgufTensorOps.matrixColumns(t2);
        if (K0 != K1 || K0 != K2 || vectorBatch.length < batchSize * K0) {
            return false;
        }
        int K = K0;

        int N0 = (int) GgufTensorOps.matrixRows(t0);
        int N1 = (int) GgufTensorOps.matrixRows(t1);
        int N2 = (int) GgufTensorOps.matrixRows(t2);
        int totalA = batchSize * K;
        int totalC0 = batchSize * N0;
        int totalC1 = batchSize * N1;
        int totalC2 = batchSize * N2;
        if (out0.length < totalC0 || out1.length < totalC1 || out2.length < totalC2) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(totalA);
            MemorySegment segC0 = cache.ensureC0(totalC0);
            MemorySegment segC1 = cache.ensureC1(totalC1);
            MemorySegment segC2 = cache.ensureC2(totalC2);

            MemorySegment.copy(vectorBatch, 0, segA, ValueLayout.JAVA_FLOAT, 0, totalA);
            MemorySegment segB0 = GgufTensorOps.tensorData(model, t0);
            MemorySegment segB1 = GgufTensorOps.tensorData(model, t1);
            MemorySegment segB2 = GgufTensorOps.tensorData(model, t2);

            int rc = (int) MATVEC_TRIPLE_BATCH_MH.invokeExact(segC0, segC1, segC2, segA, segB0, segB1, segB2,
                    K, N0, N1, N2, type0, type1, type2, batchSize);
            if (rc == 0) {
                MemorySegment.copy(segC0, ValueLayout.JAVA_FLOAT, 0, out0, 0, totalC0);
                MemorySegment.copy(segC1, ValueLayout.JAVA_FLOAT, 0, out1, 0, totalC1);
                MemorySegment.copy(segC2, ValueLayout.JAVA_FLOAT, 0, out2, 0, totalC2);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal matVecTripleBatch failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes fused FFN across a batch of tokens in a single command buffer submission.
     */
    public static boolean ffnGatedFusedBatch(GGUFModel model,
                                             GGUFTensorInfo ffnGate,
                                             GGUFTensorInfo ffnUp,
                                             GGUFTensorInfo ffnDown,
                                             float[] inABatch,
                                             float[] outDownBatch,
                                             boolean isGeGLU,
                                             int batchSize) {
        if (!METAL_ENABLED || FFN_GATED_FUSED_BATCH_MH == null ||
            ffnGate == null || ffnUp == null || ffnDown == null || inABatch == null || outDownBatch == null || batchSize <= 0) {
            return false;
        }
        ensureModelRegistered(model);

        int typeGate = (int) ffnGate.typeId();
        int typeUp = (int) ffnUp.typeId();
        int typeDown = (int) ffnDown.typeId();
        if (!isTypeSupported(typeGate) || !isTypeSupported(typeUp) || !isTypeSupported(typeDown)) {
            return false;
        }

        int dim = (int) GgufTensorOps.matrixColumns(ffnGate);
        int ffnDim = (int) GgufTensorOps.matrixRows(ffnGate);
        int totalDim = batchSize * dim;
        if (inABatch.length < totalDim || outDownBatch.length < totalDim) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segA = cache.ensureA(totalDim);
            MemorySegment segOut = cache.ensureC0(totalDim);

            MemorySegment.copy(inABatch, 0, segA, ValueLayout.JAVA_FLOAT, 0, totalDim);

            MemorySegment segGateW = GgufTensorOps.tensorData(model, ffnGate);
            MemorySegment segUpW = GgufTensorOps.tensorData(model, ffnUp);
            MemorySegment segDownW = GgufTensorOps.tensorData(model, ffnDown);

            int activationKind = isGeGLU ? 2 : 1; // 2=GeGLU, 1=SwiGLU

            int rc = (int) FFN_GATED_FUSED_BATCH_MH.invokeExact(
                    segOut, segA, segGateW, segUpW, segDownW,
                    dim, ffnDim, typeGate, typeUp, typeDown, activationKind, batchSize);

            if (rc == 0) {
                MemorySegment.copy(segOut, ValueLayout.JAVA_FLOAT, 0, outDownBatch, 0, totalDim);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal ffnGatedFusedBatch failed, falling back", t);
            return false;
        }
    }

    /**
     * Executes fused wO projection + postAttnNorm + attn residual + ffnNorm + FFN + postFfnNorm + ffn residual
     * all in a SINGLE Metal command buffer with in-encoder barriers.
     *
     * @param model GGUF model
     * @param x in-out residual array (length dim) — updated in place on success
     * @param attnFull attention output (wO matvec input, length wO_K)
     * @param wO wO weight tensor
     * @param postAttnW optional postAttnNorm weights (or null)
     * @param ffnNormW ffnNorm weights (required)
     * @param postFfnW optional postFfnNorm weights (or null)
     * @param eps RMSNorm epsilon
     * @param ffnGate gate weight tensor
     * @param ffnUp up weight tensor
     * @param ffnDown down weight tensor
     * @param isGeGLU true for GeGLU (Gemma), false for SwiGLU (Llama)
     * @return true if executed on GPU successfully, false to fall back to CPU
     */
    public static boolean woFfnFused(
            GGUFModel model,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU
    ) {
        return woFfnFused(model, -1, x, attnFull, wO, postAttnW, ffnNormW, postFfnW, eps, ffnGate, ffnUp, ffnDown, isGeGLU);
    }

    public static boolean woFfnFused(
            GGUFModel model,
            int layer,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU
    ) {
        if (!METAL_ENABLED || WO_FFN_FUSED_MH == null ||
                wO == null || ffnGate == null || ffnUp == null || ffnDown == null || ffnNormW == null ||
                x == null || attnFull == null) {
            return false;
        }
        ensureModelRegistered(model);

        int typeWO = (int) wO.typeId();
        int typeGate = (int) ffnGate.typeId();
        int typeUp = (int) ffnUp.typeId();
        int typeDown = (int) ffnDown.typeId();
        if (!isTypeSupported(typeWO) || !isTypeSupported(typeGate) ||
                !isTypeSupported(typeUp) || !isTypeSupported(typeDown)) {
            return false;
        }

        int wO_K = (int) GgufTensorOps.matrixColumns(wO);
        int wO_N = (int) GgufTensorOps.matrixRows(wO);
        int ffnDim = (int) GgufTensorOps.matrixRows(ffnGate);

        if (attnFull.length < wO_K || x.length < wO_N) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segX = cache.ensureX(wO_N);
            MemorySegment segAttn = cache.ensureAttn(wO_K);

            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0, wO_N);
            MemorySegment.copy(attnFull, 0, segAttn, ValueLayout.JAVA_FLOAT, 0, wO_K);

            int hasPostAttn = (postAttnW != null && postAttnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostAttn = MemorySegment.NULL;
            if (hasPostAttn == 1) {
                segPostAttn = (layer >= 0) ? normSlice(0, layer) : MemorySegment.NULL;
                if (segPostAttn.equals(MemorySegment.NULL)) {
                    segPostAttn = cache.ensureNorm1(wO_N);
                    MemorySegment.copy(postAttnW, 0, segPostAttn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segFfnNorm = (layer >= 0) ? normSlice(1, layer) : MemorySegment.NULL;
            if (segFfnNorm.equals(MemorySegment.NULL)) {
                segFfnNorm = cache.ensureNorm2(wO_N);
                MemorySegment.copy(ffnNormW, 0, segFfnNorm, ValueLayout.JAVA_FLOAT, 0, wO_N);
            }

            int hasPostFfn = (postFfnW != null && postFfnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostFfn = MemorySegment.NULL;
            if (hasPostFfn == 1) {
                segPostFfn = (layer >= 0) ? normSlice(2, layer) : MemorySegment.NULL;
                if (segPostFfn.equals(MemorySegment.NULL)) {
                    segPostFfn = cache.ensureNorm3(wO_N);
                    MemorySegment.copy(postFfnW, 0, segPostFfn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segWO = GgufTensorOps.tensorData(model, wO);
            MemorySegment segGate = GgufTensorOps.tensorData(model, ffnGate);
            MemorySegment segUp = GgufTensorOps.tensorData(model, ffnUp);
            MemorySegment segDown = GgufTensorOps.tensorData(model, ffnDown);

            int activationKind = isGeGLU ? 2 : 1;
            int addOne = 0; // Standard RMSNorm (GGUF weights already include any offsets)

            int rc = (int) WO_FFN_FUSED_MH.invokeExact(
                    segX, segAttn,
                    segWO, typeWO, wO_K, wO_N,
                    segPostAttn, hasPostAttn, addOne,
                    segFfnNorm, addOne,
                    segPostFfn, hasPostFfn, addOne,
                    eps,
                    segGate, segUp, segDown,
                    typeGate, typeUp, typeDown,
                    ffnDim, activationKind);

            if (rc == 0) {
                MemorySegment.copy(segX, ValueLayout.JAVA_FLOAT, 0, x, 0, wO_N);
                return true;
            } else {
                return false;
            }
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal woFfnFused failed, falling back", t);
            return false;
        }
    }

    public static boolean woFfnNextQkvFused(
            GGUFModel model,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            float outScale,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU,
            boolean hasNext,
            float[] nextAttnNormW,
            float[] nextQ, float[] nextK, float[] nextV,
            GGUFTensorInfo nextWQ, GGUFTensorInfo nextWK, GGUFTensorInfo nextWV
    ) {
        return woFfnNextQkvFused(model, -1, x, attnFull, wO, postAttnW, ffnNormW, postFfnW, eps, outScale,
                ffnGate, ffnUp, ffnDown, isGeGLU, hasNext, nextAttnNormW, nextQ, nextK, nextV, nextWQ, nextWK, nextWV);
    }

    public static boolean woFfnNextQkvFused(
            GGUFModel model,
            int layer,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            float outScale,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU,
            boolean hasNext,
            float[] nextAttnNormW,
            float[] nextQ, float[] nextK, float[] nextV,
            GGUFTensorInfo nextWQ, GGUFTensorInfo nextWK, GGUFTensorInfo nextWV
    ) {
        if (!METAL_ENABLED || WO_FFN_NEXT_QKV_FUSED_MH == null ||
                wO == null || ffnGate == null || ffnUp == null || ffnDown == null || ffnNormW == null ||
                x == null || attnFull == null) {
            System.err.printf("[woFfnNextQkvFused guard failed] METAL_ENABLED=%b MH=%s wO=%s gate=%s norm=%s%n",
                    METAL_ENABLED, (WO_FFN_NEXT_QKV_FUSED_MH != null), (wO != null), (ffnGate != null), (ffnNormW != null));
            return false;
        }
        ensureModelRegistered(model);

        int typeWO = (int) wO.typeId();
        int typeGate = (int) ffnGate.typeId();
        int typeUp = (int) ffnUp.typeId();
        int typeDown = (int) ffnDown.typeId();
        if (!isTypeSupported(typeWO) || !isTypeSupported(typeGate) ||
                !isTypeSupported(typeUp) || !isTypeSupported(typeDown)) {
            return false;
        }

        int wO_K = (int) GgufTensorOps.matrixColumns(wO);
        int wO_N = (int) GgufTensorOps.matrixRows(wO);
        int ffnDim = (int) GgufTensorOps.matrixRows(ffnGate);

        if (attnFull.length < wO_K || x.length < wO_N) {
            return false;
        }

        int hasNextInt = (hasNext && nextWQ != null && nextAttnNormW != null && nextQ != null) ? 1 : 0;
        int nextTypeQ = 0, nextTypeK = 0, nextTypeV = 0;
        int nextN0 = 0, nextN1 = 0, nextN2 = 0;
        MemorySegment segNextWQ = MemorySegment.NULL;
        MemorySegment segNextWK = MemorySegment.NULL;
        MemorySegment segNextWV = MemorySegment.NULL;

        if (hasNextInt == 1) {
            nextTypeQ = (int) nextWQ.typeId();
            if (!isTypeSupported(nextTypeQ)) return false;
            nextN0 = (int) GgufTensorOps.matrixRows(nextWQ);
            segNextWQ = GgufTensorOps.tensorData(model, nextWQ);

            if (nextWK != null && nextK != null) {
                nextTypeK = (int) nextWK.typeId();
                if (!isTypeSupported(nextTypeK)) return false;
                nextN1 = (int) GgufTensorOps.matrixRows(nextWK);
                segNextWK = GgufTensorOps.tensorData(model, nextWK);
            }
            if (nextWV != null && nextV != null) {
                nextTypeV = (int) nextWV.typeId();
                if (!isTypeSupported(nextTypeV)) return false;
                nextN2 = (int) GgufTensorOps.matrixRows(nextWV);
                segNextWV = GgufTensorOps.tensorData(model, nextWV);
            }
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segX = cache.ensureX(wO_N);
            MemorySegment segAttn = cache.ensureAttn(wO_K);

            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0, wO_N);
            MemorySegment.copy(attnFull, 0, segAttn, ValueLayout.JAVA_FLOAT, 0, wO_K);

            int hasPostAttn = (postAttnW != null && postAttnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostAttn = MemorySegment.NULL;
            if (hasPostAttn == 1) {
                segPostAttn = (layer >= 0) ? normSlice(0, layer) : MemorySegment.NULL;
                if (segPostAttn.equals(MemorySegment.NULL)) {
                    segPostAttn = cache.ensureNorm1(wO_N);
                    MemorySegment.copy(postAttnW, 0, segPostAttn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segFfnNorm = (layer >= 0) ? normSlice(1, layer) : MemorySegment.NULL;
            if (segFfnNorm.equals(MemorySegment.NULL)) {
                segFfnNorm = cache.ensureNorm2(wO_N);
                MemorySegment.copy(ffnNormW, 0, segFfnNorm, ValueLayout.JAVA_FLOAT, 0, wO_N);
            }

            int hasPostFfn = (postFfnW != null && postFfnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostFfn = MemorySegment.NULL;
            if (hasPostFfn == 1) {
                segPostFfn = (layer >= 0) ? normSlice(2, layer) : MemorySegment.NULL;
                if (segPostFfn.equals(MemorySegment.NULL)) {
                    segPostFfn = cache.ensureNorm3(wO_N);
                    MemorySegment.copy(postFfnW, 0, segPostFfn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segWO = GgufTensorOps.tensorData(model, wO);
            MemorySegment segGate = GgufTensorOps.tensorData(model, ffnGate);
            MemorySegment segUp = GgufTensorOps.tensorData(model, ffnUp);
            MemorySegment segDown = GgufTensorOps.tensorData(model, ffnDown);

            int activationKind = isGeGLU ? 2 : 1;

            MemorySegment segNextAttnNorm = MemorySegment.NULL;
            MemorySegment segNextQ = MemorySegment.NULL;
            MemorySegment segNextK = MemorySegment.NULL;
            MemorySegment segNextV = MemorySegment.NULL;

            if (hasNextInt == 1) {
                segNextAttnNorm = (layer >= 0) ? normSlice(3, layer + 1) : MemorySegment.NULL;
                if (segNextAttnNorm.equals(MemorySegment.NULL)) {
                    segNextAttnNorm = cache.ensureA(wO_N);
                    MemorySegment.copy(nextAttnNormW, 0, segNextAttnNorm, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
                segNextQ = cache.ensureC0(nextN0);
                if (nextN1 > 0) segNextK = cache.ensureC1(nextN1);
                if (nextN2 > 0) segNextV = cache.ensureC2(nextN2);
            }

            int rc = (int) WO_FFN_NEXT_QKV_FUSED_MH.invokeExact(
                    segX, segAttn,
                    segWO, typeWO, wO_K, wO_N,
                    segPostAttn, hasPostAttn,
                    segFfnNorm,
                    segPostFfn, hasPostFfn,
                    eps,
                    outScale,
                    segGate, segUp, segDown,
                    typeGate, typeUp, typeDown,
                    ffnDim, activationKind,
                    hasNextInt,
                    segNextAttnNorm,
                    segNextQ, segNextK, segNextV,
                    segNextWQ, segNextWK, segNextWV,
                    nextTypeQ, nextTypeK, nextTypeV,
                    nextN0, nextN1, nextN2);

            if (rc == 0) {
                MemorySegment.copy(segX, ValueLayout.JAVA_FLOAT, 0, x, 0, wO_N);
                if (hasNextInt == 1) {
                    MemorySegment.copy(segNextQ, ValueLayout.JAVA_FLOAT, 0, nextQ, 0, nextN0);
                    if (nextN1 > 0 && nextK != null) {
                        MemorySegment.copy(segNextK, ValueLayout.JAVA_FLOAT, 0, nextK, 0, nextN1);
                    }
                    if (nextN2 > 0 && nextV != null) {
                        MemorySegment.copy(segNextV, ValueLayout.JAVA_FLOAT, 0, nextV, 0, nextN2);
                    }
                }
                return true;
            } else {
                System.err.println("[woFfnNextQkvFused failed] rc=" + rc);
                return false;
            }
        } catch (Throwable t) {
            System.err.println("[woFfnNextQkvFused exception] " + t.getMessage());
            t.printStackTrace(System.err);
            return false;
        }
    }

    public static boolean woFfnLogitsFused(
            GGUFModel model,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            float outScale,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU,
            float[] outputNormW,
            GGUFTensorInfo outputW,
            float[] logits,
            float cap
    ) {
        return woFfnLogitsFused(model, -1, x, attnFull, wO, postAttnW, ffnNormW, postFfnW, eps, outScale,
                ffnGate, ffnUp, ffnDown, isGeGLU, outputNormW, outputW, logits, cap);
    }

    public static boolean woFfnLogitsFused(
            GGUFModel model,
            int layer,
            float[] x,
            float[] attnFull,
            GGUFTensorInfo wO,
            float[] postAttnW,
            float[] ffnNormW,
            float[] postFfnW,
            float eps,
            float outScale,
            GGUFTensorInfo ffnGate,
            GGUFTensorInfo ffnUp,
            GGUFTensorInfo ffnDown,
            boolean isGeGLU,
            float[] outputNormW,
            GGUFTensorInfo outputW,
            float[] logits,
            float cap
    ) {
        if (!METAL_ENABLED || WO_FFN_LOGITS_FUSED_MH == null ||
                wO == null || ffnGate == null || ffnUp == null || ffnDown == null || ffnNormW == null ||
                outputNormW == null || outputW == null ||
                x == null || attnFull == null || logits == null) {
            return false;
        }
        ensureModelRegistered(model);

        int typeWO = (int) wO.typeId();
        int typeGate = (int) ffnGate.typeId();
        int typeUp = (int) ffnUp.typeId();
        int typeDown = (int) ffnDown.typeId();
        int typeOutputW = (int) outputW.typeId();
        if (!isTypeSupported(typeWO) || !isTypeSupported(typeGate) ||
                !isTypeSupported(typeUp) || !isTypeSupported(typeDown) ||
                !isTypeSupported(typeOutputW)) {
            return false;
        }

        int wO_K = (int) GgufTensorOps.matrixColumns(wO);
        int wO_N = (int) GgufTensorOps.matrixRows(wO);
        int ffnDim = (int) GgufTensorOps.matrixRows(ffnGate);
        int outputK = (int) GgufTensorOps.matrixColumns(outputW);
        int outputN = (int) GgufTensorOps.matrixRows(outputW);

        if (attnFull.length < wO_K || x.length < wO_N || logits.length < outputN || outputNormW.length < outputK) {
            return false;
        }

        try {
            MetalBufferCache cache = BUFFERS.get();
            MemorySegment segX = cache.ensureX(wO_N);
            MemorySegment segAttn = cache.ensureAttn(wO_K);

            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0, wO_N);
            MemorySegment.copy(attnFull, 0, segAttn, ValueLayout.JAVA_FLOAT, 0, wO_K);

            int hasPostAttn = (postAttnW != null && postAttnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostAttn = MemorySegment.NULL;
            if (hasPostAttn == 1) {
                segPostAttn = (layer >= 0) ? normSlice(0, layer) : MemorySegment.NULL;
                if (segPostAttn.equals(MemorySegment.NULL)) {
                    segPostAttn = cache.ensureNorm1(wO_N);
                    MemorySegment.copy(postAttnW, 0, segPostAttn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segFfnNorm = (layer >= 0) ? normSlice(1, layer) : MemorySegment.NULL;
            if (segFfnNorm.equals(MemorySegment.NULL)) {
                segFfnNorm = cache.ensureNorm2(wO_N);
                MemorySegment.copy(ffnNormW, 0, segFfnNorm, ValueLayout.JAVA_FLOAT, 0, wO_N);
            }

            int hasPostFfn = (postFfnW != null && postFfnW.length >= wO_N) ? 1 : 0;
            MemorySegment segPostFfn = MemorySegment.NULL;
            if (hasPostFfn == 1) {
                segPostFfn = (layer >= 0) ? normSlice(2, layer) : MemorySegment.NULL;
                if (segPostFfn.equals(MemorySegment.NULL)) {
                    segPostFfn = cache.ensureNorm3(wO_N);
                    MemorySegment.copy(postFfnW, 0, segPostFfn, ValueLayout.JAVA_FLOAT, 0, wO_N);
                }
            }

            MemorySegment segWO = GgufTensorOps.tensorData(model, wO);
            MemorySegment segGate = GgufTensorOps.tensorData(model, ffnGate);
            MemorySegment segUp = GgufTensorOps.tensorData(model, ffnUp);
            MemorySegment segDown = GgufTensorOps.tensorData(model, ffnDown);

            int activationKind = isGeGLU ? 2 : 1;

            MemorySegment segOutputNorm = cache.ensureOutputNorm(outputK);
            MemorySegment.copy(outputNormW, 0, segOutputNorm, ValueLayout.JAVA_FLOAT, 0, outputK);

            MemorySegment segOutputW = GgufTensorOps.tensorData(model, outputW);
            MemorySegment segLogits = cache.ensureLogits(outputN);

            int rc = (int) WO_FFN_LOGITS_FUSED_MH.invokeExact(
                    segX, segAttn,
                    segWO, typeWO, wO_K, wO_N,
                    segPostAttn, hasPostAttn,
                    segFfnNorm,
                    segPostFfn, hasPostFfn,
                    eps,
                    outScale,
                    segGate, segUp, segDown,
                    typeGate, typeUp, typeDown,
                    ffnDim, activationKind,
                    segOutputNorm,
                    segOutputW, typeOutputW, outputK, outputN,
                    segLogits, cap);

            if (rc == 0) {
                MemorySegment.copy(segX, ValueLayout.JAVA_FLOAT, 0, x, 0, wO_N);
                MemorySegment.copy(segLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, outputN);
                return true;
            } else {
                System.err.println("[woFfnLogitsFused failed] rc=" + rc);
                return false;
            }
        } catch (Throwable t) {
            System.err.println("[woFfnLogitsFused exception] " + t.getMessage());
            t.printStackTrace(System.err);
            return false;
        }
    }

    /**
     * Applies logit soft-capping on the GPU (in-place): {@code logits[i] = cap * tanh(logits[i] / cap)}.
     *
     * <p>Eliminates 262 144 {@code Math.tanh} calls per token on the CPU (6–14 ms → ~0.5 ms).</p>
     *
     * @param logits float array to modify in-place
     * @param cap    soft-cap value (e.g. 30.0f for Gemma-4)
     * @return true if GPU softcap succeeded; false means caller should fall back to CPU tanh loop
     */
    public static boolean softcapLogits(float[] logits, float cap) {
        if (!METAL_ENABLED || SOFTCAP_MH == null || logits == null || cap <= 0f) {
            return false;
        }
        int N = logits.length;
        try {
            MetalBufferCache cache = BUFFERS.get();
            // Reuse segA as the logit staging buffer (off-heap, page-aligned → usable with newBufferWithBytesNoCopy)
            MemorySegment seg = cache.ensureA(N);
            // Copy logits → off-heap
            MemorySegment.copy(logits, 0, seg, ValueLayout.JAVA_FLOAT, 0, N);
            // GPU kernel: modifies seg in-place
            int rc = (int) SOFTCAP_MH.invokeExact(seg, N, cap);
            if (rc != 0) return false;
            // Copy result back
            MemorySegment.copy(seg, ValueLayout.JAVA_FLOAT, 0, logits, 0, N);
            return true;
        } catch (Throwable t) {
            LOG.log(Level.FINE, "Metal softcapLogits failed, falling back to CPU", t);
            return false;
        }
    }

    // ── Single Command Buffer Decode Step API ────────────────────────────────

    public static boolean isDecodeStepAvailable() {
        return METAL_ENABLED && DECODE_STEP_MH != null;
    }

    public static boolean initKVCache(int nLayers, int maxSeq, int[] layerStrides) {
        if (!isDecodeStepAvailable() || INIT_KV_CACHE_MH == null) return false;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment stridesSeg = arena.allocate((long) nLayers * Integer.BYTES);
            MemorySegment.copy(layerStrides, 0, stridesSeg, ValueLayout.JAVA_INT, 0, nLayers);
            int rc = (int) INIT_KV_CACHE_MH.invokeExact(nLayers, maxSeq, stridesSeg);
            return rc == 0;
        } catch (Throwable t) {
            LOG.log(Level.WARNING, "Failed to init Metal KV cache: " + t.getMessage(), t);
            return false;
        }
    }

    public static boolean syncKVCache(int layer, int seqLen, float[] kData, float[] vData) {
        if (!isDecodeStepAvailable() || SYNC_KV_CACHE_MH == null || kData == null || vData == null) return false;
        try (Arena arena = Arena.ofConfined()) {
            int len = kData.length;
            MemorySegment kSeg = arena.allocate((long) len * Float.BYTES);
            MemorySegment vSeg = arena.allocate((long) len * Float.BYTES);
            MemorySegment.copy(kData, 0, kSeg, ValueLayout.JAVA_FLOAT, 0, len);
            MemorySegment.copy(vData, 0, vSeg, ValueLayout.JAVA_FLOAT, 0, len);
            int rc = (int) SYNC_KV_CACHE_MH.invokeExact(layer, seqLen, kSeg, vSeg);
            return rc == 0;
        } catch (Throwable t) {
            LOG.log(Level.WARNING, "Failed to sync Metal KV cache: " + t.getMessage(), t);
            return false;
        }
    }

    public static boolean registerDecodeLayer(
            int layer,
            GGUFModel model,
            GGUFTensorInfo wQ, GGUFTensorInfo wK, GGUFTensorInfo wV, GGUFTensorInfo wO,
            GGUFTensorInfo gate, GGUFTensorInfo up, GGUFTensorInfo down,
            int n0, int n1, int n2, int wO_K, int wO_N, int ffnDim,
            int nH, int nKVH, int headDim, int stride,
            float freqBase, int isSwa) {
        if (!isDecodeStepAvailable() || REGISTER_DECODE_LAYER_MH == null) return false;
        try {
            MemorySegment segWQ = (wQ != null) ? GgufTensorOps.tensorData(model, wQ) : MemorySegment.NULL;
            MemorySegment segWK = (wK != null) ? GgufTensorOps.tensorData(model, wK) : MemorySegment.NULL;
            MemorySegment segWV = (wV != null) ? GgufTensorOps.tensorData(model, wV) : MemorySegment.NULL;
            MemorySegment segWO = (wO != null) ? GgufTensorOps.tensorData(model, wO) : MemorySegment.NULL;
            MemorySegment segGate = (gate != null) ? GgufTensorOps.tensorData(model, gate) : MemorySegment.NULL;
            MemorySegment segUp = (up != null) ? GgufTensorOps.tensorData(model, up) : MemorySegment.NULL;
            MemorySegment segDown = (down != null) ? GgufTensorOps.tensorData(model, down) : MemorySegment.NULL;

            int typeQ = (wQ != null) ? (int) wQ.typeId() : 0;
            int typeK = (wK != null) ? (int) wK.typeId() : 0;
            int typeV = (wV != null) ? (int) wV.typeId() : 0;
            int typeWO = (wO != null) ? (int) wO.typeId() : 0;
            int typeGate = (gate != null) ? (int) gate.typeId() : 0;
            int typeUp = (up != null) ? (int) up.typeId() : 0;
            int typeDown = (down != null) ? (int) down.typeId() : 0;

            int rc = (int) REGISTER_DECODE_LAYER_MH.invokeExact(
                    layer,
                    segWQ, segWK, segWV, segWO,
                    segGate, segUp, segDown,
                    typeQ, typeK, typeV, typeWO, typeGate, typeUp, typeDown,
                    n0, n1, n2, wO_K, wO_N, ffnDim,
                    nH, nKVH, headDim, stride,
                    freqBase, isSwa);
            return rc == 0;
        } catch (Throwable t) {
            LOG.log(Level.WARNING, "Failed to register decode layer " + layer + ": " + t.getMessage(), t);
            return false;
        }
    }

    private static volatile MemorySegment g_outputNormSeg = null;

    public static boolean registerOutput(
            GGUFModel model,
            float[] outputNormW,
            GGUFTensorInfo outputW, int outputK, int outputN,
            float cap) {
        if (!isDecodeStepAvailable() || REGISTER_OUTPUT_MH == null) return false;
        try {
            MemorySegment segOutputNorm = MemorySegment.NULL;
            if (outputNormW != null) {
                if (g_outputNormSeg == null || g_outputNormSeg.byteSize() < (long) outputK * Float.BYTES) {
                    Arena arena = Arena.ofAuto();
                    g_outputNormSeg = arena.allocate((long) outputK * Float.BYTES);
                }
                MemorySegment.copy(outputNormW, 0, g_outputNormSeg, ValueLayout.JAVA_FLOAT, 0, outputK);
                segOutputNorm = g_outputNormSeg;
            }
            MemorySegment segOutputW = (outputW != null) ? GgufTensorOps.tensorData(model, outputW) : MemorySegment.NULL;
            int typeOutputW = (outputW != null) ? (int) outputW.typeId() : 0;

            int rc = (int) REGISTER_OUTPUT_MH.invokeExact(segOutputNorm, segOutputW, typeOutputW, outputK, outputN, cap);
            return rc == 0;
        } catch (Throwable t) {
            LOG.log(Level.WARNING, "Failed to register output: " + t.getMessage(), t);
            return false;
        }
    }

    public static boolean decodeStep(int pos, int slidingWindow, float eps, float[] x, float[] logits) {
        if (!isDecodeStepAvailable() || DECODE_STEP_MH == null || x == null || logits == null) return false;
        try {
            MetalBufferCache cache = BUFFERS.get();
            int dim = x.length;
            int vocab = logits.length;
            MemorySegment segX = cache.ensureX(dim);
            MemorySegment segLogits = cache.ensureLogits(vocab);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0, dim);

            int rc = (int) DECODE_STEP_MH.invokeExact(pos, slidingWindow, eps, segX, segLogits);
            if (rc == 0) {
                MemorySegment.copy(segLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocab);
                return true;
            }
            return false;
        } catch (Throwable t) {
            LOG.log(Level.WARNING, "Failed in decodeStep: " + t.getMessage(), t);
            return false;
        }
    }

    private GgufMetalBridge() {}
}

