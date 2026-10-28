/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AccelWeightBridge.java
 * ──────────────────────
 * Zero-copy bridge between FFM-mmap'd safetensor files and AccelTensor.
 * Replaces SafetensorWeightBridge — no LibTorch dependency.
 *
 * BF16/F16 weights are upcast to F32 at load time using pure Java.
 */
package tech.kayys.aljabr.safetensor.loader.bridge;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Zero-copy bridge from mmap'd safetensor files to {@link AccelTensor}.
 * <p>
 * F32 weights are wrapped directly (zero-copy).
 * BF16/F16 weights are upcast to F32 (copy).
 * @author bhangun
 */
@ApplicationScoped
public class AccelWeightBridge {

    private static final Logger log = Logger.getLogger(AccelWeightBridge.class);

    /**
     * Bridge a single SafetensorTensor to an AccelTensor.
     * <p>
     * F32 tensors are zero-copy wrapped.
     * BF16/F16 tensors are upcast to F32.
     */
    public AccelTensor bridge(SafetensorTensor st) {
        SafetensorDType dtype = st.dtype();
        long[] shape = st.info().shape();
        String name = st.name();

        AccelTensor t;
        if (dtype == SafetensorDType.F32) {
            // Zero-copy: wrap the mmap'd segment directly
            log.tracef("Bridge zero-copy: '%s' F32 shape=%s", name, java.util.Arrays.toString(shape));
            t = AccelTensor.wrapSegment(st.segment(), shape);
        } else if (dtype == SafetensorDType.I8 || dtype == SafetensorDType.U32 || name.endsWith(".qweight")) {
             // Handle as Quantized INT8, INT4, or packed U32 bits (MLX)
             log.debugf("Bridge wrapped (quantized): '%s' %s shape=%s", name, dtype, java.util.Arrays.toString(shape));
             t = AccelTensor.wrapSegment(st.segment(), shape);
             // We'll set the quantization metadata in bridgeAll where we have the full session
        } else if (dtype == SafetensorDType.BF16 || dtype == SafetensorDType.F16) {
             if (shape.length >= 2) {
                 log.debugf("Bridge wrapped (half): '%s' %s shape=%s", name, dtype, java.util.Arrays.toString(shape));
                 t = AccelTensor.wrapSegment(st.segment(), shape);
                 t.withQuantization(
                         dtype == SafetensorDType.BF16
                                 ? AccelTensor.QuantType.BF16
                                 : AccelTensor.QuantType.F16,
                         null,
                         null,
                         -1);
             } else {
                 log.debugf("Bridge upcast (1D half): '%s' %s shape=%s", name, dtype, java.util.Arrays.toString(shape));
                 t = upcastToF32(st, dtype);
             }
        } else {
            // For integer types, convert to float
            log.debugf("Bridge convert: '%s' %s → F32", name, dtype);
            t = convertToF32(st, dtype);
        }

        return t;
    }

    /**
     * Bridge all tensors from an open session.
     */
    public Map<String, AccelTensor> bridgeAll(SafetensorShardSession session) {
        Set<String> names = session.tensorNames();
        Map<String, AccelTensor> result = new HashMap<>(names.size() * 2);
        try {
            for (String name : names) {
                if (isMetadataTensor(name)) continue; 

                SafetensorTensor st = session.tensor(name);
                AccelTensor t = bridge(st);
                
                if (isWeightTensor(name)) {
                    t = attachQuantizationMetadata(t, name, session);
                }
                
                result.put(name, t);
            }
        } catch (Exception e) {
            result.values().forEach(AccelTensor::close);
            throw new RuntimeException("Failed to bridge weights: " + e.getMessage(), e);
        }
        return result;
    }

    // ── Internal ──────────────────────────────────────────────────────

    private AccelTensor upcastToF32(SafetensorTensor st, SafetensorDType dtype) {
        long n = st.numElements();
        long[] shape = st.info().shape();
        
        // Allocate off-heap F32 tensor
        AccelTensor t = AccelTensor.zeros(shape);
        MemorySegment dst = t.dataPtr();
        
        if (n <= Integer.MAX_VALUE) {
            // Fast path for smaller tensors using float[]
            float[] values = st.toFloatArray();
            MemorySegment.copy(MemorySegment.ofArray(values), ValueLayout.JAVA_FLOAT, 0, 
                               dst, ValueLayout.JAVA_FLOAT, 0, n);
            return t;
        }

        // Slow path for large tensors: element-by-element upcast in native memory
        log.warnf("Upcasting LARGE tensor '%s' (%d elements) — this may be slow", st.name(), n);
        for (long i = 0; i < n; i++) {
            float val = (dtype == SafetensorDType.BF16) ? st.getBF16AsFloat(i) : st.getF16AsFloat(i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
        }
        return t;
    }

    private AccelTensor convertToF32(SafetensorTensor st, SafetensorDType dtype) {
        long n = st.numElements();
        long[] shape = st.info().shape();
        MemorySegment seg = st.segment();
        
        AccelTensor t = AccelTensor.zeros(shape);
        MemorySegment dst = t.dataPtr();

        for (long i = 0; i < n; i++) {
            float f;
            switch (dtype) {
                case I8 -> f = seg.get(ValueLayout.JAVA_BYTE, i);
                case I16, U16 -> f = seg.getAtIndex(ValueLayout.JAVA_SHORT.withByteAlignment(1), i);
                case I32, U32 -> f = seg.getAtIndex(ValueLayout.JAVA_INT.withByteAlignment(1), i);
                case I64, U64 -> f = seg.getAtIndex(ValueLayout.JAVA_LONG.withByteAlignment(1), i);
                case F64 -> f = (float) seg.getAtIndex(ValueLayout.JAVA_DOUBLE.withByteAlignment(1), i);
                case U8 -> f = Byte.toUnsignedInt(seg.get(ValueLayout.JAVA_BYTE, i));
                case BOOL -> f = seg.get(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0f : 0.0f;
                default -> throw new UnsupportedOperationException("Cannot convert " + dtype + " to F32");
            }
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, f);
        }

        return t;
    }

    private boolean isWeightTensor(String name) {
        return name.endsWith(".weight") || name.endsWith(".qweight");
    }

    private boolean isMetadataTensor(String name) {
        return name.endsWith(".scales") || name.endsWith(".qzeros") || name.endsWith(".g_idx");
    }

    private AccelTensor attachQuantizationMetadata(AccelTensor t, String weightName, SafetensorShardSession session) {
        String baseName = weightName.substring(0, weightName.lastIndexOf("."));
        SafetensorTensor st = session.tensor(weightName);
        
        if (st.dtype() == SafetensorDType.BF16 && st.info().shape().length >= 2) {
            t.withQuantization(AccelTensor.QuantType.BF16, null, null, -1);
            return t;
        } else if (st.dtype() == SafetensorDType.F16 && st.info().shape().length >= 2) {
            t.withQuantization(AccelTensor.QuantType.F16, null, null, -1);
            return t;
        }
        
        String scalesName = baseName + ".scales";
        if (session.findTensor(scalesName).isPresent()) {
            SafetensorTensor scalesSt = session.tensor(scalesName);
            MemorySegment scalesSeg = scalesSt.segment();
            
            AccelTensor.QuantType type;
            if (st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.I32 ||
                st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.U32 ||
                (st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.I8 && weightName.endsWith(".qweight"))) {
                type = AccelTensor.QuantType.INT4;
            } else if (st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.U8 && weightName.endsWith(".qweight")) {
                type = AccelTensor.QuantType.NF4;
            } else {
                type = AccelTensor.QuantType.INT8;
            }
            
            if (type == AccelTensor.QuantType.NF4 || type == AccelTensor.QuantType.INT4) {
                long[] logicalShape = t.shape().clone();
                if (st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.U8 ||
                    st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.I8) {
                    logicalShape[logicalShape.length - 1] *= 2;
                } else if (st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.I32 ||
                           st.dtype() == tech.kayys.aljabr.safetensor.loader.SafetensorDType.U32) {
                    logicalShape[logicalShape.length - 1] *= 8;
                }
                t = AccelTensor.wrapSegment(t.dataPtr(), logicalShape);
            }
            
            String zerosName = baseName + ".qzeros";
            MemorySegment zerosSeg = session.findTensor(zerosName).isPresent() 
                    ? session.tensor(zerosName).segment() : null;
            
            int groupSize = 128; 
            
            t.withQuantization(type, scalesSeg, zerosSeg, groupSize);
            log.debugf("Attached %s quantization to '%s'", type, weightName);
        }
        return t;
    }
}
