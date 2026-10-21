package tech.kayys.aljabr.gguf.runtime;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector API species and feature flags for GGUF tensor kernels.
 *
 * <p>The runtime uses this helper to keep SIMD lane choices and property-driven
 * vector enablement consistent across dense and quantized paths.</p>
 * @author bhangun
 */
final class GgufVectorConfig {
    static final VectorSpecies<Byte> Q4_DOT_BYTE_SPECIES = ByteVector.SPECIES_64;
    static final VectorSpecies<Float> Q4_DOT_FLOAT_SPECIES = FloatVector.SPECIES_256;
    static final int Q4_DOT_VECTOR_LANES = Q4_DOT_BYTE_SPECIES.length();
    static final VectorSpecies<Float> FLOAT_SUM_SPECIES = FloatVector.SPECIES_PREFERRED;
    static final int FLOAT_SUM_VECTOR_LANES = FLOAT_SUM_SPECIES.length();
    static final VectorSpecies<Integer> INT_SUM_SPECIES = intSpeciesForFloatLanes(FLOAT_SUM_VECTOR_LANES);
    static final VectorSpecies<Short> BF16_DOT_SHORT_SPECIES = shortSpeciesForFloatLanes(FLOAT_SUM_VECTOR_LANES);
    // 128-bit float vectors are enough to make byte-dot SIMD worthwhile on Apple Silicon.
    static final boolean BYTE_DOT_VECTOR_PREFERRED = FLOAT_SUM_VECTOR_LANES >= 4;
    static final boolean DENSE_F32_VECTOR_DOT_ENABLED =
            Boolean.parseBoolean(System.getProperty("aljabr.gguf.dense_f32.vector_dot", "true"));
    static final boolean DENSE_BF16_VECTOR_DOT_ENABLED =
            INT_SUM_SPECIES != null
                    && BF16_DOT_SHORT_SPECIES != null
                    && Boolean.parseBoolean(System.getProperty("aljabr.gguf.dense_bf16.vector_dot", "true"));
    static final boolean RAW_Q8_VECTOR_DOT_ENABLED =
            Boolean.parseBoolean(System.getProperty("aljabr.gguf.raw_q8.vector_dot", "true"));
    static final boolean Q4_DOT_VECTOR_ENABLED =
            byteDotVectorEnabled("aljabr.gguf.quant.vector_dot", "aljabr.gguf.q4k.vector_dot");
    static final boolean SIGNED_BYTE_DOT_VECTOR_ENABLED =
            byteDotVectorEnabled("aljabr.gguf.signed_byte.vector_dot", null);

    private GgufVectorConfig() {
    }

    static boolean byteDotVectorPreferred() {
        return BYTE_DOT_VECTOR_PREFERRED;
    }

    static int byteDotVectorLanes() {
        return Q4_DOT_VECTOR_LANES;
    }

    static int preferredFloatVectorLanes() {
        return FLOAT_SUM_VECTOR_LANES;
    }

    static boolean resolveByteDotVectorEnabled(String configured) {
        if (configured == null || configured.isBlank() || "auto".equalsIgnoreCase(configured.trim())) {
            return BYTE_DOT_VECTOR_PREFERRED;
        }
        return Boolean.parseBoolean(configured.trim());
    }

    private static VectorSpecies<Integer> intSpeciesForFloatLanes(int lanes) {
        return switch (lanes) {
            case 2 -> IntVector.SPECIES_64;
            case 4 -> IntVector.SPECIES_128;
            case 8 -> IntVector.SPECIES_256;
            case 16 -> IntVector.SPECIES_512;
            default -> null;
        };
    }

    private static VectorSpecies<Short> shortSpeciesForFloatLanes(int lanes) {
        return switch (lanes) {
            case 4 -> ShortVector.SPECIES_64;
            case 8 -> ShortVector.SPECIES_128;
            case 16 -> ShortVector.SPECIES_256;
            case 32 -> ShortVector.SPECIES_512;
            default -> null;
        };
    }

    private static boolean byteDotVectorEnabled(String primaryProperty, String legacyProperty) {
        String configured = System.getProperty(primaryProperty);
        if (configured == null && legacyProperty != null) {
            configured = System.getProperty(legacyProperty);
        }
        return resolveByteDotVectorEnabled(configured);
    }
}
