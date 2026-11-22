package tech.kayys.aljabr.cache.quantization;

import tech.kayys.aljabr.cache.KVCacheCompressionCodec;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Pure-Java reference implementation of the TurboQuant MSE KV-cache codec.
 *
 * <p>Each token/head vector is normalized, padded to a power-of-two dimension,
 * mixed by a seeded randomized Walsh-Hadamard transform, scaled by the square
 * root of that dimension, and quantized with the spherical Lloyd-Max codebook.
 * The original L2 norm is stored as an FP64 value per vector. Indices use the
 * canonical bit layout defined by {@link TurboQuantIndexBitPacking}.</p>
 *
 * <p>This codec implements the MSE reconstruction path only. It does not
 * implement QJL residual correction, fused attention, accelerator kernels,
 * or serving policy. It is a correctness/reference implementation.</p>
 */
public final class TurboQuantMseCodec implements KVCacheCompressionCodec {

    public static final String CODEC_ID = "turboquant-mse-reference";
    public static final int CODEC_VERSION = 1;

    private static final int METADATA_MAGIC = 0x54514D53; // "TQMS"
    private static final int METADATA_VERSION = 1;
    private static final int METADATA_HEADER_BYTES = 32;

    private final int bits;
    private final long seed;

    public TurboQuantMseCodec(int bits, long seed) {
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }
        this.bits = bits;
        this.seed = seed;
    }

    public int bits() {
        return bits;
    }

    public long seed() {
        return seed;
    }

    @Override
    public String id() {
        return CODEC_ID;
    }

    @Override
    public int version() {
        return CODEC_VERSION;
    }

    @Override
    public KVCacheEncodedBlock encode(
            float[] source,
            int sourceOffset,
            KVCacheCompressionContext context) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(context, "context");
        checkRange(source.length, sourceOffset, context.elementCount(), "source");

        int originalDimension = context.headDimension();
        int transformDimension = transformDimension(originalDimension);
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        TurboQuantScalarCodebook codebook =
                TurboQuantScalarCodebook.forDimension(transformDimension, bits);
        TurboQuantHadamardTransform transform =
                new TurboQuantHadamardTransform(transformDimension, seed);
        int quantizedElementCount = Math.multiplyExact(vectorCount, transformDimension);
        int[] indices = new int[quantizedElementCount];
        float[] vector = new float[transformDimension];
        float[] transformed = new float[transformDimension];
        double[] norms = new double[vectorCount];
        double coordinateScale = Math.sqrt(transformDimension);

        int indexOffset = 0;
        for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
            int sourceVectorOffset = sourceOffset + vectorIndex * originalDimension;
            double normSquared = 0.0;
            for (int i = 0; i < originalDimension; i++) {
                float value = source[sourceVectorOffset + i];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("source values must be finite");
                }
                normSquared += (double) value * value;
            }
            double norm = Math.sqrt(normSquared);
            if (!Double.isFinite(norm)) {
                throw new IllegalArgumentException("source vector norm must be finite");
            }
            norms[vectorIndex] = norm;
            if (norm == 0.0) {
                // Zero vectors decode exactly regardless of the codebook index.
                indexOffset += transformDimension;
                continue;
            }

            for (int i = 0; i < transformDimension; i++) {
                vector[i] = i < originalDimension
                        ? (float) (source[sourceVectorOffset + i] / norm)
                        : 0.0f;
            }
            transform.forward(vector, 0, transformed, 0);
            for (int i = 0; i < transformDimension; i++) {
                indices[indexOffset + i] = codebook.quantize(transformed[i] * coordinateScale);
            }
            indexOffset += transformDimension;
        }

        byte[] payload = TurboQuantIndexBitPacking.pack(indices, bits);
        byte[] metadata = encodeMetadata(originalDimension, transformDimension, vectorCount, norms);
        return new KVCacheEncodedBlock(id(), version(), context.elementCount(), payload, metadata);
    }

    @Override
    public void decode(
            KVCacheEncodedBlock encoded,
            float[] destination,
            int destinationOffset,
            KVCacheCompressionContext context) {
        Objects.requireNonNull(encoded, "encoded");
        Objects.requireNonNull(destination, "destination");
        Objects.requireNonNull(context, "context");
        if (!id().equals(encoded.codecId()) || version() != encoded.codecVersion()) {
            throw new IllegalArgumentException("encoded block uses an unsupported codec identity or version");
        }
        if (encoded.elementCount() != context.elementCount()) {
            throw new IllegalArgumentException("encoded element count does not match compression context");
        }
        checkRange(destination.length, destinationOffset, context.elementCount(), "destination");

        int originalDimension = context.headDimension();
        int expectedTransformDimension = transformDimension(originalDimension);
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        double[] norms = decodeMetadata(
                encoded.metadata(), originalDimension, expectedTransformDimension, vectorCount);
        int quantizedElementCount = Math.multiplyExact(vectorCount, expectedTransformDimension);
        int[] indices = TurboQuantIndexBitPacking.unpack(encoded.payload(), quantizedElementCount, bits);

        TurboQuantScalarCodebook codebook =
                TurboQuantScalarCodebook.forDimension(expectedTransformDimension, bits);
        TurboQuantHadamardTransform transform =
                new TurboQuantHadamardTransform(expectedTransformDimension, seed);
        float[] transformed = new float[expectedTransformDimension];
        float[] reconstructed = new float[expectedTransformDimension];
        double inverseScale = 1.0 / Math.sqrt(expectedTransformDimension);

        int indexOffset = 0;
        for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
            double norm = norms[vectorIndex];
            if (norm == 0.0) {
                for (int i = 0; i < originalDimension; i++) {
                    destination[destinationOffset + vectorIndex * originalDimension + i] = 0.0f;
                }
                indexOffset += expectedTransformDimension;
                continue;
            }

            for (int i = 0; i < expectedTransformDimension; i++) {
                transformed[i] = (float) (codebook.reconstruct(indices[indexOffset + i]) * inverseScale);
            }
            transform.inverse(transformed, 0, reconstructed, 0);
            int destinationVectorOffset = destinationOffset + vectorIndex * originalDimension;
            for (int i = 0; i < originalDimension; i++) {
                destination[destinationVectorOffset + i] = (float) (reconstructed[i] * norm);
            }
            indexOffset += expectedTransformDimension;
        }
    }

    private byte[] encodeMetadata(
            int originalDimension,
            int transformDimension,
            int vectorCount,
            double[] norms) {
        int metadataLength;
        try {
            metadataLength = Math.addExact(
                    METADATA_HEADER_BYTES, Math.multiplyExact(vectorCount, Double.BYTES));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("TurboQuant metadata exceeds supported array size", overflow);
        }
        ByteBuffer metadata = ByteBuffer.allocate(metadataLength).order(ByteOrder.BIG_ENDIAN);
        metadata.putInt(METADATA_MAGIC);
        metadata.putInt(METADATA_VERSION);
        metadata.putInt(bits);
        metadata.putInt(originalDimension);
        metadata.putInt(transformDimension);
        metadata.putInt(vectorCount);
        metadata.putLong(seed);
        for (double norm : norms) {
            metadata.putDouble(norm);
        }
        return metadata.array();
    }

    private double[] decodeMetadata(
            byte[] metadataBytes,
            int expectedOriginalDimension,
            int expectedTransformDimension,
            int expectedVectorCount) {
        Objects.requireNonNull(metadataBytes, "metadata");
        int expectedLength;
        try {
            expectedLength = Math.addExact(
                    METADATA_HEADER_BYTES, Math.multiplyExact(expectedVectorCount, Double.BYTES));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("expected TurboQuant metadata exceeds supported size", overflow);
        }
        if (metadataBytes.length != expectedLength) {
            throw new IllegalArgumentException("invalid TurboQuant metadata length");
        }

        ByteBuffer metadata = ByteBuffer.wrap(metadataBytes).order(ByteOrder.BIG_ENDIAN);
        if (metadata.getInt() != METADATA_MAGIC || metadata.getInt() != METADATA_VERSION) {
            throw new IllegalArgumentException("unsupported TurboQuant metadata format");
        }
        if (metadata.getInt() != bits) {
            throw new IllegalArgumentException("metadata bit width does not match codec configuration");
        }
        if (metadata.getInt() != expectedOriginalDimension
                || metadata.getInt() != expectedTransformDimension
                || metadata.getInt() != expectedVectorCount) {
            throw new IllegalArgumentException("metadata geometry does not match compression context");
        }
        if (metadata.getLong() != seed) {
            throw new IllegalArgumentException("metadata transform seed does not match codec configuration");
        }

        double[] norms = new double[expectedVectorCount];
        for (int i = 0; i < norms.length; i++) {
            double norm = metadata.getDouble();
            if (!Double.isFinite(norm) || norm < 0.0) {
                throw new IllegalArgumentException("metadata contains an invalid vector norm");
            }
            norms[i] = norm;
        }
        return norms;
    }

    private static int transformDimension(int originalDimension) {
        int required = Math.max(4, originalDimension);
        int dimension = Integer.highestOneBit(required);
        if (dimension != required) {
            if (dimension > (1 << 29)) {
                throw new IllegalArgumentException("head dimension is too large to pad safely");
            }
            dimension <<= 1;
        }
        return dimension;
    }

    private static void checkRange(int arrayLength, int offset, int count, String name) {
        if (offset < 0 || offset > arrayLength - count) {
            throw new IndexOutOfBoundsException(name + " does not contain the complete KV block");
        }
    }
}
