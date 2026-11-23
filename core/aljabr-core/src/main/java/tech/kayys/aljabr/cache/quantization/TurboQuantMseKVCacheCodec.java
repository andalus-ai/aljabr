package tech.kayys.aljabr.cache.quantization;

import tech.kayys.aljabr.cache.KVCacheCompressionCodec;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Pure-Java reference TurboQuant MSE codec for contiguous KV-cache blocks.
 *
 * <p>Each token/head vector is encoded independently: its L2 norm is retained
 * as metadata, the normalized vector is zero-padded to a power-of-two length,
 * transformed with a seeded randomized Walsh-Hadamard transform, scaled by
 * the square root of the padded dimension, quantized with the spherical
 * Lloyd-Max codebook, and bit-packed.</p>
 *
 * <p>Decoding reverses those operations and crops padding. The encoded block
 * carries the seed, geometry, bit width, and one double-precision norm per
 * vector; decoding does not depend on mutable session or process-global state.</p>
 *
 * <p>This is the MSE path only. It does not implement QJL residual correction,
 * fused attention, backend kernels, or serving-time codec selection.</p>
 */
public final class TurboQuantMseKVCacheCodec implements KVCacheCompressionCodec {

    public static final String CODEC_ID = "turboquant-mse";
    public static final int CODEC_VERSION = 1;

    private static final int METADATA_MAGIC = 0x54514D53; // "TQMS"
    private static final int METADATA_VERSION = 1;
    private static final int METADATA_HEADER_BYTES = 52;

    private final int bits;
    private final long seed;
    private final ConcurrentMap<Integer, TurboQuantScalarCodebook> codebooks =
            new ConcurrentHashMap<>();

    /**
     * Creates a deterministic codec instance.
     *
     * @param bits scalar codebook width, from one through eight
     * @param seed randomized Hadamard transform seed, persisted per block
     */
    public TurboQuantMseKVCacheCodec(int bits, long seed) {
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }
        this.bits = bits;
        this.seed = seed;
    }

    @Override
    public String id() {
        return CODEC_ID;
    }

    @Override
    public int version() {
        return CODEC_VERSION;
    }

    public int bits() {
        return bits;
    }

    public long seed() {
        return seed;
    }

    @Override
    public KVCacheEncodedBlock encode(
            float[] source,
            int sourceOffset,
            KVCacheCompressionContext context) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(context, "context");
        checkRange(source.length, sourceOffset, context.elementCount(), "sourceOffset");

        int headDimension = context.headDimension();
        if (headDimension < 3) {
            throw new IllegalArgumentException("headDimension must be at least 3 for the spherical codebook");
        }
        int paddedDimension = paddedDimension(headDimension);
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        int elementCount = context.elementCount();
        int encodedIndexCount = Math.multiplyExact(vectorCount, paddedDimension);

        TurboQuantScalarCodebook codebook = codebooks.computeIfAbsent(
                paddedDimension, dimension -> TurboQuantScalarCodebook.forDimension(dimension, bits));
        TurboQuantHadamardTransform transform = new TurboQuantHadamardTransform(paddedDimension, seed);
        float[] vector = new float[paddedDimension];
        int[] indices = new int[encodedIndexCount];
        double[] norms = new double[vectorCount];
        double coordinateScale = Math.sqrt(paddedDimension);

        int sourceIndex = sourceOffset;
        int vectorIndex = 0;
        for (int tokenHead = 0; tokenHead < vectorCount; tokenHead++) {
            double squaredNorm = 0.0;
            for (int i = 0; i < headDimension; i++) {
                float value = source[sourceIndex + i];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("source values must be finite");
                }
                squaredNorm += (double) value * value;
            }

            double norm = Math.sqrt(squaredNorm);
            if (!Double.isFinite(norm)) {
                throw new IllegalArgumentException("source vector norm is not finite");
            }
            norms[tokenHead] = norm;
            for (int i = 0; i < paddedDimension; i++) {
                vector[i] = i < headDimension && norm != 0.0
                        ? (float) (source[sourceIndex + i] / norm)
                        : 0.0f;
            }

            transform.forward(vector, 0, vector, 0);
            for (int i = 0; i < paddedDimension; i++) {
                indices[vectorIndex++] = codebook.quantize(vector[i] * coordinateScale);
            }
            sourceIndex += headDimension;
        }

        byte[] payload = TurboQuantBitPacking.packIndices(indices, bits);
        byte[] metadata = encodeMetadata(context, paddedDimension, vectorCount, norms);
        return new KVCacheEncodedBlock(id(), version(), elementCount, payload, metadata);
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
            throw new IllegalArgumentException("encoded element count does not match KV block geometry");
        }
        checkRange(destination.length, destinationOffset, context.elementCount(), "destinationOffset");
        if (context.headDimension() < 3) {
            throw new IllegalArgumentException("headDimension must be at least 3 for the spherical codebook");
        }

        int headDimension = context.headDimension();
        int paddedDimension = paddedDimension(headDimension);
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        double[] norms = decodeMetadata(encoded.metadata(), context, paddedDimension, vectorCount);
        int encodedIndexCount = Math.multiplyExact(vectorCount, paddedDimension);
        int[] indices = TurboQuantBitPacking.unpackIndices(encoded.payload(), encodedIndexCount, bits);

        TurboQuantScalarCodebook codebook = codebooks.computeIfAbsent(
                paddedDimension, dimension -> TurboQuantScalarCodebook.forDimension(dimension, bits));
        TurboQuantHadamardTransform transform = new TurboQuantHadamardTransform(paddedDimension, seed);
        float[] transformed = new float[paddedDimension];
        float[] reconstructed = new float[context.elementCount()];
        double inverseScale = 1.0 / Math.sqrt(paddedDimension);

        int indexOffset = 0;
        int outputOffset = 0;
        for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
            for (int i = 0; i < paddedDimension; i++) {
                transformed[i] = (float) (codebook.reconstruct(indices[indexOffset++]) * inverseScale);
            }
            double norm = norms[vectorIndex];
            if (norm == 0.0d) {
                // Canonicalize zero vectors instead of allowing signed zero from
                // the inverse transform to leak into the decoded representation.
                for (int i = 0; i < headDimension; i++) {
                    reconstructed[outputOffset++] = 0.0f;
                }
                continue;
            }
            transform.inverse(transformed, 0, transformed, 0);

            for (int i = 0; i < headDimension; i++) {
                // A zero-norm input has no direction; publish canonical +0.0
                // rather than preserving a sign bit from quantized reconstruction.
                double value = norm == 0.0 ? 0.0 : transformed[i] * norm;
                if (!Double.isFinite(value) || Math.abs(value) > Float.MAX_VALUE) {
                    throw new IllegalArgumentException("decoded value exceeds finite float range");
                }
                // Canonicalize the zero-vector case so decode does not expose signed -0.0 values.
                reconstructed[outputOffset++] = norm == 0.0 ? 0.0f : (float) value;
            }
        }

        // Publish only after all metadata, payload, and reconstructed values validate.
        System.arraycopy(reconstructed, 0, destination, destinationOffset, reconstructed.length);
    }

    private byte[] encodeMetadata(
            KVCacheCompressionContext context,
            int paddedDimension,
            int vectorCount,
            double[] norms) {
        int normBytes = Math.multiplyExact(vectorCount, Double.BYTES);
        ByteBuffer metadata = ByteBuffer.allocate(Math.addExact(METADATA_HEADER_BYTES, normBytes))
                .order(ByteOrder.BIG_ENDIAN);
        metadata.putInt(METADATA_MAGIC);
        metadata.putInt(METADATA_VERSION);
        metadata.putInt(context.layer());
        metadata.putInt(context.firstPosition());
        metadata.putInt(context.tokenCount());
        metadata.putInt(context.kvHeads());
        metadata.putInt(context.headDimension());
        metadata.putInt(paddedDimension);
        metadata.putInt(tensorKindCode(context.tensorKind()));
        metadata.putInt(vectorCount);
        metadata.putInt(bits);
        metadata.putLong(seed);
        for (double norm : norms) {
            metadata.putDouble(norm);
        }
        return metadata.array();
    }

    private double[] decodeMetadata(
            byte[] encodedMetadata,
            KVCacheCompressionContext context,
            int expectedPaddedDimension,
            int expectedVectorCount) {
        int expectedBytes;
        try {
            expectedBytes = Math.addExact(
                    METADATA_HEADER_BYTES, Math.multiplyExact(expectedVectorCount, Double.BYTES));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("KV metadata size exceeds supported indexing", overflow);
        }
        if (encodedMetadata.length != expectedBytes) {
            throw new IllegalArgumentException("encoded metadata has an invalid length");
        }

        ByteBuffer metadata = ByteBuffer.wrap(encodedMetadata).order(ByteOrder.BIG_ENDIAN);
        if (metadata.getInt() != METADATA_MAGIC || metadata.getInt() != METADATA_VERSION) {
            throw new IllegalArgumentException("encoded metadata has an unsupported format or version");
        }
        if (metadata.getInt() != context.layer()
                || metadata.getInt() != context.firstPosition()
                || metadata.getInt() != context.tokenCount()
                || metadata.getInt() != context.kvHeads()
                || metadata.getInt() != context.headDimension()
                || metadata.getInt() != expectedPaddedDimension
                || metadata.getInt() != tensorKindCode(context.tensorKind())
                || metadata.getInt() != expectedVectorCount
                || metadata.getInt() != bits
                || metadata.getLong() != seed) {
            throw new IllegalArgumentException("encoded metadata does not match codec configuration or KV geometry");
        }

        double[] norms = new double[expectedVectorCount];
        for (int i = 0; i < norms.length; i++) {
            double norm = metadata.getDouble();
            if (!Double.isFinite(norm) || norm < 0.0) {
                throw new IllegalArgumentException("encoded vector norm is invalid at index " + i);
            }
            norms[i] = norm;
        }
        return norms;
    }

    private static int tensorKindCode(KVCacheTensorKind tensorKind) {
        return switch (tensorKind) {
            case KEY -> 1;
            case VALUE -> 2;
        };
    }

    private static int paddedDimension(int dimension) {
        if (dimension <= 0 || dimension > (1 << 30)) {
            throw new IllegalArgumentException("headDimension is outside supported Hadamard dimensions");
        }
        if ((dimension & (dimension - 1)) == 0) {
            return dimension;
        }
        return Integer.highestOneBit(dimension - 1) << 1;
    }

    private static void checkRange(int arrayLength, int offset, int count, String name) {
        if (offset < 0 || count < 0 || offset > arrayLength - count) {
            throw new IndexOutOfBoundsException(name + " does not address the complete KV block");
        }
    }
}
