package tech.kayys.aljabr.cache.quantization;

import tech.kayys.aljabr.cache.KVCacheCompressionCodec;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Reference TurboQuant product codec combining MSE reconstruction and QJL
 * residual correction for inner-product-oriented KV-cache compression.
 *
 * <p>A total width of {@code b} uses {@code b - 1} MSE index bits per
 * coordinate and one QJL sign bit per coordinate. The MSE stage reconstructs
 * the original-space vector first; QJL is then applied to the original-space
 * residual {@code source - mseReconstruction}, as specified by TurboQuant.</p>
 *
 * <p>The codec stores the nested MSE payload, a contiguous bit-packed QJL sign
 * stream, the nested MSE metadata, and one FP64 residual norm per token/head
 * vector. It is a CPU reference implementation, not a fused attention kernel.
 * The QJL projection is regenerated deterministically from the codec seed.</p>
 *
 * <p>Supported total widths are 2 through 9 bits because the current MSE
 * reference codebook supports 1 through 8 bits. The extra per-vector FP64
 * norms and format headers are metadata overhead beyond the nominal
 * {@code b} bits per coordinate.</p>
 */
public final class TurboQuantProdCodec implements KVCacheCompressionCodec {

    public static final String CODEC_ID = "turboquant-prod-reference";
    public static final int CODEC_VERSION = 1;

    private static final int PAYLOAD_MAGIC = 0x54515050; // "TQPP"
    private static final int METADATA_MAGIC = 0x5451504D; // "TQPM"
    private static final int FORMAT_VERSION = 1;
    private static final int PAYLOAD_HEADER_BYTES = 16;
    private static final int METADATA_HEADER_BYTES = 32;
    private static final long QJL_SEED_SALT = 0x514A4C2D54555242L;

    private final int bits;
    private final long seed;
    private final long qjlSeed;
    private final TurboQuantMseCodec mseCodec;

    public TurboQuantProdCodec(int bits, long seed) {
        if (bits < 2 || bits > 9) {
            throw new IllegalArgumentException("total bits must be between 2 and 9");
        }
        this.bits = bits;
        this.seed = seed;
        this.qjlSeed = mix64(seed ^ QJL_SEED_SALT);
        this.mseCodec = new TurboQuantMseCodec(bits - 1, seed);
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

        KVCacheEncodedBlock mseBlock = mseCodec.encode(source, sourceOffset, context);
        float[] mseReconstruction = new float[context.elementCount()];
        mseCodec.decode(mseBlock, mseReconstruction, 0, context);

        int dimension = context.headDimension();
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        int qjlBitsCount = Math.multiplyExact(vectorCount, dimension);
        int qjlBytes = TurboQuantIndexBitPacking.byteCount(qjlBitsCount, 1);
        int[] allSigns = new int[qjlBitsCount];
        double[] residualNorms = new double[vectorCount];
        TurboQuantQjlResidual qjl = new TurboQuantQjlResidual(dimension, qjlSeed);
        float[] residual = new float[dimension];

        int signOffset = 0;
        for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
            int vectorOffset = vectorIndex * dimension;
            for (int i = 0; i < dimension; i++) {
                residual[i] = source[sourceOffset + vectorOffset + i]
                        - mseReconstruction[vectorOffset + i];
            }
            TurboQuantQjlResidual.EncodedResidual encodedResidual = qjl.encode(residual, 0);
            residualNorms[vectorIndex] = encodedResidual.norm();
            byte[] packedVectorSigns = encodedResidual.packedSigns();
            int[] vectorSigns = TurboQuantIndexBitPacking.unpack(packedVectorSigns, dimension, 1);
            System.arraycopy(vectorSigns, 0, allSigns, signOffset, dimension);
            signOffset += dimension;
        }

        byte[] qjlPayload = TurboQuantIndexBitPacking.pack(allSigns, 1);
        if (qjlPayload.length != qjlBytes) {
            throw new IllegalStateException("unexpected QJL payload size");
        }
        byte[] payload = encodePayload(mseBlock.payload(), qjlPayload);
        byte[] metadata = encodeMetadata(
                dimension, vectorCount, qjlSeed, mseBlock.metadata(), residualNorms);
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

        int dimension = context.headDimension();
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        DecodedMetadata decodedMetadata = decodeMetadata(
                encoded.metadata(), dimension, vectorCount);
        DecodedPayload decodedPayload = decodePayload(encoded.payload(), vectorCount, dimension);

        KVCacheEncodedBlock mseBlock = new KVCacheEncodedBlock(
                mseCodec.id(),
                mseCodec.version(),
                context.elementCount(),
                decodedPayload.msePayload,
                decodedMetadata.mseMetadata);
        mseCodec.decode(mseBlock, destination, destinationOffset, context);

        int[] signs = TurboQuantIndexBitPacking.unpack(
                decodedPayload.qjlPayload, Math.multiplyExact(vectorCount, dimension), 1);
        TurboQuantQjlResidual qjl = new TurboQuantQjlResidual(dimension, qjlSeed);
        int signOffset = 0;
        for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
            int[] vectorSigns = new int[dimension];
            System.arraycopy(signs, signOffset, vectorSigns, 0, dimension);
            byte[] packedVectorSigns = TurboQuantIndexBitPacking.pack(vectorSigns, 1);
            float[] correction = qjl.decode(packedVectorSigns, decodedMetadata.residualNorms[vectorIndex]);
            int vectorOffset = destinationOffset + vectorIndex * dimension;
            for (int i = 0; i < dimension; i++) {
                float value = destination[vectorOffset + i] + correction[i];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("decoded TurboQuant value is outside the finite float range");
                }
                destination[vectorOffset + i] = value;
            }
            signOffset += dimension;
        }
    }

    private byte[] encodePayload(byte[] msePayload, byte[] qjlPayload) {
        int length;
        try {
            length = Math.addExact(
                    PAYLOAD_HEADER_BYTES, Math.addExact(msePayload.length, qjlPayload.length));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("TurboQuant Prod payload exceeds supported size", overflow);
        }
        ByteBuffer buffer = ByteBuffer.allocate(length).order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(PAYLOAD_MAGIC);
        buffer.putInt(FORMAT_VERSION);
        buffer.putInt(msePayload.length);
        buffer.putInt(qjlPayload.length);
        buffer.put(msePayload);
        buffer.put(qjlPayload);
        return buffer.array();
    }

    private DecodedPayload decodePayload(byte[] bytes, int vectorCount, int dimension) {
        Objects.requireNonNull(bytes, "payload");
        if (bytes.length < PAYLOAD_HEADER_BYTES) {
            throw new IllegalArgumentException("truncated TurboQuant Prod payload");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        if (buffer.getInt() != PAYLOAD_MAGIC || buffer.getInt() != FORMAT_VERSION) {
            throw new IllegalArgumentException("unsupported TurboQuant Prod payload format");
        }
        int mseLength = buffer.getInt();
        int qjlLength = buffer.getInt();
        int expectedQjlLength = TurboQuantIndexBitPacking.byteCount(
                Math.multiplyExact(vectorCount, dimension), 1);
        if (mseLength <= 0 || qjlLength != expectedQjlLength
                || (long) PAYLOAD_HEADER_BYTES + mseLength + qjlLength != bytes.length) {
            throw new IllegalArgumentException("invalid TurboQuant Prod payload lengths");
        }
        byte[] msePayload = new byte[mseLength];
        byte[] qjlPayload = new byte[qjlLength];
        buffer.get(msePayload);
        buffer.get(qjlPayload);
        return new DecodedPayload(msePayload, qjlPayload);
    }

    private byte[] encodeMetadata(
            int dimension,
            int vectorCount,
            long projectionSeed,
            byte[] mseMetadata,
            double[] residualNorms) {
        int length;
        try {
            length = Math.addExact(
                    METADATA_HEADER_BYTES,
                    Math.addExact(mseMetadata.length, Math.multiplyExact(vectorCount, Double.BYTES)));
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("TurboQuant Prod metadata exceeds supported size", overflow);
        }
        ByteBuffer buffer = ByteBuffer.allocate(length).order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(METADATA_MAGIC);
        buffer.putInt(FORMAT_VERSION);
        buffer.putInt(bits);
        buffer.putInt(dimension);
        buffer.putInt(vectorCount);
        buffer.putInt(mseMetadata.length);
        buffer.putLong(projectionSeed);
        buffer.put(mseMetadata);
        for (double norm : residualNorms) {
            buffer.putDouble(norm);
        }
        return buffer.array();
    }

    private DecodedMetadata decodeMetadata(byte[] bytes, int dimension, int vectorCount) {
        Objects.requireNonNull(bytes, "metadata");
        if (bytes.length < METADATA_HEADER_BYTES) {
            throw new IllegalArgumentException("truncated TurboQuant Prod metadata");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        if (buffer.getInt() != METADATA_MAGIC || buffer.getInt() != FORMAT_VERSION) {
            throw new IllegalArgumentException("unsupported TurboQuant Prod metadata format");
        }
        if (buffer.getInt() != bits
                || buffer.getInt() != dimension
                || buffer.getInt() != vectorCount) {
            throw new IllegalArgumentException("TurboQuant Prod metadata geometry/configuration mismatch");
        }
        int mseMetadataLength = buffer.getInt();
        if (buffer.getLong() != qjlSeed) {
            throw new IllegalArgumentException("TurboQuant Prod QJL seed does not match codec configuration");
        }
        long expectedLength = (long) METADATA_HEADER_BYTES
                + mseMetadataLength + (long) vectorCount * Double.BYTES;
        if (mseMetadataLength <= 0 || expectedLength != bytes.length) {
            throw new IllegalArgumentException("invalid TurboQuant Prod metadata lengths");
        }
        byte[] mseMetadata = new byte[mseMetadataLength];
        buffer.get(mseMetadata);
        double[] residualNorms = new double[vectorCount];
        for (int i = 0; i < vectorCount; i++) {
            double norm = buffer.getDouble();
            if (!Double.isFinite(norm) || norm < 0.0) {
                throw new IllegalArgumentException("metadata contains an invalid QJL residual norm");
            }
            residualNorms[i] = norm;
        }
        return new DecodedMetadata(mseMetadata, residualNorms);
    }

    private static void checkRange(int arrayLength, int offset, int count, String name) {
        if (offset < 0 || offset > arrayLength - count) {
            throw new IndexOutOfBoundsException(name + " does not contain the complete KV block");
        }
    }

    private static long mix64(long value) {
        value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
        value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
        return value ^ (value >>> 31);
    }

    private record DecodedPayload(byte[] msePayload, byte[] qjlPayload) {
    }

    private record DecodedMetadata(byte[] mseMetadata, double[] residualNorms) {
    }
}
