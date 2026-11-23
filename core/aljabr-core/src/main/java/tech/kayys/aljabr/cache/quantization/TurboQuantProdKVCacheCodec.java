package tech.kayys.aljabr.cache.quantization;

import tech.kayys.aljabr.cache.KVCacheCompressionCodec;
import tech.kayys.aljabr.cache.KVCacheCompressionContext;
import tech.kayys.aljabr.cache.KVCacheEncodedBlock;
import tech.kayys.aljabr.cache.KVCacheTensorKind;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Pure-Java reference implementation of the TurboQuant-prod KV-cache codec.
 *
 * <p>For total width {@code bits >= 2}, the codec first applies the MSE
 * quantizer at {@code bits - 1} bits, then applies one-bit QJL to the residual
 * in the original vector coordinate space. At {@code bits == 1}, it uses the
 * paper's pure-QJL path with no MSE stage.</p>
 *
 * <p>The QJL residual projection is a dense Gaussian reference transform.
 * This implementation is intended for correctness and format validation,
 * not production throughput; it uses O(headDimension^2) projection state.</p>
 */
public final class TurboQuantProdKVCacheCodec implements KVCacheCompressionCodec {

    public static final String CODEC_ID = "turboquant-prod";
    public static final int CODEC_VERSION = 1;

    private static final int METADATA_MAGIC = 0x54515052; // "TQPR"
    private static final int METADATA_VERSION = 1;
    private static final int METADATA_HEADER_BYTES = 56;
    private static final long QJL_SEED_MIX = 0x6A09E667F3BCC909L;

    private final int bits;
    private final long seed;
    private final long qjlSeed;
    private final TurboQuantMseKVCacheCodec mseCodec;

    /**
     * @param bits total encoded width per coordinate, including the QJL bit
     * @param seed deterministic seed for both stages (with a domain-separated QJL seed)
     */
    public TurboQuantProdKVCacheCodec(int bits, long seed) {
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }
        this.bits = bits;
        this.seed = seed;
        this.qjlSeed = seed ^ QJL_SEED_MIX;
        this.mseCodec = bits >= 2 ? new TurboQuantMseKVCacheCodec(bits - 1, seed) : null;
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
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        int qjlBytesPerVector = packedSignBytes(headDimension);
        int qjlPayloadLength = Math.multiplyExact(vectorCount, qjlBytesPerVector);
        byte[] basePayload;
        byte[] baseMetadata;
        float[] baseReconstruction = new float[context.elementCount()];

        if (mseCodec != null) {
            KVCacheEncodedBlock base = mseCodec.encode(source, sourceOffset, context);
            basePayload = base.payload();
            baseMetadata = base.metadata();
            mseCodec.decode(base, baseReconstruction, 0, context);
        } else {
            basePayload = new byte[0];
            baseMetadata = new byte[0];
        }

        TurboQuantQjlTransform qjl = new TurboQuantQjlTransform(headDimension, qjlSeed);
        byte[] qjlPayload = new byte[qjlPayloadLength];
        double[] residualNorms = new double[vectorCount];
        float[] residual = new float[headDimension];
        int sourceIndex = sourceOffset;
        int baseIndex = 0;
        int qjlOffset = 0;

        for (int vector = 0; vector < vectorCount; vector++) {
            for (int i = 0; i < headDimension; i++) {
                float original = source[sourceIndex + i];
                if (!Float.isFinite(original)) {
                    throw new IllegalArgumentException("source values must be finite");
                }
                residual[i] = original - baseReconstruction[baseIndex + i];
                if (!Float.isFinite(residual[i])) {
                    throw new IllegalArgumentException("residual values must be finite");
                }
            }
            byte[] signs = qjl.quantize(residual, 0);
            residualNorms[vector] = qjl.residualNorm(residual, 0);
            System.arraycopy(signs, 0, qjlPayload, qjlOffset, signs.length);
            qjlOffset += signs.length;
            sourceIndex += headDimension;
            baseIndex += headDimension;
        }

        byte[] payload = concatenate(basePayload, qjlPayload);
        byte[] metadata = encodeMetadata(context, basePayload.length, baseMetadata, residualNorms);
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
            throw new IllegalArgumentException("encoded element count does not match KV block geometry");
        }
        checkRange(destination.length, destinationOffset, context.elementCount(), "destinationOffset");

        int headDimension = context.headDimension();
        int vectorCount = Math.multiplyExact(context.tokenCount(), context.kvHeads());
        int qjlBytesPerVector = packedSignBytes(headDimension);
        int qjlPayloadLength = Math.multiplyExact(vectorCount, qjlBytesPerVector);
        DecodedMetadata decodedMetadata = decodeMetadata(encoded.metadata(), context, vectorCount);
        int expectedPayloadLength;
        try {
            expectedPayloadLength = Math.addExact(decodedMetadata.basePayloadLength(), qjlPayloadLength);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("encoded payload length exceeds supported indexing", overflow);
        }
        byte[] payload = encoded.payload();
        if (payload.length != expectedPayloadLength) {
            throw new IllegalArgumentException("encoded payload has an invalid length");
        }
        if (mseCodec == null) {
            if (decodedMetadata.basePayloadLength() != 0 || decodedMetadata.baseMetadata().length != 0) {
                throw new IllegalArgumentException("pure-QJL block must not contain an MSE payload");
            }
        } else if (decodedMetadata.basePayloadLength() == 0 || decodedMetadata.baseMetadata().length == 0) {
            throw new IllegalArgumentException("TurboQuant-prod block is missing its MSE payload or metadata");
        }

        float[] reconstructed = new float[context.elementCount()];
        if (mseCodec != null) {
            byte[] basePayload = slice(payload, 0, decodedMetadata.basePayloadLength());
            KVCacheEncodedBlock base = new KVCacheEncodedBlock(
                    TurboQuantMseKVCacheCodec.CODEC_ID,
                    TurboQuantMseKVCacheCodec.CODEC_VERSION,
                    context.elementCount(),
                    basePayload,
                    decodedMetadata.baseMetadata());
            mseCodec.decode(base, reconstructed, 0, context);
        }

        TurboQuantQjlTransform qjl = new TurboQuantQjlTransform(headDimension, qjlSeed);
        int payloadOffset = decodedMetadata.basePayloadLength();
        int outputOffset = 0;
        float[] correction = new float[headDimension];
        for (int vector = 0; vector < vectorCount; vector++) {
            byte[] signs = slice(payload, payloadOffset, qjlBytesPerVector);
            // Validate canonical packed bits even for a zero-norm residual.
            TurboQuantBitPacking.unpackIndices(signs, headDimension, 1);
            qjl.reconstruct(signs, decodedMetadata.residualNorms()[vector], correction, 0);
            for (int i = 0; i < headDimension; i++) {
                double value = (double) reconstructed[outputOffset + i] + correction[i];
                if (!Double.isFinite(value) || Math.abs(value) > Float.MAX_VALUE) {
                    throw new IllegalArgumentException("decoded value exceeds finite float range");
                }
                reconstructed[outputOffset + i] = (float) value;
            }
            outputOffset += headDimension;
            payloadOffset += qjlBytesPerVector;
        }

        // Publish only after the complete block has passed validation and reconstruction.
        System.arraycopy(reconstructed, 0, destination, destinationOffset, reconstructed.length);
    }

    private byte[] encodeMetadata(
            KVCacheCompressionContext context,
            int basePayloadLength,
            byte[] baseMetadata,
            double[] residualNorms) {
        int normsBytes = Math.multiplyExact(residualNorms.length, Double.BYTES);
        int metadataLength;
        try {
            metadataLength = Math.addExact(
                    Math.addExact(METADATA_HEADER_BYTES, baseMetadata.length), normsBytes);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("encoded metadata exceeds supported indexing", overflow);
        }
        ByteBuffer metadata = ByteBuffer.allocate(metadataLength).order(ByteOrder.BIG_ENDIAN);
        metadata.putInt(METADATA_MAGIC);
        metadata.putInt(METADATA_VERSION);
        metadata.putInt(context.layer());
        metadata.putInt(context.firstPosition());
        metadata.putInt(context.tokenCount());
        metadata.putInt(context.kvHeads());
        metadata.putInt(context.headDimension());
        metadata.putInt(bits);
        metadata.putInt(tensorKindCode(context.tensorKind()));
        metadata.putLong(qjlSeed);
        metadata.putInt(basePayloadLength);
        metadata.putInt(baseMetadata.length);
        metadata.putInt(residualNorms.length);
        metadata.put(baseMetadata);
        for (double norm : residualNorms) {
            metadata.putDouble(norm);
        }
        return metadata.array();
    }

    private DecodedMetadata decodeMetadata(
            byte[] bytes,
            KVCacheCompressionContext context,
            int expectedVectorCount) {
        if (bytes.length < METADATA_HEADER_BYTES) {
            throw new IllegalArgumentException("encoded metadata is truncated");
        }
        ByteBuffer metadata = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        if (metadata.getInt() != METADATA_MAGIC || metadata.getInt() != METADATA_VERSION) {
            throw new IllegalArgumentException("encoded metadata has an unsupported format or version");
        }
        if (metadata.getInt() != context.layer()
                || metadata.getInt() != context.firstPosition()
                || metadata.getInt() != context.tokenCount()
                || metadata.getInt() != context.kvHeads()
                || metadata.getInt() != context.headDimension()
                || metadata.getInt() != bits
                || metadata.getInt() != tensorKindCode(context.tensorKind())
                || metadata.getLong() != qjlSeed) {
            throw new IllegalArgumentException("encoded metadata does not match codec configuration or KV geometry");
        }
        int basePayloadLength = metadata.getInt();
        int baseMetadataLength = metadata.getInt();
        int vectorCount = metadata.getInt();
        if (basePayloadLength < 0 || baseMetadataLength < 0 || vectorCount != expectedVectorCount) {
            throw new IllegalArgumentException("encoded metadata contains invalid nested lengths or vector count");
        }
        long expectedLength = (long) METADATA_HEADER_BYTES + baseMetadataLength
                + (long) vectorCount * Double.BYTES;
        if (expectedLength != bytes.length) {
            throw new IllegalArgumentException("encoded metadata has an invalid length");
        }
        byte[] baseMetadata = new byte[baseMetadataLength];
        metadata.get(baseMetadata);
        double[] residualNorms = new double[vectorCount];
        for (int i = 0; i < vectorCount; i++) {
            double norm = metadata.getDouble();
            if (!Double.isFinite(norm) || norm < 0.0) {
                throw new IllegalArgumentException("encoded residual norm is invalid at index " + i);
            }
            residualNorms[i] = norm;
        }
        return new DecodedMetadata(basePayloadLength, baseMetadata, residualNorms);
    }

    private static int tensorKindCode(KVCacheTensorKind tensorKind) {
        return switch (tensorKind) {
            case KEY -> 1;
            case VALUE -> 2;
        };
    }

    private static int packedSignBytes(int dimension) {
        return (int) (((long) dimension + 7L) >>> 3);
    }

    private static byte[] concatenate(byte[] left, byte[] right) {
        byte[] result = new byte[Math.addExact(left.length, right.length)];
        System.arraycopy(left, 0, result, 0, left.length);
        System.arraycopy(right, 0, result, left.length, right.length);
        return result;
    }

    private static byte[] slice(byte[] source, int offset, int length) {
        if (offset < 0 || length < 0 || offset > source.length - length) {
            throw new IllegalArgumentException("encoded payload section is out of bounds");
        }
        byte[] result = new byte[length];
        System.arraycopy(source, offset, result, 0, length);
        return result;
    }

    private static void checkRange(int arrayLength, int offset, int count, String name) {
        if (offset < 0 || count < 0 || offset > arrayLength - count) {
            throw new IndexOutOfBoundsException(name + " does not address the complete KV block");
        }
    }

    private record DecodedMetadata(
            int basePayloadLength,
            byte[] baseMetadata,
            double[] residualNorms) {
    }
}
