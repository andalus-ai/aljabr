package tech.kayys.aljabr.gguf.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class GgufDotTest {
    @Test
    void plainScalarDotsHandleOffsetsAndIndependentAccumulators() {
        int qBase = 6;
        int vBase = 4;
        byte[] quants = patternedQuants(qBase + 48);
        float[] vector = patternedVector(vBase + 48);

        assertEquals(
                referencePlain(16, quants, qBase, vector, vBase),
                GgufDot.dotSignedByte16Scalar(quants, qBase, vector, vBase),
                1.0e-5f);
        assertEquals(
                referencePlain(32, quants, qBase + 5, vector, vBase + 7),
                GgufDot.dotSignedByte32Scalar(quants, qBase + 5, vector, vBase + 7),
                1.0e-5f);
    }

    @Test
    void blockScalarDotsHandleWideBlocksOffsetsAndTails() {
        int qBase = 9;
        int vBase = 11;
        byte[] quants = patternedQuants(qBase + 256);
        float[] vector = patternedVector(vBase + 256);

        assertEquals(
                referencePlain(256, quants, qBase, vector, vBase),
                GgufDot.dotSignedByteWideBlockScalar(quants, qBase, vector, vBase, 256),
                1.0e-4f);
        assertEquals(
                referencePlain(37, quants, qBase + 4, vector, vBase + 6),
                GgufDot.dotSignedByteBlockScalar(quants, qBase + 4, vector, vBase + 6, 37),
                1.0e-5f);
    }

    @Test
    void affineScalarDotsHandleOffsetsAndIndependentAccumulators() {
        int qBase = 5;
        int vBase = 7;
        byte[] quants = patternedQuants(qBase + 48);
        float[] vector = patternedVector(vBase + 48);

        assertEquals(
                referenceAffine(16, quants, qBase, vector, vBase, -0.75f, 1.25f),
                GgufDot.dotSignedByte16AffineScalar(quants, qBase, vector, vBase, -0.75f, 1.25f),
                1.0e-5f);
        assertEquals(
                referenceAffine(32, quants, qBase + 3, vector, vBase + 5, 1.5f, -0.5f),
                GgufDot.dotSignedByte32AffineScalar(quants, qBase + 3, vector, vBase + 5, 1.5f, -0.5f),
                1.0e-5f);
    }

    @Test
    void scaledBlockVectorDotsHandleOffsetsUnrolledBlocksAndTail() {
        int blocks = 5;
        int qBase = 5;
        int scaleBase = 2;
        byte[] quants = patternedQuants(qBase + blocks * 32);
        float[] scales = patternedScales(scaleBase + blocks);
        float[] vector = patternedVector(blocks * 32);

        assertEquals(
                referenceScaledBlocks(16, blocks, quants, qBase, scales, scaleBase, vector),
                GgufDot.dotSignedByte16ScaledBlocksVector(blocks, quants, qBase, scales, scaleBase, vector),
                1.0e-3f);
        assertEquals(
                referenceScaledBlocks(32, blocks, quants, qBase, scales, scaleBase, vector),
                GgufDot.dotSignedByte32ScaledBlocksVector(blocks, quants, qBase, scales, scaleBase, vector),
                1.0e-3f);
    }

    @Test
    void affineBlockVectorDotsHandleOffsetsUnrolledBlocksAndTail() {
        int blocks = 5;
        int qBase = 7;
        int biasBase = 3;
        float biasScale = -0.625f;
        byte[] quants = patternedQuants(qBase + blocks * 32);
        float[] scales = patternedScales(biasBase + blocks);
        float[] biases = patternedBiases(biasBase + blocks);
        float[] vector = patternedVector(blocks * 32);

        assertEquals(
                referenceAffineBlocks(16, blocks, quants, qBase, scales, biases, biasBase, biasScale, vector),
                GgufDot.dotSignedByte16AffineBlocksVector(
                        blocks, quants, qBase, scales, biases, biasBase, biasScale, vector),
                1.0e-3f);
        assertEquals(
                referenceAffineBlocks(32, blocks, quants, qBase, scales, biases, biasBase, biasScale, vector),
                GgufDot.dotSignedByte32AffineBlocksVector(
                        blocks, quants, qBase, scales, biases, biasBase, biasScale, vector),
                1.0e-3f);
    }

    private static byte[] patternedQuants(int length) {
        byte[] quants = new byte[length];
        for (int index = 0; index < quants.length; index++) {
            quants[index] = (byte) (index * 37 - 91);
        }
        return quants;
    }

    private static float[] patternedVector(int length) {
        float[] vector = new float[length];
        for (int index = 0; index < vector.length; index++) {
            vector[index] = (index % 17 - 8) * 0.0625f;
        }
        return vector;
    }

    private static float[] patternedScales(int length) {
        float[] values = new float[length];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index % 5 - 2) * 0.375f + 0.25f;
        }
        return values;
    }

    private static float[] patternedBiases(int length) {
        float[] values = new float[length];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index % 7 - 3) * 0.21875f;
        }
        return values;
    }

    private static float referencePlain(int length, byte[] quants, int qBase, float[] vector, int vBase) {
        float sum = 0.0f;
        for (int index = 0; index < length; index++) {
            sum += quants[qBase + index] * vector[vBase + index];
        }
        return sum;
    }

    private static float referenceAffine(
            int length,
            byte[] quants,
            int qBase,
            float[] vector,
            int vBase,
            float scale,
            float bias) {
        float sum = 0.0f;
        for (int index = 0; index < length; index++) {
            sum += (scale * quants[qBase + index] + bias) * vector[vBase + index];
        }
        return sum;
    }

    private static float referenceScaledBlocks(
            int blockSize,
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            int scaleBase,
            float[] vector) {
        float sum = 0.0f;
        int vBase = 0;
        for (int block = 0; block < blocks; block++) {
            sum += scales[scaleBase + block] * referencePlain(blockSize, quants, qBase, vector, vBase);
            qBase += blockSize;
            vBase += blockSize;
        }
        return sum;
    }

    private static float referenceAffineBlocks(
            int blockSize,
            int blocks,
            byte[] quants,
            int qBase,
            float[] scales,
            float[] biases,
            int biasBase,
            float biasScale,
            float[] vector) {
        float sum = 0.0f;
        int vBase = 0;
        for (int block = 0; block < blocks; block++) {
            sum += referenceAffine(
                    blockSize,
                    quants,
                    qBase,
                    vector,
                    vBase,
                    scales[biasBase + block],
                    biases[biasBase + block] * biasScale);
            qBase += blockSize;
            vBase += blockSize;
        }
        return sum;
    }
}
