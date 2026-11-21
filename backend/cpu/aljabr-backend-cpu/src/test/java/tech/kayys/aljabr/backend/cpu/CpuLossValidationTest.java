package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class CpuLossValidationTest {

    private final CpuBackend backend = new CpuBackend();

    private DefaultTensor floats(float[] values, long... dimensions) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            buffer.segment().set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
        return new DefaultTensor(new Shape(dimensions), DType.F32, DeviceType.CPU, buffer, backend);
    }

    private DefaultTensor ints(int[] values, long... dimensions) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Integer.BYTES);
        for (int i = 0; i < values.length; i++) {
            buffer.segment().set(ValueLayout.JAVA_INT, (long) i * Integer.BYTES, values[i]);
        }
        return new DefaultTensor(new Shape(dimensions), DType.I32, DeviceType.CPU, buffer, backend);
    }

    @Test
    void crossEntropyRejectsOutOfRangeClassIndices() {
        DefaultTensor predictions = floats(new float[] { -0.2f, -1.6f, -2.4f, -0.1f, -0.8f, -2.0f }, 2, 3);
        DefaultTensor targets = ints(new int[] { 0, 3 }, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.crossEntropy(predictions, targets));
        } finally {
            predictions.release();
            targets.release();
        }
    }

    @Test
    void crossEntropyRejectsMismatchedBatchAndEmptyShapes() {
        DefaultTensor predictions = floats(new float[] { -0.2f, -1.6f, -2.4f }, 1, 3);
        DefaultTensor targets = ints(new int[] { 0, 1 }, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.crossEntropy(predictions, targets));
        } finally {
            predictions.release();
            targets.release();
        }

        DefaultTensor emptyPredictions = floats(new float[0], 0, 3);
        DefaultTensor emptyTargets = ints(new int[0], 0);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.crossEntropy(emptyPredictions, emptyTargets));
        } finally {
            emptyPredictions.release();
            emptyTargets.release();
        }
    }

    @Test
    void binaryCrossEntropyRejectsMismatchedShapesAndInvalidLabels() {
        DefaultTensor predictions = floats(new float[] { 0.25f, 0.75f }, 2);
        DefaultTensor mismatchedTargets = floats(new float[] { 0f, 1f }, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.binaryCrossEntropy(predictions, mismatchedTargets));
        } finally {
            predictions.release();
            mismatchedTargets.release();
        }

        DefaultTensor validShapePredictions = floats(new float[] { 0.25f, 0.75f }, 2);
        DefaultTensor invalidLabels = floats(new float[] { 0f, 0.5f }, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.binaryCrossEntropy(validShapePredictions, invalidLabels));
        } finally {
            validShapePredictions.release();
            invalidLabels.release();
        }
    }

    @Test
    void binaryCrossEntropyRejectsEmptyInputs() {
        DefaultTensor predictions = floats(new float[0], 0);
        DefaultTensor targets = floats(new float[0], 0);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> backend.binaryCrossEntropy(predictions, targets));
        } finally {
            predictions.release();
            targets.release();
        }
    }

    @Test
    void validLossInputsStillProduceScalarResults() {
        DefaultTensor logits = floats(new float[] { -0.2f, -1.6f, -2.4f }, 1, 3);
        DefaultTensor classIndex = ints(new int[] { 1 }, 1);
        Tensor crossEntropy = null;
        try {
            crossEntropy = backend.crossEntropy(logits, classIndex);
            assertArrayEquals(new float[] { 1.6f }, crossEntropy.toFloatArray(), 1e-6f);
        } finally {
            if (crossEntropy != null) crossEntropy.release();
            logits.release();
            classIndex.release();
        }

        DefaultTensor probabilities = floats(new float[] { 0.25f, 0.75f }, 2);
        DefaultTensor labels = floats(new float[] { 0f, 1f }, 2);
        Tensor binaryCrossEntropy = null;
        try {
            binaryCrossEntropy = backend.binaryCrossEntropy(probabilities, labels);
            assertEquals((float) (-(Math.log(0.75) + Math.log(0.75)) / 2.0),
                    binaryCrossEntropy.toFloatArray()[0], 1e-4f);
        } finally {
            if (binaryCrossEntropy != null) binaryCrossEntropy.release();
            probabilities.release();
            labels.release();
        }
    }
}
