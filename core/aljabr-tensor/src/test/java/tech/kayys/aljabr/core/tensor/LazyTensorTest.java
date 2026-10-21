package tech.kayys.aljabr.core.tensor;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.lazy.GraphNode;
import tech.kayys.aljabr.core.tensor.lazy.LazyTensor;
import tech.kayys.aljabr.core.tensor.lazy.OpType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LazyTensorTest {

    @Test
    public void testLazyEvaluation() {
        class DummyTensor implements Tensor {
            private Shape s;

            public DummyTensor(long... dims) {
                this.s = new Shape(dims);
            }

            @Override
            public Tensor eval() {
                return this;
            }

            @Override
            public Shape shape() {
                return s;
            }

            @Override
            public DType dtype() {
                return DType.F32;
            }

            @Override
            public DeviceType device() {
                return DeviceType.CPU;
            }

            @Override
            public ComputeBackend backend() {
                return null;
            }

            @Override
            public float item() {
                return 0;
            }

            @Override
            public void backward() {
            }

            @Override
            public Tensor grad() {
                return null;
            }

            @Override
            public void setGrad(Tensor grad) {
            }

            @Override
            public boolean requiresGrad() {
                return false;
            }

            @Override
            public void setRequiresGrad(boolean requiresGrad) {
            }

            @Override
            public long numel() {
                return s.numel();
            }

            // Unimplemented math ops
            @Override
            public Tensor add(Tensor t) {
                return null;
            }

            @Override
            public Tensor sub(Tensor t) {
                return null;
            }

            @Override
            public Tensor mul(Tensor t) {
                return null;
            }

            @Override
            public Tensor div(Tensor t) {
                return null;
            }

            @Override
            public Tensor add(float scalar) {
                return null;
            }

            @Override
            public Tensor mul(float scalar) {
                return null;
            }

            @Override
            public Tensor div(float scalar) {
                return null;
            }

            @Override
            public Tensor matmul(Tensor t) {
                return null;
            }

            @Override
            public Tensor reshape(long... newShape) {
                return null;
            }

            @Override
            public Tensor relu() {
                return null;
            }

            @Override
            public Tensor silu() {
                return null;
            }

            @Override
            public Tensor crossEntropy(Tensor t) {
                return null;
            }

            @Override
            public Tensor binaryCrossEntropy(Tensor t) {
                return null;
            }

            @Override
            public Tensor dropout(float p, boolean training) {
                return null;
            }

            @Override
            public Tensor slice(long[] offsets, long[] sizes) {
                return null;
            }

            @Override
            public List<Tensor> split(int axis, int parts) {
                return null;
            }

            @Override
            public Tensor pow(float exponent) {
                return null;
            }

            @Override
            public Tensor mean() {
                return null;
            }

            @Override
            public Tensor abs() {
                return null;
            }

            @Override
            public Tensor cast(DType dtype) {
                return null;
            }

            @Override
            public Tensor to(DeviceType device) {
                return null;
            }

            @Override
            public Tensor zerosLike() {
                return null;
            }

            @Override
            public Tensor sqrt() {
                return null;
            }

            @Override
            public Tensor sigmoid() {
                return null;
            }

            @Override
            public Tensor tanh() {
                return null;
            }

            @Override
            public Tensor log() {
                return null;
            }

            @Override
            public Tensor exp() {
                return null;
            }

            @Override
            public Tensor flatten() {
                return null;
            }

            @Override
            public Tensor unsqueeze(int dim) {
                return null;
            }

            @Override
            public Tensor squeeze() {
                return null;
            }

            @Override
            public Tensor transpose() {
                return null;
            }

            @Override
            public Tensor transpose(int dim0, int dim1) {
                return null;
            }

            @Override
            public Tensor gelu() {
                return null;
            }

            @Override
            public Tensor softmax(int dim) {
                return null;
            }

            @Override
            public Tensor softmax() {
                return null;
            }

            @Override
            public Tensor logSoftmax(int dim) {
                return null;
            }

            @Override
            public Tensor mean(int dim, boolean keepDim) {
                return null;
            }

            @Override
            public Tensor sum() {
                return null;
            }

            @Override
            public Tensor sum(int dim, boolean keepDim) {
                return null;
            }

            @Override
            public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
                return null;
            }

            @Override
            public Tensor rmsNorm(Tensor weight, float eps) {
                return null;
            }

            @Override
            public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
                    float momentum, float eps) {
                return null;
            }

            @Override
            public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
                return null;
            }

            @Override
            public Tensor maxPool2d(int kernelSize, int stride, int padding) {
                return null;
            }

            @Override
            public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
                return null;
            }

            @Override
            public Tensor attention(Tensor K, Tensor V) {
                return null;
            }

            @Override
            public Tensor embedding(Tensor weight, long paddingIdx) {
                return null;
            }
        }

        Tensor t1 = new DummyTensor(2, 2);
        Tensor t2 = new DummyTensor(2, 2);
        Tensor t3 = new DummyTensor(2, 2);

        class DummyBackend implements ComputeBackend {
            public int callCount = 0;

            @Override
            public Tensor matmul(Tensor a, Tensor b) {
                callCount++;
                return t3;
            }

            // Add stub methods for other interfaces to compile
            @Override
            public Tensor add(Tensor a, Tensor b) {
                return null;
            }

            @Override
            public Tensor sub(Tensor a, Tensor b) {
                return null;
            }

            @Override
            public Tensor mul(Tensor a, float scalar) {
                return null;
            }

            @Override
            public Tensor mul(Tensor a, Tensor b) {
                return null;
            }

            @Override
            public Tensor div(Tensor a, float scalar) {
                return null;
            }

            @Override
            public Tensor div(Tensor a, Tensor b) {
                return null;
            }

            @Override
            public Tensor addScalar(Tensor a, float scalar) {
                return null;
            }

            @Override
            public Tensor reshape(Tensor a, long... newShape) {
                return null;
            }

            @Override
            public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
                return null;
            }

            @Override
            public List<Tensor> split(Tensor a, int axis, int parts) {
                return null;
            }

            @Override
            public Tensor attention(Tensor Q, Tensor K, Tensor V) {
                return null;
            }

            @Override
            public Tensor softmax(Tensor a) {
                return null;
            }

            @Override
            public Tensor pow(Tensor a, float exponent) {
                return null;
            }

            @Override
            public Tensor mean(Tensor a) {
                return null;
            }

            @Override
            public Tensor abs(Tensor a) {
                return null;
            }

            @Override
            public Tensor crossEntropy(Tensor pred, Tensor target) {
                return null;
            }

            @Override
            public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
                return null;
            }

            @Override
            public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
                return null;
            }

            @Override
            public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
                return null;
            }

            @Override
            public Tensor zerosLike(Tensor a) {
                return null;
            }

            @Override
            public Tensor sqrt(Tensor a) {
                return null;
            }

            @Override
            public Tensor relu(Tensor a) {
                return null;
            }

            @Override
            public Tensor sigmoid(Tensor a) {
                return null;
            }

            @Override
            public Tensor tanh(Tensor a) {
                return null;
            }

            @Override
            public Tensor log(Tensor a) {
                return null;
            }

            @Override
            public Tensor exp(Tensor a) {
                return null;
            }

            @Override
            public Tensor silu(Tensor a) {
                return null;
            }

            @Override
            public Tensor flatten(Tensor a) {
                return null;
            }

            @Override
            public Tensor unsqueeze(Tensor a, int dim) {
                return null;
            }

            @Override
            public Tensor squeeze(Tensor a) {
                return null;
            }

            @Override
            public Tensor transpose(Tensor a) {
                return null;
            }

            @Override
            public Tensor transpose(Tensor a, int dim0, int dim1) {
                return null;
            }

            @Override
            public Tensor gelu(Tensor a) {
                return null;
            }

            @Override
            public Tensor softmax(Tensor a, int dim) {
                return null;
            }

            @Override
            public Tensor logSoftmax(Tensor a, int dim) {
                return null;
            }

            @Override
            public Tensor mean(Tensor a, int dim, boolean keepDim) {
                return null;
            }

            @Override
            public Tensor sum(Tensor a) {
                return null;
            }

            @Override
            public Tensor sum(Tensor a, int dim, boolean keepDim) {
                return null;
            }

            @Override
            public Tensor max(Tensor a) {
                return null;
            }

            @Override
            public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
                return null;
            }

            @Override
            public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
                return null;
            }

            @Override
            public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar,
                    boolean training, float momentum, float eps) {
                return null;
            }

            @Override
            public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation,
                    int groups) {
                return null;
            }

            @Override
            public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
                return null;
            }

            @Override
            public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
                return null;
            }

            @Override
            public Tensor dropout(Tensor input, float p, boolean training) {
                return null;
            }

            @Override
            public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
                return null;
            }

            @Override
            public long numel(Tensor a) {
                return 0;
            }
        }

        DummyBackend mockBackend = new DummyBackend();

        GraphNode n1 = GraphNode.constant(t1);
        LazyTensor lazy1 = new LazyTensor(n1, mockBackend);

        Tensor result = lazy1.matmul(t2);

        // Assert that result is lazy and not yet evaluated
        assertTrue(result instanceof LazyTensor);
        LazyTensor lazyResult = (LazyTensor) result;
        assertFalse(lazyResult.getNode().isEvaluated());
        assertEquals(OpType.MATMUL, lazyResult.getNode().getOp());

        // Backend should not have been called yet
        assertEquals(0, mockBackend.callCount);

        // Trigger evaluation
        Tensor evalResult = result.eval();

        // Now it should be evaluated
        assertTrue(lazyResult.getNode().isEvaluated());
        assertEquals(t3, evalResult);
        assertEquals(1, mockBackend.callCount);
    }
}
