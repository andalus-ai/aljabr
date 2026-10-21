package tech.kayys.aljabr.core.tensor.lazy;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.tensor.DType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GraphFusionTest {

    @Test
    public void testFlashAttentionFusion() {
        ComputeBackend backend = new NoOpBackend();

        Tensor Q = new MockTensor(new Shape(4, 4), DType.F32);
        Tensor K = new MockTensor(new Shape(4, 4), DType.F32);
        Tensor V = new MockTensor(new Shape(4, 4), DType.F32);

        // Wrap in lazy nodes
        GraphNode qNode = GraphNode.constant(Q);
        GraphNode kNode = GraphNode.constant(K);
        GraphNode vNode = GraphNode.constant(V);

        LazyTensor lazyQ = new LazyTensor(qNode, backend);
        LazyTensor lazyK = new LazyTensor(kNode, backend);
        LazyTensor lazyV = new LazyTensor(vNode, backend);

        // Perform attention manually: MATMUL(SOFTMAX(DIV_SCALAR(MATMUL(Q,
        // TRANSPOSE(K)))), V)
        Tensor kTransposed = lazyK.transpose();
        Tensor qk = lazyQ.matmul(kTransposed);
        Tensor qkScaled = qk.div((float) Math.sqrt(4.0));
        Tensor attentionWeights = qkScaled.softmax();
        LazyTensor output = (LazyTensor) attentionWeights.matmul(lazyV);

        // The graph should currently be deep (Matmul -> Softmax -> Div -> Matmul -> Q,
        // K', V)
        assertEquals(OpType.MATMUL, output.getNode().getOp());

        // Run optimizer
        GraphNode optimized = GraphOptimizer.optimize(output.getNode());

        // Verify it fused to ATTENTION(Q, K, V)
        assertEquals(OpType.ATTENTION, optimized.getOp(), "The graph was not correctly fused into an ATTENTION node");
        assertEquals(3, optimized.getInputs().size());
        assertEquals(qNode, optimized.getInputs().get(0));

        // Note: The K input here actually points to K, not TRANSPOSE(K), because the
        // fusion rule explicitly extracts K.
        assertEquals(kNode, optimized.getInputs().get(1));
        assertEquals(vNode, optimized.getInputs().get(2));

        // Since we are using NoOpBackend, evaluating might return null or mock tensor.
        // Graph optimization validation is sufficient here.
    }
}

// Simple NoOpBackend for testing
class NoOpBackend implements ComputeBackend {
    @Override
    public Tensor add(Tensor a, Tensor b) {
        return null;
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        return null;
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
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
    public Tensor mul(Tensor a, float scalar) {
        return null;
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return null;
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        return null;
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return null;
    }

    @Override
    public Tensor relu(Tensor a) {
        return null;
    }

    @Override
    public Tensor silu(Tensor a) {
        return null;
    }

    @Override
    public Tensor gelu(Tensor a) {
        return null;
    }

    @Override
    public Tensor softmax(Tensor a) {
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
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        return null;
    }

    @Override
    public java.util.List<Tensor> split(Tensor a, int axis, int parts) {
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
    public Tensor abs(Tensor a) {
        return null;
    }

    @Override
    public Tensor sqrt(Tensor a) {
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
    public Tensor transpose(Tensor a, int d0, int d1) {
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
    public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
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
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
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

    @Override
    public Tensor max(Tensor a) {
        return null;
    }
}

class MockTensor implements Tensor {
    private final Shape shape;
    private final DType dtype;

    public MockTensor(Shape shape, DType dtype) {
        this.shape = shape;
        this.dtype = dtype;
    }

    @Override
    public Tensor eval() {
        return this;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    @Override
    public tech.kayys.aljabr.core.tensor.DeviceType device() {
        return tech.kayys.aljabr.core.tensor.DeviceType.CPU;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public tech.kayys.aljabr.core.backend.ComputeBackend backend() {
        return new NoOpBackend();
    }

    @Override
    public Tensor add(Tensor other) {
        return null;
    }

    @Override
    public Tensor sub(Tensor other) {
        return null;
    }

    @Override
    public Tensor mul(Tensor other) {
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
    public Tensor matmul(Tensor other) {
        return null;
    }

    @Override
    public Tensor reshape(long... newShape) {
        return null;
    }

    @Override
    public Tensor softmax() {
        return null;
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
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
    public Tensor crossEntropy(Tensor target) {
        return null;
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor target) {
        return null;
    }

    @Override
    public Tensor div(Tensor other) {
        return null;
    }

    @Override
    public Tensor add(float scalar) {
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
    public Tensor cast(DType dtype) {
        return null;
    }

    @Override
    public Tensor to(tech.kayys.aljabr.core.tensor.DeviceType device) {
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
    public Tensor relu() {
        return null;
    }

    @Override
    public Tensor gelu() {
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
    public Tensor silu() {
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
    public Tensor softmax(int dim) {
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
    public java.util.List<Tensor> split(int axis, int parts) {
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
    public Tensor dropout(float p, boolean training) {
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

    @Override
    public long numel() {
        return shape.numel();
    }
}
