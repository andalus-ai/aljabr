package tech.kayys.aljabr.core.tensor.lazy;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.util.List;

/**
 * A Tensor implementation that defers computation by recording operations into
 * a GraphNode DAG.
 * When actual data is requested via eval() or array access, the DAG is
 * compiled,
 * optimized, and executed on the provided ComputeBackend.
 * @author bhangun
 */
public class LazyTensor implements Tensor {

    private final GraphNode node;
    private final ComputeBackend executionBackend;
    private final GraphExecutor executor;

    public LazyTensor(GraphNode node, ComputeBackend executionBackend) {
        this.node = node;
        this.executionBackend = executionBackend;
        this.executor = new GraphExecutor(executionBackend);
    }

    public GraphNode getNode() {
        return node;
    }

    @Override
    public Tensor eval() {
        if (!node.isEvaluated()) {
            GraphNode optimized = GraphOptimizer.optimize(node);
            executor.evaluate(optimized);
            node.setMaterializedData(optimized.getMaterializedData());
        }
        return node.getMaterializedData();
    }

    @Override
    public Shape shape() {
        return node.getOutputShape();
    }

    @Override
    public float[] toFloatArray() {
        return eval().toFloatArray();
    }

    @Override
    public DType dtype() {
        return node.getOutputDType();
    }

    @Override
    public DeviceType device() {
        if (node.isEvaluated()) {
            return node.getMaterializedData().device();
        }
        return DeviceType.CPU;
    }

    @Override
    public ComputeBackend backend() {
        return executionBackend;
    }

    @Override
    public float item() {
        return eval().item();
    }

    @Override
    public void backward() {
        eval().backward();
    }

    @Override
    public Tensor grad() {
        return eval().grad();
    }

    @Override
    public void setGrad(Tensor grad) {
        eval().setGrad(grad);
    }

    @Override
    public boolean requiresGrad() {
        return eval().requiresGrad();
    }

    @Override
    public void setRequiresGrad(boolean requiresGrad) {
        eval().setRequiresGrad(requiresGrad);
    }

    @Override
    public long numel() {
        return shape().numel();
    }

    // ── Math Operations (DAG recording)
    // ──────────────────────────────────────────────────────

    private LazyTensor op(OpType type, List<Tensor> inputs, Object[] args, Shape outShape, DType outDType) {
        List<GraphNode> inputNodes = inputs.stream()
                .map(t -> (t instanceof LazyTensor lt) ? lt.getNode() : GraphNode.constant(t))
                .toList();
        GraphNode newNode = new GraphNode(type, inputNodes, args, outShape, outDType);
        return new LazyTensor(newNode, executionBackend);
    }

    @Override
    public Tensor add(Tensor t) {
        return op(OpType.ADD, List.of(this, t), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor sub(Tensor t) {
        return op(OpType.SUB, List.of(this, t), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor mul(Tensor t) {
        return op(OpType.MUL, List.of(this, t), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor div(Tensor t) {
        return op(OpType.DIV, List.of(this, t), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor add(float scalar) {
        return op(OpType.ADD_SCALAR, List.of(this), new Object[] { scalar }, this.shape(), this.dtype());
    }

    @Override
    public Tensor mul(float scalar) {
        return op(OpType.MUL_SCALAR, List.of(this), new Object[] { scalar }, this.shape(), this.dtype());
    }

    @Override
    public Tensor div(float scalar) {
        return op(OpType.DIV_SCALAR, List.of(this), new Object[] { scalar }, this.shape(), this.dtype());
    }

    @Override
    public Tensor matmul(Tensor t) {
        Shape s = new Shape(this.shape().dim(this.shape().rank() - 2), t.shape().dim(t.shape().rank() - 1));
        return op(OpType.MATMUL, List.of(this, t), new Object[0], s, this.dtype());
    }

    @Override
    public Tensor reshape(long... newShape) {
        return op(OpType.RESHAPE, List.of(this), new Object[] { newShape }, new Shape(newShape), this.dtype());
    }

    @Override
    public Tensor relu() {
        return op(OpType.RELU, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor silu() {
        return op(OpType.SILU, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor crossEntropy(Tensor t) {
        return op(OpType.CROSS_ENTROPY, List.of(this, t), new Object[0], new Shape(1), this.dtype());
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor t) {
        return op(OpType.BINARY_CROSS_ENTROPY, List.of(this, t), new Object[0], new Shape(1), this.dtype());
    }

    @Override
    public Tensor dropout(float p, boolean training) {
        return op(OpType.DROPOUT, List.of(this), new Object[] { p, training }, this.shape(), this.dtype());
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
        return op(OpType.SLICE, List.of(this), new Object[] { offsets, sizes }, new Shape(sizes), this.dtype());
    }

    @Override
    public List<Tensor> split(int axis, int parts) {
        // Splitting a lazy tensor is tricky as it returns a list.
        // For simplicity, we might evaluate it first, but ideally we'd have a split
        // node.
        // We'll throw an exception for now as lazy lists are hard to represent without
        // a tuple type.
        throw new UnsupportedOperationException("Lazy split not fully supported yet.");
    }

    @Override
    public Tensor pow(float exponent) {
        return op(OpType.POW, List.of(this), new Object[] { exponent }, this.shape(), this.dtype());
    }

    @Override
    public Tensor mean() {
        return op(OpType.MEAN, List.of(this), new Object[0], new Shape(1), this.dtype());
    }

    @Override
    public Tensor abs() {
        return op(OpType.ABS, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor cast(DType dtype) {
        return op(OpType.CAST, List.of(this), new Object[] { dtype }, this.shape(), dtype);
    }

    @Override
    public Tensor to(DeviceType device) {
        return op(OpType.TO, List.of(this), new Object[] { device }, this.shape(), this.dtype());
    }

    @Override
    public Tensor zerosLike() {
        return op(OpType.ZEROS_LIKE, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor sqrt() {
        return op(OpType.SQRT, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor sigmoid() {
        return op(OpType.SIGMOID, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor tanh() {
        return op(OpType.TANH, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor log() {
        return op(OpType.LOG, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor exp() {
        return op(OpType.EXP, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor flatten() {
        return op(OpType.FLATTEN, List.of(this), new Object[0], new Shape(this.shape().numel()), this.dtype());
    }

    @Override
    public Tensor unsqueeze(int dim) {
        return op(OpType.UNSQUEEZE, List.of(this), new Object[] { dim }, this.shape() /* needs shape inference */,
                this.dtype());
    }

    @Override
    public Tensor squeeze() {
        return op(OpType.SQUEEZE, List.of(this), new Object[0], this.shape() /* needs shape inference */, this.dtype());
    }

    @Override
    public Tensor transpose() {
        Shape s = this.shape();
        if (s.rank() >= 2) {
            long[] dims = s.dims().clone();
            long tmp = dims[dims.length - 1];
            dims[dims.length - 1] = dims[dims.length - 2];
            dims[dims.length - 2] = tmp;
            s = new Shape(dims);
        }
        return op(OpType.TRANSPOSE, List.of(this), new Object[0], s, this.dtype());
    }

    @Override
    public Tensor transpose(int dim0, int dim1) {
        return op(OpType.TRANSPOSE, List.of(this), new Object[] { dim0, dim1 },
                this.shape() /* needs shape inference */, this.dtype());
    }

    @Override
    public Tensor gelu() {
        return op(OpType.GELU, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor softmax(int dim) {
        return op(OpType.SOFTMAX, List.of(this), new Object[] { dim }, this.shape(), this.dtype());
    }

    @Override
    public Tensor softmax() {
        return op(OpType.SOFTMAX, List.of(this), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor logSoftmax(int dim) {
        return op(OpType.LOG_SOFTMAX, List.of(this), new Object[] { dim }, this.shape(), this.dtype());
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        return op(OpType.MEAN, List.of(this), new Object[] { dim, keepDim }, this.shape() /* needs shape inference */,
                this.dtype());
    }

    @Override
    public Tensor sum() {
        return op(OpType.SUM, List.of(this), new Object[0], new Shape(1), this.dtype());
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        return op(OpType.SUM, List.of(this), new Object[] { dim, keepDim }, this.shape() /* needs shape inference */,
                this.dtype());
    }

    @Override
    public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        List<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(this);
        if (weight != null)
            inputs.add(weight);
        if (bias != null)
            inputs.add(bias);
        return op(OpType.LAYER_NORM, inputs, new Object[] { normalizedShape, eps }, this.shape(), this.dtype());
    }

    @Override
    public Tensor rmsNorm(Tensor weight, float eps) {
        List<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(this);
        if (weight != null)
            inputs.add(weight);
        return op(OpType.RMS_NORM, inputs, new Object[] { eps }, this.shape(), this.dtype());
    }

    @Override
    public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
            float momentum, float eps) {
        return op(OpType.BATCH_NORM, List.of(this, weight, bias, runningMean, runningVar),
                new Object[] { training, momentum, eps }, this.shape(), this.dtype());
    }

    @Override
    public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        List<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(this);
        if (weight != null)
            inputs.add(weight);
        if (bias != null)
            inputs.add(bias);
        return op(OpType.CONV2D, inputs, new Object[] { stride, padding, dilation, groups },
                this.shape() /* shape inference */, this.dtype());
    }

    @Override
    public Tensor maxPool2d(int kernelSize, int stride, int padding) {
        return op(OpType.MAX_POOL_2D, List.of(this), new Object[] { kernelSize, stride, padding },
                this.shape() /* shape inference */, this.dtype());
    }

    @Override
    public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
        return op(OpType.ADAPTIVE_AVG_POOL_2D, List.of(this), new Object[] { outputH, outputW },
                this.shape() /* shape inference */, this.dtype());
    }

    @Override
    public Tensor attention(Tensor K, Tensor V) {
        return op(OpType.ATTENTION, List.of(this, K, V), new Object[0], this.shape(), this.dtype());
    }

    @Override
    public Tensor embedding(Tensor weight, long paddingIdx) {
        return op(OpType.EMBEDDING, List.of(this, weight), new Object[] { paddingIdx },
                this.shape() /* shape inference */, weight.dtype());
    }

    @Override
    public Tensor applyRoPE(int posOffset, float freqBase, boolean isNeox) {
        return op(OpType.ROPE, List.of(this), new Object[] { posOffset, freqBase, isNeox }, this.shape(), this.dtype());
    }
}
