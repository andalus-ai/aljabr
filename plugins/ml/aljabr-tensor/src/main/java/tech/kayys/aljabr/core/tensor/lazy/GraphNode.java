package tech.kayys.aljabr.core.tensor.lazy;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.BackwardFunction;

import java.util.List;
import java.util.Objects;

/**
 * One value in a computation DAG. CONSTANT nodes reference caller-owned
 * materialized tensors; operation nodes describe forward work and an optional
 * local vector-Jacobian product.
 *
 * <p>Graph structure is immutable after construction. Materialization is
 * populated by the executor. This class does not own or release leaf tensors.</p>
 */
public final class GraphNode {

    private final OpType op;
    private final List<GraphNode> inputs;
    private final Object[] args;
    private final Shape outputShape;
    private final DType outputDType;
    private final BackwardFunction backwardFunction;
    private volatile Tensor materializedData;

    public GraphNode(OpType op, List<GraphNode> inputs, Object[] args, Shape outputShape, DType outputDType) {
        this(op, inputs, args, outputShape, outputDType, null);
    }

    public GraphNode(OpType op, List<GraphNode> inputs, Object[] args, Shape outputShape, DType outputDType,
            BackwardFunction backwardFunction) {
        this.op = Objects.requireNonNull(op, "op");
        this.inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
        this.args = Objects.requireNonNull(args, "args").clone();
        this.outputShape = Objects.requireNonNull(outputShape, "outputShape");
        this.outputDType = Objects.requireNonNull(outputDType, "outputDType");
        this.backwardFunction = backwardFunction;
        if (op == OpType.CONSTANT && !this.inputs.isEmpty()) {
            throw new IllegalArgumentException("A CONSTANT node cannot have inputs");
        }
    }

    public BackwardFunction getBackwardFunction() {
        return backwardFunction;
    }

    public boolean hasBackwardFunction() {
        return backwardFunction != null;
    }

    public static GraphNode constant(Tensor data) {
        Objects.requireNonNull(data, "data");
        GraphNode node = new GraphNode(OpType.CONSTANT, List.of(), new Object[0], data.shape(), data.dtype());
        node.setMaterializedData(data);
        return node;
    }

    public OpType getOp() {
        return op;
    }

    public List<GraphNode> getInputs() {
        return inputs;
    }

    /** Returns a defensive copy so callers cannot mutate recorded arguments. */
    public Object[] getArgs() {
        return args.clone();
    }

    public Shape getOutputShape() {
        return outputShape;
    }

    public DType getOutputDType() {
        return outputDType;
    }

    public Tensor getMaterializedData() {
        return materializedData;
    }

    public void setMaterializedData(Tensor materializedData) {
        this.materializedData = Objects.requireNonNull(materializedData, "materializedData");
    }

    public boolean isEvaluated() {
        return materializedData != null;
    }

    /**
     * True when this value is a differentiable leaf or depends on one.
     * Operation nodes inherit this property from their inputs.
     */
    public boolean requiresGrad() {
        if (op == OpType.CONSTANT) {
            Tensor data = materializedData;
            return data != null && data.requiresGrad();
        }
        return inputs.stream().anyMatch(GraphNode::requiresGrad);
    }
}
