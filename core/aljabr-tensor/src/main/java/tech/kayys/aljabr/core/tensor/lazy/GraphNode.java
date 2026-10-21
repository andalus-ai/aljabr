package tech.kayys.aljabr.core.tensor.lazy;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.util.List;

/**
 * Represents a single node in the computation graph.
 * A node can either be a structural CONSTANT (representing a materialized
 * Tensor)
 * or an operation with dependencies (inputs) and arguments.
 * @author bhangun
 */
public class GraphNode {

    private final OpType op;
    private final List<GraphNode> inputs;
    private final Object[] args;
    private final Shape outputShape;
    private final DType outputDType;

    // Cached materialized tensor to avoid re-evaluation.
    private Tensor materializedData;

    public GraphNode(OpType op, List<GraphNode> inputs, Object[] args, Shape outputShape, DType outputDType) {
        this.op = op;
        this.inputs = inputs;
        this.args = args;
        this.outputShape = outputShape;
        this.outputDType = outputDType;
    }

    public static GraphNode constant(Tensor data) {
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

    public Object[] getArgs() {
        return args;
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
        this.materializedData = materializedData;
    }

    public boolean isEvaluated() {
        return materializedData != null;
    }
}
