package tech.kayys.aljabr.core.tensor.lazy;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Topologically sorts and evaluates a GraphNode DAG.
 * @author bhangun
 */
public class GraphExecutor {

    private final ComputeBackend backend;

    public GraphExecutor(ComputeBackend backend) {
        this.backend = backend;
    }

    public Tensor evaluate(GraphNode root) {
        if (root.isEvaluated()) {
            return root.getMaterializedData();
        }

        List<GraphNode> topoOrder = new ArrayList<>();
        Set<GraphNode> visited = new HashSet<>();

        buildTopoOrder(root, visited, topoOrder);

        for (GraphNode node : topoOrder) {
            if (!node.isEvaluated()) {
                Tensor result = executeNode(node);
                node.setMaterializedData(result);
            }
        }

        return root.getMaterializedData();
    }

    /**
     * Executes reverse-mode differentiation for a materialized graph.
     *
     * <p>The supplied output gradient is borrowed. Gradients accumulated on
     * differentiable CONSTANT leaves become owned by those leaf tensors.
     * Intermediate gradient tensors are released by this executor.</p>
     *
     * @throws IllegalArgumentException when the seed shape does not match the output
     * @throws UnsupportedOperationException when a differentiable operation has no VJP
     */
    public void backward(GraphNode root, Tensor outputGradient) {
        if (!root.requiresGrad()) {
            return;
        }
        if (!root.getOutputShape().equals(outputGradient.shape())) {
            throw new IllegalArgumentException("Output gradient shape " + outputGradient.shape()
                    + " does not match graph output shape " + root.getOutputShape());
        }
        if (!root.isEvaluated()) {
            evaluate(root);
        }

        List<GraphNode> topoOrder = new ArrayList<>();
        buildTopoOrder(root, new HashSet<>(), topoOrder);
        java.util.IdentityHashMap<GraphNode, Tensor> gradients = new java.util.IdentityHashMap<>();

        try (AutogradContext.Scope ignored = AutogradContext.pauseRecording()) {
            gradients.put(root, outputGradient.add(0.0f));
            try {
                for (int index = topoOrder.size() - 1; index >= 0; index--) {
                    GraphNode node = topoOrder.get(index);
                    Tensor gradient = gradients.remove(node);
                    if (gradient == null) {
                        continue;
                    }

                    if (node.getOp() == OpType.CONSTANT) {
                        Tensor leaf = node.getMaterializedData();
                        if (leaf != null && leaf.requiresGrad()) {
                            Tensor previous = leaf.grad();
                            if (previous == null) {
                                leaf.setGrad(gradient);
                            } else {
                                Tensor accumulated = previous.add(gradient);
                                leaf.setGrad(accumulated);
                                gradient.release();
                            }
                        } else {
                            gradient.release();
                        }
                        continue;
                    }

                    if (!node.requiresGrad()) {
                        gradient.release();
                        continue;
                    }
                    if (!node.hasBackwardFunction()) {
                        gradient.release();
                        throw new UnsupportedOperationException(
                                "No autograd rule registered for operation " + node.getOp());
                    }

                    List<Tensor> inputValues = node.getInputs().stream()
                            .map(GraphNode::getMaterializedData)
                            .toList();
                    Tensor[] inputGradients;
                    try {
                        inputGradients = node.getBackwardFunction().apply(
                                node.getMaterializedData(), gradient, inputValues, node.getArgs());
                    } catch (RuntimeException | Error failure) {
                        gradient.release();
                        throw failure;
                    }
                    if (inputGradients == null || inputGradients.length != node.getInputs().size()) {
                        release(inputGradients);
                        gradient.release();
                        throw new IllegalStateException("Backward rule for " + node.getOp()
                                + " must return one gradient slot per input");
                    }

                    // Reject incompatible contributions before transferring any ownership.
                    for (int i = 0; i < inputGradients.length; i++) {
                        Tensor contribution = inputGradients[i];
                        if (contribution != null
                                && !node.getInputs().get(i).getOutputShape().equals(contribution.shape())) {
                            release(inputGradients);
                            gradient.release();
                            throw new IllegalArgumentException("Backward rule for " + node.getOp()
                                    + " returned gradient shape " + contribution.shape()
                                    + " for input shape " + node.getInputs().get(i).getOutputShape()
                                    + "; broadcasting gradients are not yet supported");
                        }
                    }

                    for (int i = 0; i < inputGradients.length; i++) {
                        Tensor contribution = inputGradients[i];
                        GraphNode input = node.getInputs().get(i);
                        if (contribution == null) {
                            continue;
                        }
                        if (input.requiresGrad()) {
                            accumulate(gradients, input, contribution);
                        } else {
                            contribution.release();
                        }
                    }
                    gradient.release();
                }
            } finally {
                gradients.values().forEach(Tensor::release);
                gradients.clear();
            }
        }
    }

    private static void accumulate(
            java.util.IdentityHashMap<GraphNode, Tensor> gradients,
            GraphNode node,
            Tensor contribution) {
        Tensor previous = gradients.get(node);
        if (previous == null) {
            gradients.put(node, contribution);
            return;
        }
        Tensor accumulated = previous.add(contribution);
        previous.release();
        contribution.release();
        gradients.put(node, accumulated);
    }

    private static void release(Tensor[] tensors) {
        if (tensors != null) {
            for (Tensor tensor : tensors) {
                if (tensor != null) {
                    tensor.release();
                }
            }
        }
    }

    private void buildTopoOrder(GraphNode node, Set<GraphNode> visited, List<GraphNode> topoOrder) {
        if (visited.contains(node))
            return;
        visited.add(node);

        for (GraphNode input : node.getInputs()) {
            buildTopoOrder(input, visited, topoOrder);
        }

        topoOrder.add(node);
    }

    private Tensor executeNode(GraphNode node) {
        List<GraphNode> inputs = node.getInputs();
        Object[] args = node.getArgs();

        return switch (node.getOp()) {
            case CONSTANT -> throw new IllegalStateException("CONSTANT node should already be evaluated.");

            // Arithmetic
            case ADD -> backend.add(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());
            case SUB -> backend.sub(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());
            case MUL -> backend.mul(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());
            case DIV -> backend.div(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());
            case ADD_SCALAR -> backend.addScalar(inputs.get(0).getMaterializedData(), (float) args[0]);
            case MUL_SCALAR -> backend.mul(inputs.get(0).getMaterializedData(), (float) args[0]);
            case DIV_SCALAR -> backend.div(inputs.get(0).getMaterializedData(), (float) args[0]);

            // Matmul
            case MATMUL -> backend.matmul(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());

            // Shape
            case RESHAPE -> backend.reshape(inputs.get(0).getMaterializedData(), (long[]) args[0]);
            case SLICE -> backend.slice(inputs.get(0).getMaterializedData(), (long[]) args[0], (long[]) args[1]);
            case CAT -> backend.cat(inputs.stream().map(GraphNode::getMaterializedData).toList(),
                    (int) args[0]);
            case FLATTEN -> backend.flatten(inputs.get(0).getMaterializedData());
            case UNSQUEEZE -> backend.unsqueeze(inputs.get(0).getMaterializedData(), (int) args[0]);
            case SQUEEZE -> backend.squeeze(inputs.get(0).getMaterializedData());
            case TRANSPOSE -> {
                if (args.length == 0)
                    yield backend.transpose(inputs.get(0).getMaterializedData());
                yield backend.transpose(inputs.get(0).getMaterializedData(), (int) args[0], (int) args[1]);
            }

            // Activations
            case RELU -> backend.relu(inputs.get(0).getMaterializedData());
            case SILU -> backend.silu(inputs.get(0).getMaterializedData());
            case SIGMOID -> backend.sigmoid(inputs.get(0).getMaterializedData());
            case TANH -> backend.tanh(inputs.get(0).getMaterializedData());
            case GELU -> backend.gelu(inputs.get(0).getMaterializedData());

            // Math
            case POW -> backend.pow(inputs.get(0).getMaterializedData(), (float) args[0]);
            case ABS -> backend.abs(inputs.get(0).getMaterializedData());
            case SQRT -> backend.sqrt(inputs.get(0).getMaterializedData());
            case EXP -> backend.exp(inputs.get(0).getMaterializedData());
            case LOG -> backend.log(inputs.get(0).getMaterializedData());

            // Reductions
            case SUM -> {
                if (args.length == 0)
                    yield backend.sum(inputs.get(0).getMaterializedData());
                yield backend.sum(inputs.get(0).getMaterializedData(), (int) args[0], (boolean) args[1]);
            }
            case MEAN -> {
                if (args.length == 0)
                    yield backend.mean(inputs.get(0).getMaterializedData());
                yield backend.mean(inputs.get(0).getMaterializedData(), (int) args[0], (boolean) args[1]);
            }
            case MAX -> backend.max(inputs.get(0).getMaterializedData());

            // Normalization
            case SOFTMAX -> {
                if (args.length == 0)
                    yield backend.softmax(inputs.get(0).getMaterializedData());
                yield backend.softmax(inputs.get(0).getMaterializedData(), (int) args[0]);
            }
            case LOG_SOFTMAX -> backend.logSoftmax(inputs.get(0).getMaterializedData(), (int) args[0]);
            case LAYER_NORM -> {
                Tensor weight = inputs.size() > 1 ? inputs.get(1).getMaterializedData() : null;
                Tensor bias = inputs.size() > 2 ? inputs.get(2).getMaterializedData() : null;
                yield backend.layerNorm(inputs.get(0).getMaterializedData(), (long[]) args[0], weight, bias,
                        (float) args[1]);
            }
            case RMS_NORM -> {
                Tensor weight = inputs.size() > 1 ? inputs.get(1).getMaterializedData() : null;
                yield backend.rmsNorm(inputs.get(0).getMaterializedData(), weight, (float) args[0]);
            }
            case BATCH_NORM -> backend.batchNorm(inputs.get(0).getMaterializedData(),
                    inputs.get(1).getMaterializedData(), inputs.get(2).getMaterializedData(),
                    inputs.get(3).getMaterializedData(), inputs.get(4).getMaterializedData(),
                    (boolean) args[0], (float) args[1], (float) args[2]);

            // NN Layers
            case ATTENTION -> backend.attention(inputs.get(0).getMaterializedData(),
                    inputs.get(1).getMaterializedData(), inputs.get(2).getMaterializedData());
            case CONV2D -> {
                Tensor weight = inputs.size() > 1 ? inputs.get(1).getMaterializedData() : null;
                Tensor bias = inputs.size() > 2 ? inputs.get(2).getMaterializedData() : null;
                yield backend.conv2d(inputs.get(0).getMaterializedData(), weight, bias, (int) args[0], (int) args[1],
                        (int) args[2], (int) args[3]);
            }
            case MAX_POOL_2D ->
                backend.maxPool2d(inputs.get(0).getMaterializedData(), (int) args[0], (int) args[1], (int) args[2]);
            case ADAPTIVE_AVG_POOL_2D ->
                backend.adaptiveAvgPool2d(inputs.get(0).getMaterializedData(), (int) args[0], (int) args[1]);
            case UPSAMPLE_NEAREST_2D ->
                backend.upsampleNearest2d(inputs.get(0).getMaterializedData(), (int) args[0], (int) args[1]);
            case DROPOUT -> backend.dropout(inputs.get(0).getMaterializedData(), (float) args[0], (boolean) args[1]);
            case EMBEDDING -> backend.embedding(inputs.get(1).getMaterializedData(),
                    inputs.get(0).getMaterializedData(), (long) args[0]);

            // Loss
            case CROSS_ENTROPY ->
                backend.crossEntropy(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());
            case BINARY_CROSS_ENTROPY ->
                backend.binaryCrossEntropy(inputs.get(0).getMaterializedData(), inputs.get(1).getMaterializedData());

            // Device & Types
            case CAST ->
                backend.cast(inputs.get(0).getMaterializedData(), (tech.kayys.aljabr.core.tensor.DType) args[0]);
            case TO -> backend.to(inputs.get(0).getMaterializedData(),
                    (tech.kayys.aljabr.core.tensor.DeviceType) args[0]);
            case ZEROS_LIKE -> backend.zerosLike(inputs.get(0).getMaterializedData());

            default ->
                throw new UnsupportedOperationException("GraphExecutor does not yet support op: " + node.getOp());
        };
    }
}
