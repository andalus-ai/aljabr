package tech.kayys.aljabr.core.tensor.lazy;

/**
 * Enumeration of operation types supported by the computation graph.
 * These map to methods on the ComputeBackend.
 * @author bhangun
 */
public enum OpType {
    // Structural
    CONSTANT,

    // Arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    ADD_SCALAR,
    MUL_SCALAR,
    DIV_SCALAR,

    // Matmul
    MATMUL,

    // Shape & Indexing
    RESHAPE,
    SLICE,
    SPLIT,
    FLATTEN,
    UNSQUEEZE,
    SQUEEZE,
    TRANSPOSE,

    // Activations
    RELU,
    SIGMOID,
    TANH,
    GELU,
    SILU,

    // Math
    POW,
    ABS,
    SQRT,
    EXP,
    LOG,

    // Reductions
    SUM,
    MEAN,
    MAX,

    // Normalization
    SOFTMAX,
    LOG_SOFTMAX,
    LAYER_NORM,
    RMS_NORM,
    BATCH_NORM,

    // NN Layers
    ATTENTION,
    CONV2D,
    MAX_POOL_2D,
    ADAPTIVE_AVG_POOL_2D,
    DROPOUT,
    EMBEDDING,
    ROPE,

    // Loss
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY,

    // Device & Types
    CAST,
    TO,
    ZEROS_LIKE
}
