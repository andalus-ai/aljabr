package tech.kayys.aljabr.autograd;

import java.util.*;
/**
 * 
 * Core class for tech module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class GradRegistry {
    private final Map<String, GradFn> map = new HashMap<>();

    // Singleton instance
    private static final GradRegistry INSTANCE = new GradRegistry();

    // Static initializer block - runs when class is first loaded
    static {
        INSTANCE.registerDefaultGradients();
    }

    private GradRegistry() {
    } // Private constructor

    public static GradRegistry getInstance() {
        return INSTANCE;
    }

    private void registerDefaultGradients() {
        register("attention", new AttentionGrad());
        register("matmul", new MatMulGrad());
        register("cross_entropy", new CrossEntropyGrad());
        register("softmax", new SoftmaxGrad());
        register("layernorm", new LayerNormGrad());

        // ADD THESE:
        register("gelu", new GeluGrad());
        register("add", new AddGrad());
        register("mul", new MulGrad());
        register("sub", new SubGrad());
        register("div", new DivGrad());
        register("relu", new ReluGrad());
        register("sigmoid", new SigmoidGrad());
    }

    public void register(String opType, GradFn fn) {
        map.put(opType, fn);
    }

    public GradFn get(String opType) {
        return map.get(opType);
    }

    public int size() {
        return map.size();
    }
}