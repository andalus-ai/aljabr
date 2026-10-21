package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;

/**
 * CROSS ENTROPY (LLM CORE)
 * Forward (simplified)
 * loss = -sum(target * log(prob))
 * Backward (key result)
 * dLogits = softmax(logits) - target
 * 
 * registry.register("cross_entropy", new CrossEntropyGrad());
 * registry.register("softmax", new SoftmaxGrad());
 * registry.register("layernorm", new LayerNormGrad());
 * registry.register("attention", new AttentionGrad());
 * @author bhangun
 */
public final class CrossEntropyGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef logits = op.inputs().get(0);
                GValueRef target = op.inputs().get(1);
                GValueId softmax = new GValueId(op.name() + "_softmax");
                GValueId dLogits = new GValueId(op.name() + "_dlogits");
                // softmax(logits)
                ctx.addOp(new GOp(
                                "softmax",
                                op.name() + "_softmax",
                                List.of(logits),
                                List.of(softmax),
                                Map.of()));
                // dLogits = softmax - target
                ctx.addOp(new GOp(
                                "sub",
                                op.name() + "_grad",
                                List.of(
                                                new GValueRef(softmax),
                                                target),
                                List.of(dLogits),
                                Map.of()));
                // scale by gradOut
                GValueId finalGrad = new GValueId(op.name() + "_scaled");
                ctx.addOp(new GOp(
                                "mul",
                                op.name() + "_scale",
                                List.of(
                                                new GValueRef(dLogits),
                                                new GValueRef(gradOut)),
                                List.of(finalGrad),
                                Map.of()));
                return Map.of(logits.id(), finalGrad);
        }
}