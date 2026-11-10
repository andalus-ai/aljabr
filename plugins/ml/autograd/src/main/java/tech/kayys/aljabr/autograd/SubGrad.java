package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
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
public final class SubGrad implements GradFn {
    @Override
    public Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx) {
        GValueRef a = op.inputs().get(0);
        GValueRef b = op.inputs().get(1);
        GValueId db = new GValueId(op.name() + "_db");

        // da = gradOut
        // db = -gradOut
        ctx.addOp(new GOp("mul", op.name() + "_neg",
                List.of(new GValueRef(gradOut), new GValueRef(new GValueId("const_neg_1"))),
                List.of(db), Map.of()));

        return Map.of(a.id(), gradOut, b.id(), db);
    }
}
